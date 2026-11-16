// PkView —— 人机 PK（学以致用）：人与 RL 在同一格上"同步出招"，逐步对比动作与理由
//
// 每一步：
//   1. 玩家先声明"我在探索 / 我在寻优"，可选一个理由；
//   2. 按方向键（或点按钮）出招——RL 同时在玩家所在的同一格上用自己的 Q 表做 ε-greedy 决策；
//   3. 双方动作一起揭晓，按四种情况解释异同：
//      都寻优（一致 / 不一致）、都探索、一个探索一个寻优。
// 迷雾模式：玩家只能看到走过的格子和撞过的墙，与 RL 的信息条件相同，亲身体会"为什么必须探索"。

import { useCallback, useEffect, useMemo, useState } from "react"
import { Eye, EyeOff, RotateCcw, Swords } from "lucide-react"

import { MazeCanvas } from "@/components/MazeCanvas"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { detectBackend } from "@/lib/backend"
import { ACTION_ARROWS, ACTION_NAMES, MODE_STYLE } from "@/lib/decision"
import { pkPrepareLocal } from "@/lib/localEngine"
import {
  envStep, initialState, rlDecide, shortestPath, simulateRl, stateIndex,
  type PkEnvState, type RlDecision,
} from "@/lib/pk"
import { cn } from "@/lib/utils"
import type { HyperParams, MazeGrid, RewardConfigForm } from "@/types"

type Mode = "explore" | "exploit"

const REASONS: Record<Mode, string[]> = {
  exploit: ["离终点更近", "这条路走过，靠谱", "避开墙和陷阱", "看起来是最短路"],
  explore: ["没去过，想试试", "看看有没有近路", "随便走走碰碰运气"],
}

interface Opponent {
  label: string
  episode: number
  epsilon: number
  q_table: number[][]
}

interface PkData {
  opponents: Opponent[]
  bonus_cells: [number, number][]
  max_steps: number
  key: string // 生成这批对手时的迷宫 + 配置，变化后需重新准备
}

interface StepLog {
  no: number
  pos: [number, number]
  human: { a: number; mode: Mode; reason: string | null }
  rl: RlDecision
  outcome: string
  reward: number
}

interface Props {
  grid: MazeGrid
  reward: RewardConfigForm
  hyper: HyperParams
  seed: number
  active: boolean
}

const fmt = (v: number, d = 3) => (Math.abs(v) < 5e-5 ? "0" : v.toFixed(d))

/** 四种情况的对比解释 */
function explain(log: StepLog, fog: boolean): { title: string; body: string; tone: string } {
  const { human, rl } = log
  const hq = rl.q[human.a]
  const rq = rl.q[rl.a]
  const qMax = Math.max(...rl.q)
  const allZero = rl.q.every((v) => Math.abs(v) < 1e-9)
  const reason = human.reason ? `“${human.reason}”` : "（未填写）"
  const dice = `u=${fmt(rl.u)} ${rl.explore ? "<" : "≥"} ε=${fmt(rl.epsilon)}`
  const hName = `${ACTION_ARROWS[human.a]}${ACTION_NAMES[human.a]}`
  const rName = `${ACTION_ARROWS[rl.a]}${ACTION_NAMES[rl.a]}`

  if (human.mode === "exploit" && !rl.explore) {
    if (human.a === rl.a) {
      return {
        title: "✅ 都在寻优，选择一致",
        tone: "border-emerald-200 bg-emerald-50",
        body: allZero
          ? `不过 RL 在这一格的 Q 值全是 0（它其实没来过），只是碰巧和你选得一样。`
          : `你的依据是${reason}，RL 的依据是 Q 值最高（${rName} Q=${fmt(rq)}）。依据不同、结论相同——说明 RL 通过试错学到的“印象分”与人的判断吻合，它已经学会了这一格的最优走法。`,
      }
    }
    let body: string
    if (allZero) {
      body = `RL 在这一格的 Q 值全是 0（从没来过），它的“寻优”只能在并列方向里随机挑，谈不上依据；你的依据是${reason}。`
    } else if (Math.abs(qMax - hq) < 0.01) {
      body = `在 RL 看来，你选的 ${hName}（Q=${fmt(hq)}）和它选的 ${rName}（Q=${fmt(rq)}）几乎一样好——很可能两条都是最短路，殊途同归。`
    } else {
      body = `RL 认为 ${rName} 更好（Q=${fmt(rq)}），而你选的 ${hName} 在它眼里只值 Q=${fmt(hq)}。` +
        (fog
          ? "现在你和 RL 掌握的信息一样少，谁的估计更准，看最后谁先到终点。"
          : "可能的原因：你能看到整张地图（相当于手里有“地图模型”），RL 只能靠试错经验估计；也可能 RL 还没训练充分（看它训练了多少回合）。")
    }
    return { title: "🤔 都在寻优，选择不同", tone: "border-amber-200 bg-amber-50", body }
  }
  if (human.mode === "explore" && rl.explore) {
    return {
      title: "🎲 都在探索",
      tone: "border-orange-200 bg-orange-50",
      body: `探索就是有意尝试不确定的选择，所以动作${human.a === rl.a ? "碰巧相同" : "各不相同"}都很正常。你的理由是${reason}；RL 的理由是掷骰 ${dice}，于是从 4 个方向里等概率随机抽中 ${rName}。`,
    }
  }
  if (human.mode === "explore") {
    return {
      title: "🧭 你在探索，RL 在寻优",
      tone: "border-sky-200 bg-sky-50",
      body: `RL 的骰子落在寻优区（${dice}），按 Q 值走经验里最好的 ${rName}；你选择试试新路${reason}。如果新路真的更好，RL 也只有靠探索才能发现——这就是“探索与利用的权衡”：只寻优可能错过更好的路，只探索又浪费步数。`,
    }
  }
  return {
    title: "🎯 你在寻优，RL 在探索",
    tone: "border-sky-200 bg-sky-50",
    body: (allZero
      ? `RL 的骰子落在探索区（${dice}），何况它在这一格还没有任何经验（Q 值全是 0），于是随机抽中 ${rName}。`
      : rl.best.includes(rl.a) && rl.best.length < 4
        ? `RL 的骰子落在探索区（${dice}），随机抽一个方向——这次抽中的 ${rName} 恰好也是它 Q 值最高的方向，探索也可能“蒙对”。`
        : `RL 的骰子落在探索区（${dice}）：即使它知道 ${rl.best.map((b) => ACTION_ARROWS[b]).join("")} 方向的 Q 值最高，也会随机试一下，防止错过更好的路，于是抽中 ${rName}。`) +
      (fog ? "" : "你能看到地图，不需要靠试错——这正是人通常比 RL 学得快的原因：人有先验知识。"),
  }
}

export default function PkView({ grid, reward, hyper, seed, active }: Props) {
  const [data, setData] = useState<PkData | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [oppIdx, setOppIdx] = useState(2)
  const [fog, setFog] = useState(false)

  const [st, setSt] = useState<PkEnvState | null>(null)
  const [mode, setMode] = useState<Mode>("exploit")
  const [reason, setReason] = useState<string | null>(null)
  const [logs, setLogs] = useState<StepLog[]>([])
  const [trail, setTrail] = useState<[number, number][]>([])
  const [seen, setSeen] = useState<Set<string>>(new Set())

  const cfgKey = JSON.stringify({ grid, reward, hyper, seed })
  const stale = data != null && data.key !== cfgKey
  const opp = data?.opponents[oppIdx] ?? null
  const playing = st != null && !st.done

  const prepare = async () => {
    setLoading(true)
    setErr(null)
    try {
      if ((await detectBackend()) === "offline") {
        // 在线体验版：在浏览器中训练对手（算法与后端 /api/pk/prepare 一致）
        const d = pkPrepareLocal({ maze: grid, reward, episodes: hyper.episodes, learning_rate: hyper.learning_rate,
          gamma: hyper.gamma, epsilon_start: hyper.epsilon_start, epsilon_end: hyper.epsilon_end, seed })
        setData({ ...d, key: cfgKey })
        setSt(null)
        setLogs([])
        return
      }
      const resp = await fetch("/api/pk/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          maze: grid,
          reward_config: { ...reward },
          hyperparams: {
            episodes: hyper.episodes,
            learning_rate: hyper.learning_rate,
            gamma: hyper.gamma,
            epsilon_start: hyper.epsilon_start,
            epsilon_end: hyper.epsilon_end,
          },
          seed,
        }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}：${await resp.text()}`)
      const d = (await resp.json()) as Omit<PkData, "key">
      setData({ ...d, key: cfgKey })
      setSt(null)
      setLogs([])
    } catch (e) {
      setErr(e instanceof TypeError ? "无法连接后端服务，请先启动后端" : String(e instanceof Error ? e.message : e))
    } finally {
      setLoading(false)
    }
  }

  const startGame = () => {
    const s0 = initialState(grid)
    setSt(s0)
    setLogs([])
    setTrail([s0.pos])
    setSeen(new Set([`${s0.pos[0]},${s0.pos[1]}`]))
    setMode("exploit")
    setReason(null)
  }

  const act = useCallback((a: number) => {
    if (!data || !opp || !st || st.done) return
    const q = opp.q_table[stateIndex(grid, data.bonus_cells.length, st.pos, st.mask)]
    const rl = rlDecide(q, opp.epsilon)
    const res = envStep(grid, reward, data.bonus_cells, data.max_steps, st, a)
    setLogs((ls) => [...ls, {
      no: ls.length + 1, pos: st.pos, human: { a, mode, reason }, rl, outcome: res.outcome, reward: res.reward,
    }])
    setSt(res.next)
    setTrail((t) => [...t, res.next.pos])
    setSeen((s) => {
      const n = new Set(s)
      n.add(`${res.next.pos[0]},${res.next.pos[1]}`)
      if (res.hitWall) {
        const [dr, dc] = [[-1, 0], [1, 0], [0, -1], [0, 1]][a]
        n.add(`${st.pos[0] + dr},${st.pos[1] + dc}`) // 撞过的墙也"看见"了
      }
      return n
    })
    setReason(null)
  }, [data, opp, st, grid, reward, mode, reason])

  // 方向键出招（仅在本标签页可见且对局进行中）
  useEffect(() => {
    if (!active || !playing) return
    const keys: Record<string, number> = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 }
    const onKey = (e: KeyboardEvent) => {
      if (e.key in keys) {
        e.preventDefault()
        act(keys[e.key])
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [active, playing, act])

  const last = logs.length ? logs[logs.length - 1] : null
  const lastExplain = last ? explain(last, fog) : null

  // 结算
  const summary = useMemo(() => {
    if (!st?.done || !data || !opp) return null
    const cnt = { same: 0, diff: 0, bothExplore: 0, mixed: 0 }
    for (const l of logs) {
      if (l.human.mode === "exploit" && !l.rl.explore) cnt[l.human.a === l.rl.a ? "same" : "diff"]++
      else if (l.human.mode === "explore" && l.rl.explore) cnt.bothExplore++
      else cnt.mixed++
    }
    const humanExplore = logs.filter((l) => l.human.mode === "explore").length
    const rlExplore = logs.filter((l) => l.rl.explore).length
    const rlRun = simulateRl(grid, reward, data.bonus_cells, data.max_steps, opp.q_table, opp.epsilon)
    return { cnt, humanExplore, rlExplore, rlRun, shortest: shortestPath(grid) }
  }, [st?.done, data, opp, logs, grid, reward])

  const hidden = fog && st ? (r: number, c: number) => !seen.has(`${r},${c}`) : undefined

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="border-slate-200 bg-white">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base text-slate-900">
            <Swords className="h-4 w-4 text-brand" /> 人机 PK：你和 RL 同步出招
          </CardTitle>
          <p className="text-xs text-slate-500">
            每一步先选择“我在探索 / 我在寻优”和理由，再按方向键出招。RL 会在<b>同一格</b>上同时做出它的决策，双方一起揭晓、逐步对比。
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* 准备对手 */}
          <div className="space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" onClick={prepare} disabled={loading || playing}
                className="bg-brand text-white hover:bg-brand-dark">
                {loading ? "训练对手中…" : data && !stale ? "重新训练对手" : "① 训练 RL 对手"}
              </Button>
              <span className="text-xs text-slate-500">
                用当前迷宫和左侧 Q-Learning 参数训练，保存三个阶段的 RL 作为对手
              </span>
            </div>
            {stale && <p className="text-xs text-amber-700">迷宫或参数已修改，请重新训练对手。</p>}
            {err && <p className="text-xs text-rose-600">{err}</p>}
            {data && (
              <div className="flex flex-wrap gap-1.5">
                {data.opponents.map((o, i) => (
                  <button key={o.label} type="button" disabled={playing}
                    onClick={() => setOppIdx(i)}
                    className={cn("rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors disabled:opacity-60",
                      i === oppIdx ? "border-brand bg-white ring-1 ring-brand" : "border-slate-200 bg-white hover:bg-slate-100")}>
                    <div className="font-semibold text-slate-900">{o.label}</div>
                    <div className="text-slate-500">第 {o.episode} 回合 · ε = {o.epsilon}</div>
                  </button>
                ))}
              </div>
            )}
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <div className="flex items-center gap-2">
                <Switch id="fog" checked={fog} onCheckedChange={setFog} />
                <Label htmlFor="fog" className="flex items-center gap-1 text-xs text-slate-700">
                  {fog ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  迷雾模式（只看得见走过的格子，和 RL 一样）
                </Label>
              </div>
              <Button size="sm" variant="outline" onClick={startGame} disabled={!opp || stale}
                className="ml-auto border-slate-300 bg-white text-slate-800 hover:bg-slate-100">
                {st ? <><RotateCcw className="h-4 w-4" /> 再来一局</> : "② 开始 PK"}
              </Button>
            </div>
          </div>

          <MazeCanvas
            grid={grid}
            agentPos={st?.pos ?? null}
            trail={trail}
            agentMode={st ? mode : null}
            isHidden={hidden}
          />

          {playing && (
            <div className="space-y-2 rounded-md border border-slate-200 p-3">
              <div className="flex items-center gap-2">
                <span className="whitespace-nowrap text-xs text-slate-500">这一步我在：</span>
                {(["exploit", "explore"] as Mode[]).map((m) => (
                  <button key={m} type="button" onClick={() => { setMode(m); setReason(null) }}
                    className={cn("whitespace-nowrap rounded px-3 py-1 text-xs font-semibold transition-colors",
                      mode === m ? cn(MODE_STYLE[m].bg, "text-white") : "bg-slate-100 text-slate-600 hover:bg-slate-200")}>
                    {m === "exploit" ? "🎯 寻优" : "🎲 探索"}
                  </button>
                ))}
                <span className="ml-auto whitespace-nowrap text-xs text-slate-500">第 {st.steps + 1} 步 · 累计 {fmt(st.total, 2)}</span>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-slate-500">理由（可选）：</span>
                {REASONS[mode].map((r) => (
                  <button key={r} type="button" onClick={() => setReason(reason === r ? null : r)}
                    className={cn("rounded-full border px-2 py-0.5 text-xs transition-colors",
                      reason === r ? "border-ink bg-ink text-white" : "border-slate-300 bg-white text-slate-700 hover:bg-slate-100")}>
                    {r}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-3">
                <div className="grid grid-cols-3 gap-1">
                  <span />
                  <DirBtn a={0} onClick={act} />
                  <span />
                  <DirBtn a={2} onClick={act} />
                  <DirBtn a={1} onClick={act} />
                  <DirBtn a={3} onClick={act} />
                </div>
                <p className="text-xs text-slate-500">也可以直接按键盘方向键出招。RL 的选择会在你出招后同时揭晓。</p>
              </div>
            </div>
          )}

          {summary && st && opp && (
            <div className="space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm">
              <div className="font-semibold text-slate-900">
                🏁 对局结束：{st.success ? "你到达了终点！" : "未到达终点"}
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded border border-slate-200 bg-white p-2">
                  <div className="font-semibold text-slate-900">你</div>
                  <div>{st.steps} 步，累计奖励 {fmt(st.total, 2)}</div>
                  <div>探索 {summary.humanExplore} 步 / 寻优 {st.steps - summary.humanExplore} 步</div>
                </div>
                <div className="rounded border border-slate-200 bg-white p-2">
                  <div className="font-semibold text-slate-900">{opp.label}（自己从起点走 20 次）</div>
                  <div>成功率 {(summary.rlRun.successRate * 100).toFixed(0)}%，平均 {summary.rlRun.avgSteps.toFixed(1)} 步</div>
                  <div>平均累计奖励 {fmt(summary.rlRun.avgReward, 2)}</div>
                </div>
              </div>
              <p className="text-xs text-slate-600">
                最短路 {summary.shortest ?? "—"} 步。逐步对比中：都寻优且一致 {summary.cnt.same} 次、都寻优但不同 {summary.cnt.diff} 次、
                都探索 {summary.cnt.bothExplore} 次、一个探索一个寻优 {summary.cnt.mixed} 次。
                {fog
                  ? "迷雾中你也必须靠探索才能找到终点——这正是 RL 面对的处境。"
                  : "试试打开迷雾模式再来一局：看不到地图时，你会发现自己也不得不“探索”。"}
              </p>
              <p className="text-[11px] text-slate-400">计分规则与训练一致（不含势函数塑形的提示分）。</p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="space-y-4">
        <Card className="border-slate-200 bg-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-slate-900">这一步：你 vs RL</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {!last ? (
              <div className="space-y-2 text-slate-600">
                <p>出招后，这里会并排显示你和 RL 在同一格上的选择与理由，并解释四种情况：</p>
                <ul className="list-inside list-disc space-y-0.5 text-xs">
                  <li><b>都寻优、选择一致</b>：人的判断与 RL 学到的 Q 值吻合；</li>
                  <li><b>都寻优、选择不同</b>：双方对“哪条路最好”的估计不同，看看原因；</li>
                  <li><b>都探索</b>：探索带随机性，动作不同很正常；</li>
                  <li><b>一个探索、一个寻优</b>：体会“探索与利用的权衡”。</li>
                </ul>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <SideCard who="你" mode={last.human.mode} a={last.human.a}
                    reason={last.human.reason ?? "（未填写理由）"} />
                  <SideCard who={opp?.label ?? "RL"} mode={last.rl.explore ? "explore" : "exploit"} a={last.rl.a}
                    reason={last.rl.explore
                      ? `掷骰 u=${fmt(last.rl.u)} < ε=${fmt(last.rl.epsilon)}，随机抽一个方向`
                      : `掷骰 u=${fmt(last.rl.u)} ≥ ε=${fmt(last.rl.epsilon)}，选 Q 值最高的方向`} />
                </div>
                <div className="grid grid-cols-4 gap-1.5">
                  {last.rl.q.map((qv, a) => (
                    <div key={a} className={cn("rounded border px-1.5 py-1 text-center text-[11px]",
                      last.rl.best.includes(a) ? "border-sky-300 bg-sky-50" : "border-slate-200")}>
                      <div className="text-sm leading-none">{ACTION_ARROWS[a]}</div>
                      <div className="text-slate-500">Q = <b className="text-slate-800">{fmt(qv)}</b></div>
                      <div className="text-[10px] text-slate-500">
                        {a === last.human.a && "你"}{a === last.human.a && a === last.rl.a && " · "}{a === last.rl.a && "RL"}
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-[11px] text-slate-500">RL 在这一格（第 {last.no} 步所在位置）的 Q 值，即它对每个方向的“印象分”。</p>
                {lastExplain && (
                  <div className={cn("rounded border p-2.5", lastExplain.tone)}>
                    <div className="font-semibold text-slate-900">{lastExplain.title}</div>
                    <p className="mt-1 text-slate-700">{lastExplain.body}</p>
                  </div>
                )}
                <p className="text-xs text-slate-500">你这一步的结果：{last.outcome}，奖励 {fmt(last.reward)}。</p>
              </>
            )}
          </CardContent>
        </Card>

        {logs.length > 0 && (
          <Card className="border-slate-200 bg-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-base text-slate-900">逐步对照记录</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-72 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-white">
                    <tr className="border-b border-slate-200 text-left text-slate-500">
                      <th className="py-1 pr-2 font-medium">步</th>
                      <th className="py-1 pr-2 font-medium">你</th>
                      <th className="py-1 pr-2 font-medium">RL</th>
                      <th className="py-1 font-medium">情况</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...logs].reverse().map((l) => (
                      <tr key={l.no} className="border-b border-slate-100">
                        <td className="py-1 pr-2 text-slate-500">{l.no}</td>
                        <td className={cn("py-1 pr-2", MODE_STYLE[l.human.mode].text)}>
                          {ACTION_ARROWS[l.human.a]} {MODE_STYLE[l.human.mode].label}
                        </td>
                        <td className={cn("py-1 pr-2", MODE_STYLE[l.rl.explore ? "explore" : "exploit"].text)}>
                          {ACTION_ARROWS[l.rl.a]} {l.rl.explore ? "探索" : "寻优"}
                        </td>
                        <td className="py-1 text-slate-700">{explain(l, fog).title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

function DirBtn({ a, onClick }: { a: number; onClick: (a: number) => void }) {
  return (
    <button type="button" onClick={() => onClick(a)} aria-label={`向${ACTION_NAMES[a]}`}
      className="h-9 w-9 rounded-md border border-slate-300 bg-white text-base font-bold text-ink hover:bg-slate-100 active:bg-slate-200">
      {ACTION_ARROWS[a]}
    </button>
  )
}

function SideCard({ who, mode, a, reason }: { who: string; mode: Mode; a: number; reason: string }) {
  const s = MODE_STYLE[mode]
  return (
    <div className={cn("rounded border p-2", s.soft)}>
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-semibold text-slate-900">{who}</span>
        <span className={cn("rounded px-1.5 py-0.5 text-[11px] font-semibold text-white", s.bg)}>{s.label}</span>
        <span className="ml-auto text-lg leading-none text-ink">{ACTION_ARROWS[a]}</span>
      </div>
      <p className="mt-1 text-xs text-slate-700">{reason}</p>
    </div>
  )
}
