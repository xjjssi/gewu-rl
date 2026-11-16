// DecisionCard —— "决策解说卡"：把 Q-Learning 的一步决策拆成"人能看懂"的四个环节
//
//   ① 掷骰子：u 与 ε 比较 → 这一步是【探索】还是【寻优】
//   ② 怎么选：探索 = 四个方向都可能，各自概率多少；寻优 = 依据 Q 值（印象分）选最大的
//   ③ 学到了什么：把 TD 更新公式代入本步真实数字
//   ④ 换成人：同样的处境，人会怎么想
//
// 设计意图：RL 的 ε-greedy 决策与人"试试新路 / 走熟路"的两种选择一一对应，
// 让学生在看动画的同时理解每一步背后的理论（寓教于乐，趣味性与易懂性并存）。

import { cn } from "@/lib/utils"
import { ACTION_ARROWS, ACTION_NAMES, MODE_STYLE } from "@/lib/decision"
import { Cell, type MazeGrid, type StepDecision } from "@/types"

const DIRS: [number, number][] = [[-1, 0], [1, 0], [0, -1], [0, 1]]

const fmt = (v: number, d = 3) => (Math.abs(v) < 5e-5 ? "0" : v.toFixed(d))
const pct = (v: number) => `${(v * 100).toFixed(1)}%`

/** 某方向是否被墙 / 边界挡住（智能体事先并不知道，只有试了才知道） */
function blocked(grid: MazeGrid, pos: [number, number], a: number): boolean {
  const [r, c] = [pos[0] + DIRS[a][0], pos[1] + DIRS[a][1]]
  return r < 0 || r >= grid.length || c < 0 || c >= (grid[0]?.length ?? 0) || grid[r][c] === Cell.WALL
}

/** ε-greedy 下每个动作的总概率：π(a|s) = ε/|A| + (1−ε)·[a∈best]/|best| */
function policyProbs(d: StepDecision): number[] {
  return d.q.map((_, a) => d.e / 4 + (d.b.includes(a) ? (1 - d.e) / d.b.length : 0))
}

interface Props {
  grid: MazeGrid
  decision: StepDecision
  /** 决策时所在格 / 执行后到达格 */
  from: [number, number]
  to: [number, number]
  stepNo: number
  gamma: number
}

export function DecisionCard({ grid, decision: d, from, to, stepNo, gamma }: Props) {
  const explore = d.x === 1
  const mode = explore ? MODE_STYLE.explore : MODE_STYLE.exploit
  const probs = policyProbs(d)
  const hitWall = from[0] === to[0] && from[1] === to[1]
  const landed = grid[to[0]]?.[to[1]]
  const allZero = d.q.every((v) => Math.abs(v) < 1e-9)
  const tie = d.b.length > 1
  const delta = d.q1 - d.q0

  // 结果描述
  let outcome = "走到一个普通格子"
  if (hitWall) outcome = "撞墙，原地不动"
  else if (landed === Cell.GOAL) outcome = "到达终点！回合结束"
  else if (landed === Cell.TRAP) outcome = d.t ? "掉进陷阱，回合结束" : "踩到陷阱"
  else if (landed === Cell.BONUS) outcome = d.r > 0 ? "拿到奖励格" : "经过奖励格（已领过）"

  // ④ 换成人：根据情境给一句类比
  let human: string
  if (explore) {
    human = hitWall
      ? "就像在陌生的地方随手推一扇门——推不开，但记住了“这边走不通”。"
      : "就像刚到陌生城市，不急着走熟路，而是随便拐进一条街看看——也许能发现更近的路。"
  } else if (allZero) {
    human = "就像第一次走进迷宫：哪边都没走过、印象分全是 0，想“选最好的”也只能碰运气。"
  } else if (tie) {
    human = "就像有几条路给你的印象一样好，那就随便挑一条——这叫“随机打破平局”。"
  } else {
    human = "就像每天上学走熟悉的路：凭以往经验，选印象最好的那个方向。"
  }
  const learnText =
    delta > 1e-6
      ? "结果比原来预想的好 → 对这个方向的印象分上调，下次更愿意这么走。"
      : delta < -1e-6
        ? "结果比原来预想的差 → 印象分下调，下次少这么走。"
        : "结果与预想几乎一致 → 印象分基本不变。"

  return (
    <div className="space-y-3 text-sm text-slate-700">
      {/* 标题：第几步 + 模式徽标 */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500">第 {stepNo} 步决策</span>
        <span className={cn("rounded px-2 py-0.5 text-xs font-semibold text-white", mode.bg)}>
          {explore ? "🎲 探索" : "🎯 寻优"}
        </span>
        <span className="ml-auto text-xs text-slate-500">
          实际动作：<b className="text-slate-900">{ACTION_ARROWS[d.a]} {ACTION_NAMES[d.a]}</b>
          {"，"}选中概率 π = <b className="text-slate-900">{pct(d.p)}</b>
        </span>
      </div>

      {/* ① 掷骰子 */}
      <section className="space-y-1.5">
        <h4 className="text-xs font-semibold text-slate-900">① 掷骰子：探索还是寻优？</h4>
        <div className="relative h-4 w-full overflow-hidden rounded bg-sky-100">
          <div className="absolute inset-y-0 left-0 bg-orange-300" style={{ width: `${d.e * 100}%` }} />
          <div className="absolute inset-y-[-2px] w-[3px] rounded bg-ink" style={{ left: `calc(${d.u * 100}% - 1px)` }} />
        </div>
        <div className="flex justify-between text-[11px] text-slate-500">
          <span className="text-orange-700">← 探索区 [0, ε={fmt(d.e)})</span>
          <span className="text-sky-800">寻优区 [ε, 1) →</span>
        </div>
        <p>
          随机数 u = <b>{fmt(d.u)}</b> {explore ? "<" : "≥"} ε = <b>{fmt(d.e)}</b>，所以这一步
          <b className={mode.text}>【{mode.label}】</b>。
          <span className="text-slate-500">
            （本步探索的概率是 ε = {pct(d.e)}，寻优的概率是 1−ε = {pct(1 - d.e)}）
          </span>
        </p>
      </section>

      {/* ② 怎么选 */}
      <section className="space-y-1.5">
        <h4 className="text-xs font-semibold text-slate-900">
          ② {explore ? "探索：有哪些可能？" : "寻优：依据是什么？"}
        </h4>
        {explore ? (
          <p>
            不看 Q 值，从 4 个方向里<b>等概率随机抽一个</b>（每个 ε/4 = {pct(d.e / 4)}），本次抽中
            <b> {ACTION_ARROWS[d.a]}{ACTION_NAMES[d.a]}</b>。智能体看不到地图，不知道哪边是墙，只能试了才知道。
          </p>
        ) : (
          <p>
            看当前格子的 Q 值（对每个方向的“印象分”），选最大的
            {tie ? <>；有 <b>{d.b.length}</b> 个方向并列最高，从中随机挑一个</> : null}
            ，于是选择 <b>{ACTION_ARROWS[d.a]}{ACTION_NAMES[d.a]}</b>。
          </p>
        )}
        <div className="grid grid-cols-4 gap-1.5">
          {d.q.map((qv, a) => {
            const isBest = d.b.includes(a)
            const chosen = a === d.a
            const wall = blocked(grid, from, a)
            return (
              <div
                key={a}
                className={cn(
                  "rounded border px-1.5 py-1 text-center",
                  chosen ? cn("border-2", explore ? "border-orange-500" : "border-sky-700") : "border-slate-200",
                  isBest && !explore ? "bg-sky-50" : "bg-white",
                )}
              >
                <div className="text-base leading-none">{ACTION_ARROWS[a]}</div>
                <div className="mt-0.5 text-[11px] text-slate-500">Q = <b className="text-slate-800">{fmt(qv)}</b></div>
                <div className="text-[11px] text-slate-500">π = {pct(probs[a])}</div>
                {wall && <div className="text-[10px] text-rose-600">墙</div>}
                {isBest && <div className="text-[10px] text-sky-700">最高</div>}
              </div>
            )
          })}
        </div>
        <p className="text-[11px] text-slate-500">
          π 为 ε-greedy 策略下每个方向被选中的总概率：π(a) = ε/4 + (1−ε)·[a 是最高分]/并列数。
          {explore && d.b.length < 4 && d.b.includes(d.a) && " 本次随机抽到的恰好也是最高分方向——探索也可能“蒙对”。"}
        </p>
      </section>

      {/* ③ 学到了什么 */}
      <section className="space-y-1">
        <h4 className="text-xs font-semibold text-slate-900">③ 结果与学习（TD 更新）</h4>
        <p>
          结果：{outcome}，获得奖励 r = <b>{fmt(d.r)}</b>。
        </p>
        <p className="rounded bg-slate-50 px-2 py-1 font-mono text-[12px] text-slate-800">
          {d.t
            ? <>TD 目标 = r = {fmt(d.r)}（回合已结束，没有“以后”）</>
            : <>TD 目标 = r + γ·maxQ(s′) = {fmt(d.r)} + {gamma}×{fmt(d.nm)} = {fmt(d.tt)}</>}
          <br />
          Q({ACTION_ARROWS[d.a]}) ← {fmt(d.q0)} + {fmt(d.al, 3)}×({fmt(d.tt)} − {fmt(d.q0)}) = <b>{fmt(d.q1)}</b>
          {"  "}
          <span className={delta > 1e-6 ? "text-emerald-700" : delta < -1e-6 ? "text-rose-600" : "text-slate-500"}>
            ({delta >= 0 ? "+" : ""}{fmt(delta, 4)})
          </span>
        </p>
        <p className="text-slate-600">{learnText} 学习率 α = {fmt(d.al)} 表示每次只把印象分往新目标挪 {pct(d.al)}。</p>
      </section>

      {/* ④ 换成人 */}
      <section className={cn("rounded border px-2.5 py-2", mode.soft)}>
        <h4 className="text-xs font-semibold text-slate-900">④ 如果换成人</h4>
        <p className="mt-0.5">{human}</p>
      </section>
    </div>
  )
}

