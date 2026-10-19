// RewardDesigner —— AI 奖励设计师：自然语言意图 → 奖励配置 → 三组对照实验 → 意图达成度验证

import { useState } from "react"
import { Check, FlaskConical, KeyRound, Loader2, Sparkles, Wand2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import type {
  IntentExperimentResult,
  LlmConfig,
  MazeGrid,
  Preset,
  RewardConfigForm,
  RewardDesignResult,
} from "@/types"

// 示例意图：配上能体现差异的推荐迷宫与训练预算（点击后自动载入对应迷宫）。
// - 避陷：需要迷宫里有陷阱；默认惩罚通常已能避开，差异主要体现在有限预算下的学习效率；
// - 收集：需要迷宫里有奖励格且默认方案不太爱绕路去捡；
// - 快速：在大迷宫 + 有限预算下差异最明显（默认方案学不通，PBRS/AI 方案快速收敛）。
const EXAMPLES: { text: string; preset: string; episodes: number; note: string }[] = [
  {
    text: "让智能体学会绕开陷阱，宁愿绕远路也不冒险",
    preset: "multi_wall_8x8",
    episodes: 200,
    note: "推荐：多障碍 8×8（含陷阱格），已自动载入",
  },
  {
    text: "鼓励智能体收集奖励格，哪怕多花几步",
    preset: "single_wall_6x6",
    episodes: 200,
    note: "推荐：单障碍 6×6（含奖励格），已自动载入",
  },
  {
    text: "让智能体尽快到达终点，走最短路径，步数越少越好",
    preset: "big_maze_10x10",
    episodes: 120,
    note: "推荐：大迷宫 10×10 + 有限预算 120 回合（差异最明显），已自动载入",
  },
]

const REWARD_FIELD_LABELS: { key: keyof RewardConfigForm; label: string }[] = [
  { key: "step_penalty", label: "每步代价" },
  { key: "wall_penalty", label: "撞墙惩罚" },
  { key: "goal_reward", label: "终点奖励" },
  { key: "bonus_reward", label: "奖励格收益" },
  { key: "trap_penalty", label: "陷阱惩罚" },
  { key: "trap_terminal", label: "陷阱终止回合" },
  { key: "potential_shaping", label: "势函数塑形" },
]

interface MetricRow {
  key: "success_rate" | "avg_steps" | "avg_reward_common" | "bonus_collect_rate" | "trap_trigger_rate" | "episodes_to_converge"
  label: string
  pct?: boolean
  threshold?: number
}

const METRIC_ROWS: MetricRow[] = [
  { key: "success_rate", label: "成功率", pct: true, threshold: 0.05 },
  { key: "avg_steps", label: "平均步数", threshold: 0.5 },
  { key: "avg_reward_common", label: "统一标尺回报", threshold: 0.05 },
  { key: "episodes_to_converge", label: "收敛回合", threshold: 0.5 },
  { key: "bonus_collect_rate", label: "奖励格收集率", pct: true, threshold: 0.05 },
  { key: "trap_trigger_rate", label: "陷阱触发率", pct: true, threshold: 0.05 },
]

// 预设迷宫缓存（/api/presets 只需取一次）
let presetCache: Preset[] | null = null
async function loadPresets(): Promise<Preset[]> {
  if (!presetCache) {
    const r = await fetch("/api/presets")
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    presetCache = (await r.json()).presets
  }
  return presetCache!
}

interface Props {
  grid: MazeGrid
  onApplyReward: (reward: RewardConfigForm) => void
  onGoCompare: () => void
  onLoadMaze?: (grid: MazeGrid) => void
}

export default function RewardDesigner({ grid, onApplyReward, onGoCompare, onLoadMaze }: Props) {
  const [instruction, setInstruction] = useState("")
  const [showLlm, setShowLlm] = useState(false)
  const [llm, setLlm] = useState<LlmConfig>({
    base_url: "https://api.moonshot.cn/v1",
    api_key: "",
    model: "moonshot-v1-8k",
  })
  const [design, setDesign] = useState<RewardDesignResult | null>(null)
  const [experiment, setExperiment] = useState<IntentExperimentResult | null>(null)
  const [busy, setBusy] = useState<"design" | "experiment" | null>(null)
  const [applied, setApplied] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exampleNote, setExampleNote] = useState<string | null>(null)
  const [episodes, setEpisodes] = useState(200)
  const [nSeeds, setNSeeds] = useState(5)

  const llmPayload = () =>
    llm.api_key.trim() ? { llm_config: { base_url: llm.base_url, api_key: llm.api_key, model: llm.model } } : {}

  const pickExample = async (ex: (typeof EXAMPLES)[number]) => {
    setInstruction(ex.text)
    setEpisodes(ex.episodes)
    setExampleNote(ex.note)
    // 自动载入示例推荐迷宫，保证示例在意图对应的迷宫上演示
    if (onLoadMaze) {
      try {
        const presets = await loadPresets()
        const p = presets.find((x) => x.id === ex.preset)
        if (p) onLoadMaze(p.grid)
      } catch {
        setExampleNote(ex.note.replace("，已自动载入", "（迷宫载入失败，请手动在「环境搭建」中选择）"))
      }
    }
  }

  const runDesign = async () => {
    if (!instruction.trim() || busy) return
    setBusy("design")
    setError(null)
    setApplied(false)
    try {
      const r = await fetch("/api/reward-design", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruction, maze: grid, ...llmPayload() }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setDesign(await r.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const runExperiment = async () => {
    if (!instruction.trim() || busy) return
    setBusy("experiment")
    setError(null)
    try {
      const r = await fetch("/api/intent-experiment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction, maze: grid, algo: "qlearning",
          episodes, seed: 42, n_seeds: nSeeds, ...llmPayload(),
        }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = (await r.json()) as IntentExperimentResult
      setExperiment(data)
      setDesign(data.design)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  // 三组列：默认方案 / 默认+PBRS / AI 方案
  const groupOrder = ["default", "default_pbrs", "ai"]
  const groups = experiment
    ? groupOrder
        .map((k) => experiment.groups.find((g) => g.key === k))
        .filter((g): g is NonNullable<typeof g> => Boolean(g))
    : []

  const fmtCell = (key: MetricRow["key"], m: IntentExperimentResult["groups"][number]["metrics"], pct?: boolean) => {
    const v = m[key]
    if (v === null || v === undefined) return "—"
    const stdKey = `${key}_std` as const
    const std = (m as unknown as Record<string, number | null>)[stdKey]
    if (key === "episodes_to_converge") {
      const mean = `约 ${Math.round(v)}`
      return m.converged_seeds < m.n_seeds ? `${mean}（${m.converged_seeds}/${m.n_seeds} 收敛）` : mean
    }
    const mean = pct ? `${(v * 100).toFixed(0)}%` : v.toFixed(key === "avg_steps" ? 0 : 2)
    if (std === null || std === undefined || m.n_seeds <= 1) return mean
    const s = pct ? `${(std * 100).toFixed(0)}%` : std.toFixed(key === "avg_steps" ? 0 : 2)
    return `${mean}±${s}`
  }

  return (
    <div className="space-y-4">
      <Card className="border-slate-700 bg-slate-900/60">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base text-slate-100">
            <Wand2 className="h-4 w-4 text-violet-400" /> AI 奖励设计师
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs leading-relaxed text-slate-400">
            用中文描述教学目标，系统自动设计奖励函数，并运行三组对照实验验证意图是否达成：
            意图 → 奖励配置 →「默认方案 / 默认+PBRS / AI 方案」三组训练（多种子取均值±标准差）→ 行为指标对比。
            AI 的价值体现在第三组相对第二组的增量，而不是「PBRS 开 vs 关」。
          </p>

          <Textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="例如：让智能体学会绕开陷阱，宁愿绕远路也不冒险……"
            className="min-h-20 border-slate-600 bg-slate-800 text-slate-100"
          />
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <Button key={ex.text} size="sm" variant="outline"
                className="border-slate-600 bg-slate-800/60 text-xs text-slate-300 hover:bg-slate-700"
                onClick={() => pickExample(ex)}>
                {ex.text}
              </Button>
            ))}
          </div>
          {exampleNote && <p className="text-xs text-cyan-400">{exampleNote}</p>}

          <div className="flex flex-wrap items-end gap-3">
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">每组回合数</Label>
              <Input type="number" min={20} max={2000} value={episodes}
                onChange={(e) => setEpisodes(Math.max(20, Math.min(2000, Number(e.target.value) || 200)))}
                className="h-8 w-24 border-slate-600 bg-slate-800 text-slate-100" />
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">种子数（均值±标准差）</Label>
              <Input type="number" min={1} max={8} value={nSeeds}
                onChange={(e) => setNSeeds(Math.max(1, Math.min(8, Number(e.target.value) || 5)))}
                className="h-8 w-24 border-slate-600 bg-slate-800 text-slate-100" />
            </div>
          </div>

          <button
            onClick={() => setShowLlm(!showLlm)}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200"
          >
            <KeyRound className="h-3 w-3" /> LLM 设置 {showLlm ? "▲" : "▼"}
            {!llm.api_key && <span className="text-slate-500">（未配置 LLM，将使用内置规则引擎）</span>}
          </button>
          {showLlm && (
            <div className="grid gap-3 rounded-md border border-slate-700 bg-slate-800/60 p-3 sm:grid-cols-3">
              <div className="space-y-1">
                <Label className="text-xs text-slate-400">Base URL</Label>
                <Input value={llm.base_url} onChange={(e) => setLlm({ ...llm, base_url: e.target.value })}
                  className="h-8 border-slate-600 bg-slate-800 text-slate-100" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-slate-400">API Key</Label>
                <Input type="password" value={llm.api_key} placeholder="仅本次请求使用，不会保存"
                  onChange={(e) => setLlm({ ...llm, api_key: e.target.value })}
                  className="h-8 border-slate-600 bg-slate-800 text-slate-100" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-slate-400">模型</Label>
                <Input value={llm.model} onChange={(e) => setLlm({ ...llm, model: e.target.value })}
                  className="h-8 border-slate-600 bg-slate-800 text-slate-100" />
              </div>
            </div>
          )}

          <div className="flex gap-2">
            <Button onClick={runDesign} disabled={!instruction.trim() || busy !== null}
              className="flex-1 bg-violet-600 hover:bg-violet-500">
              {busy === "design" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              生成奖励方案
            </Button>
            <Button onClick={runExperiment} disabled={!instruction.trim() || busy !== null}
              className="flex-1 bg-cyan-600 hover:bg-cyan-500">
              {busy === "experiment" ? <Loader2 className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
              {busy === "experiment" ? `三组对照实验运行中（${nSeeds} 种子 × 3 组）…` : "运行意图验证实验"}
            </Button>
          </div>
          {error && <p className="text-sm text-rose-400">{error}</p>}
        </CardContent>
      </Card>

      {design && (
        <Card className="border-slate-700 bg-slate-900/60">
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-base text-slate-100">生成的奖励方案</CardTitle>
            <Badge className={design.source === "llm" ? "bg-violet-600" : "bg-slate-600"}>
              {design.source === "llm" ? "LLM 生成" : "规则引擎"}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {REWARD_FIELD_LABELS.map(({ key, label }) => (
                <div key={key} className="rounded-md border border-slate-700 bg-slate-800/60 px-3 py-2">
                  <div className="text-[11px] text-slate-400">{label}</div>
                  <div className="font-mono text-sm text-slate-100">
                    {typeof design.reward_config[key] === "boolean"
                      ? design.reward_config[key] ? "开" : "关"
                      : design.reward_config[key]}
                  </div>
                </div>
              ))}
              <div className="flex items-center">
                <Button size="sm" variant="outline"
                  className={cn("border-slate-600 text-slate-200",
                    applied ? "bg-emerald-700 hover:bg-emerald-700" : "bg-slate-800 hover:bg-slate-700")}
                  onClick={() => { onApplyReward(design.reward_config); setApplied(true) }}>
                  <Check className="h-4 w-4" /> {applied ? "已应用到控制台" : "应用此方案"}
                </Button>
              </div>
            </div>
            <p className="whitespace-pre-line rounded-md border border-slate-700 bg-slate-800/40 p-3 text-sm leading-relaxed text-slate-300">
              {design.explanation}
            </p>
          </CardContent>
        </Card>
      )}

      {experiment && (
        <Card className="border-slate-700 bg-slate-900/60">
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-base text-slate-100">
              意图达成度验证（{experiment.n_seeds} 个种子的均值±标准差，每种子贪心评估 20 回合）
            </CardTitle>
            <span className="text-xs text-slate-500">耗时 {experiment.elapsed_sec}s</span>
          </CardHeader>
          <CardContent className="space-y-3">
            <Table>
              <TableHeader>
                <TableRow className="border-slate-700">
                  <TableHead>行为指标</TableHead>
                  {groups.map((g) => (
                    <TableHead key={g.key}
                      className={g.key === "ai" ? "text-violet-300" : g.key === "default_pbrs" ? "text-cyan-300" : "text-slate-400"}>
                      {g.label}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {METRIC_ROWS.map(({ key, label, pct, threshold }) => {
                  const th = threshold ?? 0.05
                  const ai = groups.find((g) => g.key === "ai")
                  const base = groups.find((g) => g.key === "default")
                  const va = ai?.metrics[key] ?? null
                  const vb = base?.metrics[key] ?? null
                  // AI 列相对默认方案差异超阈值时高亮
                  const changed = va !== null && vb !== null && Math.abs(va - vb) > th
                  return (
                    <TableRow key={key} className="border-slate-700">
                      <TableCell className="text-slate-300">{label}</TableCell>
                      {groups.map((g) => (
                        <TableCell key={g.key}
                          className={cn("font-mono",
                            g.key === "ai" && changed && "font-bold text-cyan-300",
                            g.key !== "ai" && "text-slate-400")}>
                          {fmtCell(key, g.metrics, pct)}
                        </TableCell>
                      ))}
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
            <p className="text-xs text-slate-500">
              「默认+PBRS」列用于剥离势函数塑形的贡献：AI 方案的价值 = AI 列相对该列的增量。
            </p>
            <p className="whitespace-pre-line rounded-md border border-slate-700 bg-slate-800/40 p-3 text-sm leading-relaxed text-slate-300">
              {experiment.interpretation}
            </p>
            <Button size="sm" variant="outline" onClick={onGoCompare}
              className="border-slate-600 bg-slate-800 text-slate-200 hover:bg-slate-700">
              到「实验对比」查看各组训练曲线 →
            </Button>

          </CardContent>
        </Card>
      )}
    </div>
  )
}
