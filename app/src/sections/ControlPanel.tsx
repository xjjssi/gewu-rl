// ControlPanel —— 训练控制台：算法选择 / 超参 / 奖励塑形 / 启停

import { useRef } from "react"
import { Play, Square } from "lucide-react"

import { ConceptTip } from "@/components/ConceptTip"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import type { TrainStatus } from "@/hooks/useTrainingSocket"
import type { ConceptKey } from "@/lib/concepts"
import type { Algo, HyperParams, RewardConfigForm } from "@/types"

export interface TrainConfig {
  algo: Algo
  hyper: HyperParams
  reward: RewardConfigForm
  seed: number
}

// 算法分组：Q-Learning 与 SB3（PPO/DQN）的超参量级差异大（学习率差 2~3 个数量级），
// 因此按组分别保存，切换算法时各组互不干塌。
type AlgoGroup = "qlearning" | "sb3"
const algoGroup = (a: Algo): AlgoGroup => (a === "qlearning" ? "qlearning" : "sb3")

const DEFAULT_HYPER: Record<AlgoGroup, HyperParams> = {
  qlearning: {
    episodes: 500,
    total_timesteps: 10000,
    learning_rate: 0.1,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_end: 0.05,
  },
  sb3: {
    episodes: 500,
    total_timesteps: 10000,
    learning_rate: 3e-4,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_end: 0.05,
  },
}

export const DEFAULT_CONFIG: TrainConfig = {
  algo: "qlearning",
  hyper: { ...DEFAULT_HYPER.qlearning },
  reward: {
    step_penalty: -0.01,
    wall_penalty: -0.05,
    goal_reward: 1.0,
    bonus_reward: 0.5,
    trap_penalty: -1.0,
    trap_terminal: false,
    potential_shaping: false,
  },
  seed: 0,
}

const STATUS_BADGE: Record<TrainStatus, { label: string; cls: string }> = {
  idle: { label: "待命", cls: "bg-slate-500" },
  running: { label: "训练中", cls: "bg-emerald-600 animate-pulse" },
  done: { label: "已完成", cls: "bg-sky-600" },
  stopped: { label: "已停止", cls: "bg-amber-600" },
  error: { label: "出错", cls: "bg-rose-600" },
}

interface Props {
  config: TrainConfig
  onChange: (c: TrainConfig) => void
  status: TrainStatus
  onStart: () => void
  onStop: () => void
}

function NumField({ label, value, onChange, step = 0.01, tip }: {
  label: string; value: number; onChange: (v: number) => void; step?: number; tip?: ConceptKey
}) {
  return (
    <div className="space-y-1">
      <Label className="flex items-center gap-1 text-xs text-slate-500">{label}{tip && <ConceptTip k={tip} />}</Label>
      <Input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-8 border-slate-300 bg-slate-100 text-slate-900"
      />
    </div>
  )
}

export default function ControlPanel({ config, onChange, status, onStart, onStop }: Props) {
  const { algo, hyper, reward } = config
  const running = status === "running"
  const group = algoGroup(algo)
  // 每组算法各自记住一套超参；切换算法时保存当前组、还原目标组
  const savedHyper = useRef<Record<AlgoGroup, HyperParams>>({
    qlearning: { ...DEFAULT_HYPER.qlearning },
    sb3: { ...DEFAULT_HYPER.sb3 },
  })
  const setHyper = (k: keyof HyperParams, v: number) => {
    const next = { ...hyper, [k]: v }
    savedHyper.current[group] = next
    onChange({ ...config, hyper: next })
  }
  const switchAlgo = (v: Algo) => {
    savedHyper.current[group] = hyper
    onChange({ ...config, algo: v, hyper: { ...savedHyper.current[algoGroup(v)] } })
  }
  const setReward = (k: keyof RewardConfigForm, v: number | boolean) =>
    onChange({ ...config, reward: { ...reward, [k]: v } })

  return (
    <div className="space-y-4">
      <Card className="border-slate-200 bg-white">
        <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
          <CardTitle className="text-base text-slate-900">训练控制台</CardTitle>
          <Badge className={STATUS_BADGE[status].cls}>{STATUS_BADGE[status].label}</Badge>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <Label className="text-xs text-slate-500">算法</Label>
            <Select value={algo} onValueChange={(v) => switchAlgo(v as Algo)} disabled={running}>
              <SelectTrigger className="border-slate-300 bg-slate-100">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="qlearning">表格 Q-Learning（从零实现）</SelectItem>
                <SelectItem value="ppo">PPO（Stable-Baselines3）</SelectItem>
                <SelectItem value="dqn">DQN（Stable-Baselines3）</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {algo === "qlearning" ? (
            <SliderRow label={`训练回合数：${hyper.episodes}`} min={100} max={2000} step={50}
              value={hyper.episodes} onChange={(v) => setHyper("episodes", v)} disabled={running} tip="episodes" />
          ) : (
            <SliderRow label={`训练步数：${hyper.total_timesteps}`} min={5000} max={50000} step={1000}
              value={hyper.total_timesteps} onChange={(v) => setHyper("total_timesteps", v)} disabled={running} />
          )}
          {algo === "qlearning" ? (
            <SliderRow label={`学习率 α：${hyper.learning_rate}`} min={0.01} max={1} step={0.01}
              value={hyper.learning_rate} onChange={(v) => setHyper("learning_rate", v)} disabled={running} tip="learning_rate" />
          ) : (
            // PPO/DQN 学习率量级为 1e-5 ~ 1e-2，线性滑杆无法表达，用对数刻度
            <LogSliderRow label={`学习率 α：${hyper.learning_rate.toExponential(1)}`}
              minExp={-5} maxExp={-2} value={hyper.learning_rate}
              onChange={(v) => setHyper("learning_rate", v)} disabled={running} tip="learning_rate" />
          )}
          <SliderRow label={`折现因子 γ：${hyper.gamma}`} min={0.8} max={0.999} step={0.001}
            value={hyper.gamma} onChange={(v) => setHyper("gamma", v)} disabled={running} tip="gamma" />
          {algo === "qlearning" && (
            <>
              <SliderRow label={`ε 初始：${hyper.epsilon_start}`} min={0.1} max={1} step={0.05}
                value={hyper.epsilon_start} onChange={(v) => setHyper("epsilon_start", v)} disabled={running} tip="epsilon" />
              <SliderRow label={`ε 终止：${hyper.epsilon_end}`} min={0} max={0.5} step={0.01}
                value={hyper.epsilon_end} onChange={(v) => setHyper("epsilon_end", v)} disabled={running} tip="epsilon" />
            </>
          )}
          <NumField label="随机种子" value={config.seed} step={1} tip="seed"
            onChange={(v) => onChange({ ...config, seed: Math.trunc(v) })} />

          <div className="flex gap-2 pt-1">
            <Button className="flex-1 bg-emerald-600 hover:bg-emerald-700" onClick={onStart} disabled={running}>
              <Play className="h-4 w-4" /> 开始训练
            </Button>
            <Button className="flex-1 border-rose-300 bg-white text-rose-600 hover:bg-rose-50 hover:text-rose-700" variant="outline" onClick={onStop} disabled={!running}>
              <Square className="h-4 w-4" /> 停止
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="border-slate-200 bg-white">
        <CardHeader className="pb-3">
          <CardTitle className="text-base text-slate-900">奖励塑形</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3">
          <NumField label="每步代价" value={reward.step_penalty} onChange={(v) => setReward("step_penalty", v)} tip="step_penalty" />
          <NumField label="撞墙惩罚" value={reward.wall_penalty} onChange={(v) => setReward("wall_penalty", v)} tip="wall_penalty" />
          <NumField label="终点奖励" value={reward.goal_reward} onChange={(v) => setReward("goal_reward", v)} tip="goal_reward" />
          <NumField label="奖励格" value={reward.bonus_reward} onChange={(v) => setReward("bonus_reward", v)} tip="bonus_reward" />
          <NumField label="陷阱惩罚" value={reward.trap_penalty} onChange={(v) => setReward("trap_penalty", v)} tip="trap_penalty" />
          <div className="col-span-2 flex items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
            <Label className="flex items-center gap-1 text-xs text-slate-700">势函数塑形（PBRS，最优策略不变）<ConceptTip k="potential_shaping" /></Label>
            <Switch checked={reward.potential_shaping}
              onCheckedChange={(v) => setReward("potential_shaping", v)} disabled={running} />
          </div>
          <div className="col-span-2 flex items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
            <Label className="flex items-center gap-1 text-xs text-slate-700">陷阱终止回合<ConceptTip k="trap_penalty" /></Label>
            <Switch checked={reward.trap_terminal}
              onCheckedChange={(v) => setReward("trap_terminal", v)} disabled={running} />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function SliderRow({ label, value, min, max, step, onChange, disabled, tip }: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; disabled?: boolean; tip?: ConceptKey
}) {
  return (
    <div className="space-y-1">
      <Label className="flex items-center gap-1 text-xs text-slate-500">{label}{tip && <ConceptTip k={tip} />}</Label>
      <Slider min={min} max={max} step={step} value={[value]} disabled={disabled}
        onValueChange={([v]) => onChange(v)} />
    </div>
  )
}

/** 对数刻度滑杆：滑杆位置 = 10 的指数，用于学习率这类跨数量级的超参 */
function LogSliderRow({ label, value, minExp, maxExp, onChange, disabled, tip }: {
  label: string; value: number; minExp: number; maxExp: number
  onChange: (v: number) => void; disabled?: boolean; tip?: ConceptKey
}) {
  const exp = Math.min(maxExp, Math.max(minExp, Math.log10(value)))
  return (
    <div className="space-y-1">
      <Label className="flex items-center gap-1 text-xs text-slate-500">{label}{tip && <ConceptTip k={tip} />}</Label>
      <Slider min={minExp} max={maxExp} step={0.05} value={[exp]} disabled={disabled}
        onValueChange={([e]) => onChange(Number(Math.pow(10, e).toPrecision(2)))} />
    </div>
  )
}
