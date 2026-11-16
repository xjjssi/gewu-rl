// LiveView —— 训练实时视图：迷宫动画 + 实时曲线 + 指标卡片 + 策略叠加
//
// 动画两种模式：
// - 训练中（live）：以固定帧率跟随 step 关键帧流，自动追赶最新位置；
// - 训练结束（replay）：进入回放模式——可选回合、播放/暂停、1×/2×/4× 倍速、
//   步数进度条拖动。即使训练 0.3 秒就结束（如 Q-Learning），也能完整回放。

import { useEffect, useMemo, useRef, useState } from "react"
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { DecisionCard } from "@/components/DecisionCard"
import { MODE_STYLE, modeStats } from "@/lib/decision"
import { MazeCanvas } from "@/components/MazeCanvas"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import type { MazeGrid } from "@/types"
import type { TrainingState } from "@/hooks/useTrainingSocket"

const chartConfig = {
  reward: { label: "回合回报", color: "#0877bd" },
  success: { label: "成功率(滑动)", color: "#2e8b57" },
} satisfies ChartConfig

const SPEEDS = [0.5, 1, 2, 4] as const

function movingAvg(xs: number[], w: number): number[] {
  const out: number[] = []
  let sum = 0
  for (let i = 0; i < xs.length; i++) {
    sum += xs[i]
    if (i >= w) sum -= xs[i - w]
    out.push(sum / Math.min(i + 1, w))
  }
  return out
}

interface Props {
  grid: MazeGrid
  training: TrainingState
}

export default function LiveView({ grid, training }: Props) {
  const { status, episodes, stepStream, trajectory, trajectories, currentEpisode, epsilon, summary, error } = training
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [playIdx, setPlayIdx] = useState(0)

  // ---- 回放模式状态 ----
  const [trajIdx, setTrajIdx] = useState(0)      // 选中的代表性轨迹下标
  const [stepIdx, setStepIdx] = useState(0)      // 轨迹内步数下标
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState<number>(1)

  const finished = status === "done" || status === "stopped"
  // 有可回放轨迹时进入回放模式（训练失败无轨迹则退回静态最后一帧）
  const replay = finished && trajectories.length > 0
  const curTraj = replay ? trajectories[Math.min(trajIdx, trajectories.length - 1)] : null
  // 决策解说（仅 Q-Learning 提供）：第 stepIdx 条决策把智能体从 cells[stepIdx] 带到 cells[stepIdx+1]
  const decisions = curTraj?.decisions
  const curDecision = decisions && stepIdx < decisions.length ? decisions[stepIdx] : null
  const stats = useMemo(() => (decisions ? modeStats(decisions) : null), [decisions])

  // 进入回放模式时：默认选最后一条轨迹（episode=-1 为最终贪心评估），从头自动播放
  const wasRunning = useRef(false)
  useEffect(() => {
    if (status === "running") wasRunning.current = true
    if (finished && wasRunning.current && trajectories.length > 0) {
      wasRunning.current = false
      setTrajIdx(trajectories.length - 1)
      setStepIdx(0)
      setPlaying(true)
    }
  }, [status, finished, trajectories.length])

  // live 模式：以固定帧率回放 step 关键帧流（自动追赶最新位置）
  useEffect(() => {
    if (status !== "running") {
      // 训练结束且无轨迹可回放时，停在最后一个位置
      if (!replay && stepStream.length > 0) setPlayIdx(stepStream.length - 1)
      return
    }
    const timer = setInterval(() => {
      setPlayIdx((i) => Math.min(i + 1, stepStream.length - 1))
    }, 30)
    return () => clearInterval(timer)
  }, [status, stepStream.length, replay])

  // replay 模式：按倍速推进当前轨迹的步数
  useEffect(() => {
    if (!replay || !playing || !curTraj) return
    const timer = setInterval(() => {
      setStepIdx((i) => {
        if (i + 1 >= curTraj.cells.length) {
          setPlaying(false) // 播到末尾自动暂停，便于观察最终位置
          return i
        }
        return i + 1
      })
    }, 150 / speed)
    return () => clearInterval(timer)
  }, [replay, playing, speed, curTraj])

  // 切换回合轨迹：从头开始播放
  const selectTraj = (i: number) => {
    setTrajIdx(i)
    setStepIdx(0)
    setPlaying(true)
  }

  const agentPos = replay && curTraj
    ? curTraj.cells[Math.min(stepIdx, curTraj.cells.length - 1)]
    : stepStream.length > 0
      ? stepStream[Math.min(playIdx, stepStream.length - 1)]
      : null
  const trail = replay && curTraj ? curTraj.cells : trajectory
  const policy = finished ? summary?.policy.cells ?? null : null

  // 曲线数据：回报 + 成功率滑动平均（20 回合窗口）
  const curveData = useMemo(() => {
    const rewards = movingAvg(episodes.map((e) => e.reward), 20)
    const succ = movingAvg(episodes.map((e) => (e.success ? 1 : 0)), 20)
    return episodes.map((e, i) => ({
      episode: e.episode,
      reward: Number(rewards[i].toFixed(3)),
      success: Number(succ[i].toFixed(3)),
    }))
  }, [episodes])

  const avgReward = curveData.length > 0 ? curveData[curveData.length - 1].reward : null
  const avgSuccess = curveData.length > 0 ? curveData[curveData.length - 1].success : null

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="border-slate-200 bg-white">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-slate-900">
            {replay ? "训练回放" : "迷宫实时动画"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <MazeCanvas
            grid={grid}
            agentPos={agentPos}
            trail={trail}
            policy={policy}
            showHeatmap={showHeatmap}
            agentAction={curDecision?.a ?? null}
            agentMode={curDecision ? (curDecision.x ? "explore" : "exploit") : null}
          />

          {replay && curTraj && (
            <div className="space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="outline"
                  className="border-slate-300 bg-slate-100 text-slate-800 hover:bg-slate-200"
                  onClick={() => {
                    if (!playing && stepIdx >= curTraj.cells.length - 1) setStepIdx(0) // 播完后点播放=重播
                    setPlaying(!playing)
                  }}>
                  {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  {playing ? "暂停" : "播放"}
                </Button>
                <Button size="sm" variant="outline" title="上一步"
                  className="border-slate-300 bg-slate-100 px-2 text-slate-800 hover:bg-slate-200"
                  disabled={stepIdx <= 0}
                  onClick={() => { setPlaying(false); setStepIdx((i) => Math.max(0, i - 1)) }}>
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button size="sm" variant="outline" title="下一步（逐步讲解）"
                  className="border-slate-300 bg-slate-100 px-2 text-slate-800 hover:bg-slate-200"
                  disabled={stepIdx >= curTraj.cells.length - 1}
                  onClick={() => { setPlaying(false); setStepIdx((i) => Math.min(curTraj.cells.length - 1, i + 1)) }}>
                  <ChevronRight className="h-4 w-4" />
                </Button>
                <div className="flex gap-1">
                  {SPEEDS.map((s) => (
                    <Button key={s} size="sm" variant="outline"
                      className={cn("border-slate-300 px-2 text-xs",
                        speed === s ? "bg-brand text-white hover:bg-brand-dark" : "bg-slate-100 text-slate-700 hover:bg-slate-200")}
                      onClick={() => setSpeed(s)}>
                      {s}×
                    </Button>
                  ))}
                </div>
                <span className="ml-auto whitespace-nowrap text-xs text-slate-500">
                  {curTraj.episode === -1 ? "最终贪心评估" : `第 ${curTraj.episode} 回合`}
                  {" · "}{stepIdx + 1}/{curTraj.cells.length} 步
                </span>
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-slate-500">回合（共 {trajectories.length} 条代表性轨迹）</Label>
                <Slider min={0} max={trajectories.length - 1} step={1}
                  value={[Math.min(trajIdx, trajectories.length - 1)]}
                  onValueChange={([i]) => selectTraj(i)} />
              </div>
              <div className="space-y-1">
                <Label className="text-xs text-slate-500">步数进度</Label>
                <Slider min={0} max={Math.max(0, curTraj.cells.length - 1)} step={1}
                  value={[Math.min(stepIdx, curTraj.cells.length - 1)]}
                  onValueChange={([i]) => { setStepIdx(i); setPlaying(false) }} />
              </div>
              {decisions && decisions.length > 0 && (
                <div className="space-y-1">
                  <Label className="text-xs text-slate-500">
                    逐步决策条（<span className="text-orange-700">■ 探索</span>
                    {" / "}<span className="text-sky-800">■ 寻优</span>，点击跳到该步）
                  </Label>
                  <div className="flex flex-wrap gap-[2px]">
                    {decisions.map((d, i) => (
                      <button key={i} type="button" title={`第 ${i + 1} 步：${d.x ? "探索" : "寻优"}`}
                        onClick={() => { setStepIdx(i); setPlaying(false) }}
                        className={cn("h-3 w-2 rounded-[1px]",
                          d.x ? MODE_STYLE.explore.bg : MODE_STYLE.exploit.bg,
                          i === stepIdx ? "ring-2 ring-ink ring-offset-1" : "opacity-80 hover:opacity-100")} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {finished && policy && (
            <div className="flex items-center gap-2">
              <Switch checked={showHeatmap} onCheckedChange={setShowHeatmap} id="heat" />
              <Label htmlFor="heat" className="text-xs text-slate-700">
                Q 值热力图（关闭时显示策略箭头底色）
              </Label>
            </div>
          )}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <MetricCard label="当前回合" value={currentEpisode || "—"} />
            <MetricCard label="探索率 ε" value={epsilon != null ? epsilon.toFixed(3) : "—"} />
            <MetricCard label="均回报(20回合)" value={avgReward != null ? avgReward.toFixed(2) : "—"} />
            <MetricCard
              label="评估成功率"
              value={summary ? `${(summary.eval_success_rate * 100).toFixed(0)}%` : avgSuccess != null ? `${(avgSuccess * 100).toFixed(0)}%*` : "—"}
            />
          </div>
          {error && <p className="text-sm text-rose-600">{error}</p>}
          {finished && summary && (
            <p className="text-xs text-slate-500">
              训练{summary.stopped ? "被手动停止" : "完成"}：共 {summary.episodes_run} 回合
              {summary.elapsed_sec != null && `，耗时 ${summary.elapsed_sec}s`}，评估成功率 {(summary.eval_success_rate * 100).toFixed(0)}%
              {summary.eval_loop_rate != null && `（绕圈回合占比 ${(summary.eval_loop_rate * 100).toFixed(0)}%，ε=0.05 统一口径）`}
            </p>
          )}
        </CardContent>
      </Card>

      <div className="space-y-4">
        {replay && curTraj && (
          <Card className="border-slate-200 bg-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-base text-slate-900">
                决策解说：这一步是探索还是寻优？
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {stats && (
                <div className="space-y-1">
                  <div className="flex h-2 w-full overflow-hidden rounded">
                    <div className={MODE_STYLE.explore.bg} style={{ width: `${stats.exploreRate * 100}%` }} />
                    <div className={cn("flex-1", MODE_STYLE.exploit.bg)} />
                  </div>
                  <p className="text-xs text-slate-600">
                    {curTraj.episode === -1 ? "最终评估" : `第 ${curTraj.episode} 回合`}共 {stats.n} 步：
                    <b className="text-orange-700">探索 {stats.explore} 步（{(stats.exploreRate * 100).toFixed(0)}%）</b>，
                    <b className="text-sky-800">寻优 {stats.exploit} 步</b>。
                    拖动“回合”滑块对比：早期回合几乎都在探索，后期大多在寻优——这就是 ε 衰减带来的“先探索、后利用”。
                  </p>
                </div>
              )}
              {curDecision && decisions ? (
                <DecisionCard
                  grid={grid}
                  decision={curDecision}
                  from={curTraj.cells[stepIdx]}
                  to={curTraj.cells[stepIdx + 1]}
                  stepNo={stepIdx + 1}
                  gamma={curTraj.gamma ?? 0.99}
                />
              ) : decisions ? (
                <p className="text-sm text-slate-600">
                  回合结束。点击“上一步”或决策条，可以逐步查看智能体每一步的选择依据。
                </p>
              ) : (
                <p className="text-sm text-slate-600">
                  逐步决策解说目前支持 Q-Learning：表格法每一步的依据（掷骰、Q 值、TD 更新）都能完整展开。
                  PPO / DQN 由神经网络输出动作，此处只展示轨迹回放。
                </p>
              )}
            </CardContent>
          </Card>
        )}
        <Card className="border-slate-200 bg-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-slate-900">回合回报曲线（20 回合滑动平均）</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-56 w-full">
              <LineChart data={curveData} margin={{ left: 8, right: 8, top: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="episode" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} width={50} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line type="monotone" dataKey="reward" stroke="var(--color-reward)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-slate-900">成功率曲线（20 回合滑动平均）</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-56 w-full">
              <LineChart data={curveData} margin={{ left: 8, right: 8, top: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="episode" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} width={50} domain={[0, 1]} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line type="monotone" dataKey="success" stroke="var(--color-success)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-900">{value}</div>
    </div>
  )
}
