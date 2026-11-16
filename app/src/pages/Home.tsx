// Home —— 格物RL 单页主界面：左侧控制台 + 右侧标签页（环境搭建 / 训练实时视图 / 实验对比）

import { useState } from "react"
import { BookOpen, Check, GraduationCap, Link2, X } from "lucide-react"

import { ConceptGuide } from "@/components/ConceptGuide"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useBackendMode } from "@/hooks/useBackendMode"
import { useTrainingSocket } from "@/hooks/useTrainingSocket"
import { buildTaskUrl, readTaskFromUrl, type SharedTask } from "@/lib/taskLink"
import ControlPanel, { DEFAULT_CONFIG, type TrainConfig } from "@/sections/ControlPanel"
import CompareView from "@/sections/CompareView"
import LiveView from "@/sections/LiveView"
import MazeEditor from "@/sections/MazeEditor"
import PkView from "@/sections/PkView"
import RewardDesigner from "@/sections/RewardDesigner"
import type { MazeGrid, RewardConfigForm } from "@/types"

// 默认迷宫：单障碍 6×6
const DEFAULT_GRID: MazeGrid = [
  [2, 0, 1, 0, 0, 0],
  [0, 0, 1, 0, 4, 0],
  [0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 5, 1, 0, 0, 0],
  [0, 0, 1, 0, 0, 3],
]

export default function Home() {
  // 打开"实验任务链接"时，直接载入教师分享的迷宫、参数与任务说明
  const [shared] = useState<SharedTask | null>(() => readTaskFromUrl())
  const [grid, setGrid] = useState<MazeGrid>(shared?.grid ?? DEFAULT_GRID)
  const [config, setConfig] = useState<TrainConfig>(shared?.config ?? DEFAULT_CONFIG)
  const [tab, setTab] = useState("editor")
  const [showGuide, setShowGuide] = useState(false)
  const [showShare, setShowShare] = useState(false)
  const [shareNote, setShareNote] = useState("")
  const [copied, setCopied] = useState<string | null>(null)
  const [noteOpen, setNoteOpen] = useState(!!shared?.note)
  const mode = useBackendMode()

  const copyTaskLink = async () => {
    const url = buildTaskUrl({ grid, config, note: shareNote })
    try {
      await navigator.clipboard.writeText(url)
      setCopied("已复制到剪贴板，发给学生即可")
    } catch {
      setCopied(url) // 剪贴板不可用时直接显示链接，手动复制
    }
  }
  const { state, start, stop } = useTrainingSocket()

  const handleStart = () => {
    const { algo, hyper, reward, seed } = config
    start({
      maze: grid,
      algo,
      hyperparams:
        algo === "qlearning"
          ? {
              episodes: hyper.episodes,
              learning_rate: hyper.learning_rate,
              gamma: hyper.gamma,
              epsilon_start: hyper.epsilon_start,
              epsilon_end: hyper.epsilon_end,
            }
          : {
              total_timesteps: hyper.total_timesteps,
              learning_rate: hyper.learning_rate,
              gamma: hyper.gamma,
            },
      reward_config: { ...reward },
      seed,
    })
    setTab("live")
  }

  return (
    <div className="min-h-screen bg-paper text-slate-900">
      <header className="border-t-[3px] border-b border-t-brand border-b-slate-200 bg-white px-6 py-3.5">
        <div className="mx-auto flex max-w-7xl items-center gap-3">
          <GraduationCap className="h-6 w-6 text-brand" />
          <h1 className="text-xl font-semibold text-ink">
            格物RL <span className="text-sm font-normal text-slate-500">· 可视化强化学习教学实验平台</span>
          </h1>
          {mode && (
            <span
              title={mode === "online" ? "已连接本地后端：全部功能可用" : "无需安装：Q-Learning 训练、决策解说、价值传播与人机 PK 在浏览器中运行"}
              className={mode === "online"
                ? "rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs text-emerald-700 ring-1 ring-emerald-200"
                : "rounded-full bg-sky-50 px-2.5 py-0.5 text-xs text-sky-800 ring-1 ring-sky-200"}
            >
              {mode === "online" ? "完整版 · 已连接后端" : "在线体验版 · 浏览器内运行"}
            </span>
          )}
          <button
            type="button"
            onClick={() => { setShowShare((v) => !v); setCopied(null) }}
            className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm text-ink transition-colors hover:bg-slate-100"
          >
            <Link2 className="h-4 w-4" />
            分享实验任务
          </button>
          <button
            type="button"
            onClick={() => setShowGuide((v) => !v)}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-1.5 text-sm text-ink transition-colors hover:bg-slate-100"
          >
            <BookOpen className="h-4 w-4" />
            RL 与人类决策
          </button>
        </div>
      </header>
      {showShare && (
        <div className="mx-auto max-w-7xl px-6 pt-6">
          <div className="space-y-2 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-ink">分享实验任务</h2>
              <span className="text-xs text-slate-500">当前迷宫与左侧全部参数会编码进链接，学生打开即得到同一实验设置</span>
              <button type="button" aria-label="关闭" onClick={() => setShowShare(false)}
                className="ml-auto rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
                <X className="h-4 w-4" />
              </button>
            </div>
            <textarea
              value={shareNote}
              onChange={(e) => setShareNote(e.target.value.slice(0, 300))}
              rows={2}
              placeholder="给学生的任务说明（可选），如：把 ε 终止值分别设为 0 和 0.2 各训练一次，在“决策解说”里比较后期回合的探索步数，并解释差异。"
              className="w-full rounded-md border border-slate-300 bg-slate-50 px-3 py-2 text-sm text-slate-800"
            />
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" onClick={copyTaskLink}
                className="inline-flex items-center gap-1.5 rounded-md bg-brand px-3 py-1.5 text-sm text-white hover:bg-brand-dark">
                <Link2 className="h-4 w-4" /> 复制任务链接
              </button>
              {copied && (
                <span className="inline-flex max-w-full items-center gap-1 break-all text-xs text-emerald-700">
                  <Check className="h-3.5 w-3.5 shrink-0" /> {copied}
                </span>
              )}
            </div>
          </div>
        </div>
      )}
      {shared?.note && noteOpen && (
        <div className="mx-auto max-w-7xl px-6 pt-6">
          <div className="flex gap-3 rounded-lg border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">
            <BookOpen className="mt-0.5 h-4 w-4 shrink-0" />
            <p><b>本次实验任务：</b>{shared.note}</p>
            <button type="button" aria-label="收起" onClick={() => setNoteOpen(false)}
              className="ml-auto self-start rounded p-1 text-sky-500 hover:bg-sky-100">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
      {showGuide && (
        <div className="mx-auto max-w-7xl px-6 pt-6">
          <ConceptGuide onClose={() => setShowGuide(false)} />
        </div>
      )}

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6 lg:grid-cols-[340px_1fr]">
        <aside>
          <ControlPanel
            config={config}
            onChange={setConfig}
            status={state.status}
            onStart={handleStart}
            onStop={stop}
          />
        </aside>

        <section>
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList className="border border-slate-200 bg-slate-100 [&>button[data-state=active]]:text-ink">
              <TabsTrigger value="editor">环境搭建</TabsTrigger>
              <TabsTrigger value="designer">AI 奖励设计师</TabsTrigger>
              <TabsTrigger value="live">训练实时视图</TabsTrigger>
              <TabsTrigger value="compare">实验对比</TabsTrigger>
              <TabsTrigger value="pk">人机 PK</TabsTrigger>
            </TabsList>
            {/* forceMount：切换标签页时保留各页状态（意图实验结果、回放进度、已勾选曲线），录屏演示时不丢失 */}
            <TabsContent value="editor" className="mt-4">
              <MazeEditor grid={grid} onChange={setGrid} />
            </TabsContent>
            <TabsContent value="designer" forceMount className="mt-4 data-[state=inactive]:hidden">
              <RewardDesigner
                grid={grid}
                onApplyReward={(reward: RewardConfigForm) => setConfig((c) => ({ ...c, reward }))}
                onGoCompare={() => setTab("compare")}
                onLoadMaze={setGrid}
              />
            </TabsContent>
            <TabsContent value="live" forceMount className="mt-4 data-[state=inactive]:hidden">
              <LiveView grid={grid} training={state} />
            </TabsContent>
            <TabsContent value="compare" forceMount className="mt-4 data-[state=inactive]:hidden">
              <CompareView active={tab === "compare"} />
            </TabsContent>
            <TabsContent value="pk" forceMount className="mt-4 data-[state=inactive]:hidden">
              {/* PK 对手固定用表格 Q-Learning；当前选的是 PPO/DQN 时退回 Q-Learning 默认超参 */}
              <PkView
                grid={grid}
                reward={config.reward}
                hyper={config.algo === "qlearning" ? config.hyper : DEFAULT_CONFIG.hyper}
                seed={config.seed}
                active={tab === "pk"}
              />
            </TabsContent>
          </Tabs>
        </section>
      </main>
    </div>
  )
}
