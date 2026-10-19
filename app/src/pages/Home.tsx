// Home —— 格物RL 单页主界面：左侧控制台 + 右侧标签页（环境搭建 / 训练实时视图 / 实验对比）

import { useState } from "react"
import { GraduationCap } from "lucide-react"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useTrainingSocket } from "@/hooks/useTrainingSocket"
import ControlPanel, { DEFAULT_CONFIG, type TrainConfig } from "@/sections/ControlPanel"
import CompareView from "@/sections/CompareView"
import LiveView from "@/sections/LiveView"
import MazeEditor from "@/sections/MazeEditor"
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
  const [grid, setGrid] = useState<MazeGrid>(DEFAULT_GRID)
  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG)
  const [tab, setTab] = useState("editor")
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
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/70 px-6 py-4">
        <div className="mx-auto flex max-w-7xl items-center gap-3">
          <GraduationCap className="h-6 w-6 text-cyan-400" />
          <h1 className="text-xl font-bold tracking-wide">
            格物RL <span className="text-sm font-normal text-slate-400">· 可视化强化学习教学实验平台</span>
          </h1>
        </div>
      </header>

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
            <TabsList className="border border-slate-700 bg-slate-900">
              <TabsTrigger value="editor">环境搭建</TabsTrigger>
              <TabsTrigger value="designer">AI 奖励设计师</TabsTrigger>
              <TabsTrigger value="live">训练实时视图</TabsTrigger>
              <TabsTrigger value="compare">实验对比</TabsTrigger>
            </TabsList>
            <TabsContent value="editor" className="mt-4">
              <MazeEditor grid={grid} onChange={setGrid} />
            </TabsContent>
            <TabsContent value="designer" className="mt-4">
              <RewardDesigner
                grid={grid}
                onApplyReward={(reward: RewardConfigForm) => setConfig((c) => ({ ...c, reward }))}
                onGoCompare={() => setTab("compare")}
                onLoadMaze={setGrid}
              />
            </TabsContent>
            <TabsContent value="live" className="mt-4">
              <LiveView grid={grid} training={state} />
            </TabsContent>
            <TabsContent value="compare" className="mt-4">
              <CompareView active={tab === "compare"} />
            </TabsContent>
          </Tabs>
        </section>
      </main>
    </div>
  )
}
