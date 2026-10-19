// 格物RL 前端类型定义

/** 格子类型编码（与后端 gridworld_edu.envs.grid_maze.CellType 一致） */
export const Cell = {
  EMPTY: 0,
  WALL: 1,
  START: 2,
  GOAL: 3,
  BONUS: 4,
  TRAP: 5,
} as const
export type CellValue = (typeof Cell)[keyof typeof Cell]

export type MazeGrid = number[][]

export type Algo = "qlearning" | "ppo" | "dqn"

export interface HyperParams {
  episodes: number
  total_timesteps: number
  learning_rate: number
  gamma: number
  epsilon_start: number
  epsilon_end: number
}

export interface RewardConfigForm {
  step_penalty: number
  wall_penalty: number
  goal_reward: number
  bonus_reward: number
  trap_penalty: number
  trap_terminal: boolean
  potential_shaping: boolean
}

export interface TrainRequestBody {
  maze: MazeGrid
  algo: Algo
  hyperparams: Record<string, number>
  reward_config: Record<string, number | boolean>
  seed: number
  label?: string
}

/** 训练 WebSocket 事件 */
export type TrainEvent =
  | { type: "started"; algo: Algo; seed: number }
  | { type: "episode"; episode: number; reward: number; steps: number; success: boolean; epsilon: number | null }
  | { type: "trajectory"; cells: [number, number][]; episode: number }
  | { type: "step"; pos: [number, number]; episode: number }
  | { type: "done"; summary: TrainSummary }
  | { type: "error"; message: string }

export interface PolicyCell {
  pos: [number, number]
  action: number
  q_values: number[]
  q_max: number
}

export interface TrainSummary {
  eval_success_rate: number
  policy: { cells: PolicyCell[] }
  episodes_run: number
  stopped: boolean
  elapsed_sec?: number
}

export interface EpisodePoint {
  episode: number
  reward: number
  steps: number
  success: boolean
  epsilon: number | null
}

export interface Preset {
  id: string
  name: string
  grid: MazeGrid
}

export interface CurriculumStage {
  stage: number
  name: string
  grid: MazeGrid
}

export interface RunMeta {
  id: string
  algo: Algo
  seed: number
  label?: string | null
  created_at: string
  episodes_run: number | null
  eval_success_rate: number | null
  stopped: boolean
}

export interface RunDetail {
  config: { algo: Algo; seed: number; label?: string | null; maze: MazeGrid; hyperparams: Record<string, number> }
  metrics: { episode: number; reward: number; steps: number; success: number }[]
  policy: { summary: TrainSummary }
}

export interface CurriculumResult {
  stages: { stage: number; name: string; curve: { episode: number; reward: number; success: number }[] }[]
  baseline: { name: string; curve: { episode: number; reward: number; success: number }[] }
  eval: { curriculum_success: number; baseline_success: number; note: string }
}

// ---- AI 奖励设计师 ----

export interface RewardDesignResult {
  reward_config: RewardConfigForm
  explanation: string
  source: "llm" | "fallback"
}

export interface BehaviorMetrics {
  success_rate: number
  success_rate_std: number
  avg_steps: number
  avg_steps_std: number
  avg_reward: number
  bonus_collect_rate: number
  bonus_collect_rate_std: number
  trap_trigger_rate: number
  trap_trigger_rate_std: number
  avg_bonus_count: number
  avg_trap_count: number
  eval_episodes: number
  episodes_to_converge: number | null  // 收敛回合（多种子均值），null 表示预算内未收敛
  episodes_to_converge_std: number
  converged_seeds: number              // 预算内收敛的种子数
  n_seeds: number
  avg_reward_common: number            // 统一标尺回报：各组策略均在默认奖励下评估
  avg_reward_common_std: number
}

export interface IntentGroup {
  key: string       // default | default_pbrs | ai
  label: string     // 默认方案 | 默认+PBRS | AI 方案
  metrics: BehaviorMetrics
}

export interface IntentExperimentResult {
  design: RewardDesignResult
  /** 三组对照：默认方案 / 默认+PBRS / AI 方案（AI 价值 = 相对第二组的增量） */
  groups: IntentGroup[]
  metrics_a: BehaviorMetrics  // 兼容字段：AI 方案
  metrics_b: BehaviorMetrics  // 兼容字段：默认方案
  metrics_c: BehaviorMetrics  // 默认+PBRS
  run_ids: Record<string, string[]>
  n_seeds: number
  seeds: number[]
  interpretation: string
  elapsed_sec: number
}

export interface LlmConfig {
  base_url: string
  api_key: string
  model: string
}
