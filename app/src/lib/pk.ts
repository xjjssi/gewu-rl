// pk.ts —— 人机 PK 的环境模拟与 RL 决策（与后端 gridworld_edu 的规则一致）
//
// 环境：上下左右移动；越界/撞墙原地不动并加撞墙惩罚；终点终止；奖励格每回合只领一次；
// 陷阱扣分（可选终止）；步数上限 4×H×W 截断。PK 计分不含势函数塑形（PBRS），
// 因为塑形只是训练时的"提示分"，不属于任务本身的得失。
//
// RL 决策：在玩家所在的同一个状态上，用 Q 表快照做一次 ε-greedy 决策，
// 与后端 QLearningAgent.select_action_explained 的定义一致。

import { Cell, type MazeGrid, type RewardConfigForm } from "@/types"

export const DIRS: [number, number][] = [[-1, 0], [1, 0], [0, -1], [0, 1]]

export interface PkEnvState {
  pos: [number, number]
  mask: number
  steps: number
  total: number
  done: boolean
  success: boolean
}

export interface StepResult {
  next: PkEnvState
  reward: number
  hitWall: boolean
  outcome: string
}

export function findCell(grid: MazeGrid, type: number): [number, number] {
  for (let r = 0; r < grid.length; r++)
    for (let c = 0; c < grid[r].length; c++) if (grid[r][c] === type) return [r, c]
  return [0, 0]
}

export function initialState(grid: MazeGrid): PkEnvState {
  return { pos: findCell(grid, Cell.START), mask: 0, steps: 0, total: 0, done: false, success: false }
}

/** 与后端 GridMazeEnv.state_index 一致：格子序号 × 2^奖励格数 + 收集掩码 */
export function stateIndex(grid: MazeGrid, nBonus: number, pos: [number, number], mask: number): number {
  const w = grid[0].length
  return (pos[0] * w + pos[1]) * (1 << nBonus) + mask
}

export function isBlocked(grid: MazeGrid, pos: [number, number], a: number): boolean {
  const [r, c] = [pos[0] + DIRS[a][0], pos[1] + DIRS[a][1]]
  return r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] === Cell.WALL
}

export function envStep(
  grid: MazeGrid, cfg: RewardConfigForm, bonusCells: [number, number][], maxSteps: number,
  st: PkEnvState, a: number,
): StepResult {
  let reward = cfg.step_penalty
  let { pos, mask } = st
  let terminated = false
  let success = false
  let outcome = "走到一个普通格子"
  const hitWall = isBlocked(grid, pos, a)
  if (hitWall) {
    reward += cfg.wall_penalty
    outcome = "撞墙，原地不动"
  } else {
    pos = [pos[0] + DIRS[a][0], pos[1] + DIRS[a][1]]
    const cell = grid[pos[0]][pos[1]]
    if (cell === Cell.GOAL) {
      reward += cfg.goal_reward
      terminated = true
      success = true
      outcome = "到达终点！"
    } else if (cell === Cell.BONUS) {
      const i = bonusCells.findIndex(([r, c]) => r === pos[0] && c === pos[1])
      const bit = 1 << i
      if (i >= 0 && !(mask & bit)) {
        reward += cfg.bonus_reward
        mask |= bit
        outcome = "拿到奖励格"
      } else outcome = "经过奖励格（已领过）"
    } else if (cell === Cell.TRAP) {
      reward += cfg.trap_penalty
      terminated = cfg.trap_terminal
      outcome = cfg.trap_terminal ? "掉进陷阱，回合结束" : "踩到陷阱"
    }
  }
  const steps = st.steps + 1
  const truncated = !terminated && steps >= maxSteps
  if (truncated) outcome += "（步数用完）"
  return {
    next: { pos, mask, steps, total: st.total + reward, done: terminated || truncated, success },
    reward,
    hitWall,
    outcome,
  }
}

export interface RlDecision {
  a: number
  explore: boolean
  u: number
  epsilon: number
  q: number[]
  best: number[]
  prob: number
}

/** ε-greedy：u < ε 探索（4 个方向等概率），否则在 Q 最大的方向中随机打破平局 */
export function rlDecide(q: number[], epsilon: number, rand: () => number = Math.random): RlDecision {
  const qMax = Math.max(...q)
  const best = q.flatMap((v, i) => (v === qMax ? [i] : []))
  const u = rand()
  const explore = u < epsilon
  const a = explore ? Math.floor(rand() * 4) : best[Math.floor(rand() * best.length)]
  const prob = epsilon / 4 + (best.includes(a) ? (1 - epsilon) / best.length : 0)
  return { a, explore, u, epsilon, q, best, prob }
}

/** BFS 最短步数（参考线）；不可达返回 null */
export function shortestPath(grid: MazeGrid): number | null {
  const start = findCell(grid, Cell.START)
  const goal = findCell(grid, Cell.GOAL)
  const h = grid.length
  const w = grid[0].length
  const dist = new Map<string, number>([[`${start[0]},${start[1]}`, 0]])
  const queue: [number, number][] = [start]
  while (queue.length) {
    const p = queue.shift()!
    const d = dist.get(`${p[0]},${p[1]}`)!
    if (p[0] === goal[0] && p[1] === goal[1]) return d
    for (let a = 0; a < 4; a++) {
      if (isBlocked(grid, p, a)) continue
      const n: [number, number] = [p[0] + DIRS[a][0], p[1] + DIRS[a][1]]
      const k = `${n[0]},${n[1]}`
      if (!dist.has(k) && h > 0 && w > 0) {
        dist.set(k, d + 1)
        queue.push(n)
      }
    }
  }
  return null
}

/** RL 自己从起点走 n 次（ε-greedy），统计平均表现，作为 PK 的比分参考 */
export function simulateRl(
  grid: MazeGrid, cfg: RewardConfigForm, bonusCells: [number, number][], maxSteps: number,
  qTable: number[][], epsilon: number, n = 20,
) {
  let succ = 0
  let steps = 0
  let total = 0
  for (let i = 0; i < n; i++) {
    let st = initialState(grid)
    while (!st.done) {
      const q = qTable[stateIndex(grid, bonusCells.length, st.pos, st.mask)]
      st = envStep(grid, cfg, bonusCells, maxSteps, st, rlDecide(q, epsilon).a).next
    }
    succ += st.success ? 1 : 0
    steps += st.steps
    total += st.total
  }
  return { successRate: succ / n, avgSteps: steps / n, avgReward: total / n }
}
