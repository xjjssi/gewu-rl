// localEngine.ts —— 浏览器内训练引擎（零安装在线体验版的核心）
//
// 逐项移植自后端教学包 gridworld_edu（Python）：
//   · GridMazeEnv：移动/撞墙/终点/奖励格（每回合一次，收集掩码编码进状态）/陷阱/步数截断，
//     γ 一致的势函数塑形 F = γ·Φ(s′) − Φ(s)，Φ = −曼哈顿距离/最大距离，终止状态 Φ=0；
//   · QLearningAgent：ε 与 α 线性衰减（α 衰减到初值 1/10）、ε-greedy 随机打破平局、
//     终止/截断区分的 TD 更新，以及与后端字段一致的"决策解说"；
//   · 统一评估口径：ε=0.05 + 容差平局偏向终点 + 环路检测（20 回合），另附纯贪心对照。
// 与后端唯一的差别是随机数发生器（此处为可设种子的 mulberry32），因此同一种子下
// 数值不会与后端逐位相同，但算法与统计行为一致。PPO/DQN 与 AI 奖励设计师仍需本地后端。

import type { MazeGrid, PolicyCell, RewardConfigForm, StepDecision, TrainEvent, TrainSummary } from "@/types"

const DIRS: [number, number][] = [[-1, 0], [1, 0], [0, -1], [0, 1]]
const WALL = 1, START = 2, GOAL = 3, BONUS = 4, TRAP = 5

/** 可设种子的伪随机数发生器 */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s |= 0
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface StepOut { state: number; reward: number; terminated: boolean; truncated: boolean; success: boolean; pos: [number, number] }

export class LocalMazeEnv {
  readonly grid: MazeGrid
  readonly cfg: RewardConfigForm
  readonly h: number
  readonly w: number
  readonly start: [number, number]
  readonly goal: [number, number]
  readonly bonusCells: [number, number][]
  readonly bonusStates: number
  readonly maxSteps: number
  readonly nStates: number
  private readonly maxDist: number
  shapingGamma = 0.99
  pos: [number, number] = [0, 0]
  mask = 0
  steps = 0

  constructor(grid: MazeGrid, cfg: RewardConfigForm) {
    this.grid = grid
    this.cfg = cfg
    this.h = grid.length
    this.w = grid[0]?.length ?? 0
    if (this.h < 4 || this.w < 4 || this.h > 12 || this.w > 12) throw new Error(`网格尺寸须在 4~12 之间，收到 ${this.h}×${this.w}`)
    const find = (t: number) => {
      const out: [number, number][] = []
      grid.forEach((row, r) => row.forEach((v, c) => { if (v === t) out.push([r, c]) }))
      return out
    }
    const s = find(START), g = find(GOAL)
    if (s.length !== 1 || g.length !== 1) throw new Error("迷宫必须恰好包含一个起点和一个终点")
    this.start = s[0]
    this.goal = g[0]
    this.bonusCells = find(BONUS)                       // 行优先，与后端 np.argwhere 顺序一致
    this.bonusStates = 1 << this.bonusCells.length
    this.maxSteps = 4 * this.h * this.w
    this.nStates = this.h * this.w * this.bonusStates
    this.maxDist = (this.h - 1) + (this.w - 1)
  }

  potential(p: [number, number]): number {
    return -(Math.abs(p[0] - this.goal[0]) + Math.abs(p[1] - this.goal[1])) / this.maxDist
  }
  stateIndex(pos = this.pos, mask = this.mask): number {
    return (pos[0] * this.w + pos[1]) * this.bonusStates + mask
  }
  blocked(p: [number, number], a: number): boolean {
    const r = p[0] + DIRS[a][0], c = p[1] + DIRS[a][1]
    return r < 0 || r >= this.h || c < 0 || c >= this.w || this.grid[r][c] === WALL
  }
  reset(): number {
    this.pos = [...this.start] as [number, number]
    this.mask = 0
    this.steps = 0
    return this.stateIndex()
  }
  step(a: number): StepOut {
    const cfg = this.cfg
    this.steps += 1
    const old = this.pos
    let reward = cfg.step_penalty
    let terminated = false
    if (this.blocked(old, a)) {
      reward += cfg.wall_penalty
    } else {
      this.pos = [old[0] + DIRS[a][0], old[1] + DIRS[a][1]]
      const cell = this.grid[this.pos[0]][this.pos[1]]
      if (cell === GOAL) {
        reward += cfg.goal_reward
        terminated = true
      } else if (cell === BONUS) {
        const i = this.bonusCells.findIndex(([r, c]) => r === this.pos[0] && c === this.pos[1])
        const bit = 1 << i
        if (!(this.mask & bit)) { reward += cfg.bonus_reward; this.mask |= bit }
      } else if (cell === TRAP) {
        reward += cfg.trap_penalty
        if (cfg.trap_terminal) terminated = true
      }
    }
    if (cfg.potential_shaping) {
      const phiNext = terminated ? 0 : this.potential(this.pos)
      reward += this.shapingGamma * phiNext - this.potential(old)
    }
    const truncated = !terminated && this.steps >= this.maxSteps
    const success = terminated && this.pos[0] === this.goal[0] && this.pos[1] === this.goal[1]
    return { state: this.stateIndex(), reward, terminated, truncated, success, pos: [...this.pos] as [number, number] }
  }
}

export class LocalQAgent {
  readonly gamma: number
  readonly epsEnd: number
  readonly rand: () => number
  q: Float64Array[]
  epsilon: number
  lr: number
  private readonly epsStep: number
  private readonly lrStep: number
  private readonly lrEnd: number

  constructor(nStates: number, gamma: number, lr: number, epsStart: number, epsEnd: number,
    decayEpisodes: number, rand: () => number) {
    this.gamma = gamma
    this.epsEnd = epsEnd
    this.rand = rand
    this.q = Array.from({ length: nStates }, () => new Float64Array(4))
    this.epsilon = epsStart
    this.lr = lr
    this.lrEnd = lr * 0.1
    const d = Math.max(1, decayEpisodes)
    this.epsStep = (epsStart - epsEnd) / d
    this.lrStep = (lr - this.lrEnd) / d
  }
  private best(s: number): number[] {
    const q = this.q[s]
    const m = Math.max(q[0], q[1], q[2], q[3])
    return [0, 1, 2, 3].filter((a) => q[a] === m)
  }
  /** ε-greedy 选动作并返回解说字段（与后端 select_action_explained 定义一致） */
  selectExplained(s: number) {
    const eps = this.epsilon
    const q = Array.from(this.q[s])
    const best = this.best(s)
    const u = this.rand()
    const explore = u < eps
    const a = explore ? Math.floor(this.rand() * 4) : best[Math.floor(this.rand() * best.length)]
    const prob = eps / 4 + (best.includes(a) ? (1 - eps) / best.length : 0)
    return { a, explore, u, eps, q, best, prob }
  }
  update(s: number, a: number, r: number, ns: number, terminated: boolean): number {
    const target = terminated ? r : r + this.gamma * Math.max(...this.q[ns])
    this.q[s][a] += this.lr * (target - this.q[s][a])
    return target
  }
  decay(): void {
    this.epsilon = Math.max(this.epsEnd, this.epsilon - this.epsStep)
    this.lr = Math.max(this.lrEnd, this.lr - this.lrStep)
  }
  /** 每个可通行格子（未收集奖励格时）的贪心动作与 Q 值，供箭头与热力图 */
  greedyPolicy(env: LocalMazeEnv): PolicyCell[] {
    const cells: PolicyCell[] = []
    for (let r = 0; r < env.h; r++) for (let c = 0; c < env.w; c++) {
      if (env.grid[r][c] === WALL) continue
      const q = Array.from(this.q[(r * env.w + c) * env.bonusStates])
      const m = Math.max(...q)
      cells.push({ pos: [r, c], action: q.indexOf(m), q_values: q.map(round4), q_max: round4(m) })
    }
    return cells
  }
}

const round4 = (x: number) => Math.round(x * 1e4) / 1e4

/** 统一评估口径（移植自 gridworld_edu/evaluation.py） */
export function evaluateLocal(env: LocalMazeEnv, agent: LocalQAgent, rand: () => number, n = 20, epsilon = 0.05, tieTol = 0.05) {
  const act = (s: number, eps: number, tol: number) => {
    if (eps > 0 && rand() < eps) return Math.floor(rand() * 4)
    const q = agent.q[s]
    const qmax = Math.max(...q)
    if (tol <= 0) {
      const b = [0, 1, 2, 3].filter((a) => q[a] === qmax)
      return b[Math.floor(rand() * b.length)]
    }
    const t = Math.max(1e-9, tol * Math.abs(qmax))
    const cands = [0, 1, 2, 3].filter((a) => q[a] >= qmax - t)
    if (cands.length === 1) return cands[0]
    let best: number[] = [], bestD = Infinity
    for (const a of cands) {
      const p: [number, number] = env.blocked(env.pos, a) ? env.pos : [env.pos[0] + DIRS[a][0], env.pos[1] + DIRS[a][1]]
      const d = Math.abs(p[0] - env.goal[0]) + Math.abs(p[1] - env.goal[1])
      if (d < bestD) { bestD = d; best = [a] } else if (d === bestD) best.push(a)
    }
    return best[Math.floor(rand() * best.length)]
  }
  const run = (eps: number, tol: number) => {
    let success = 0, loops = 0, stable = 0
    for (let i = 0; i < n; i++) {
      let s = env.reset()
      const visits = new Map<number, number>([[s, 1]])
      let looped = false, out: StepOut
      do {
        out = env.step(act(s, eps, tol))
        s = out.state
        const v = (visits.get(s) ?? 0) + 1
        visits.set(s, v)
        if (v > 2) looped = true
      } while (!out.terminated && !out.truncated)
      success += out.success ? 1 : 0
      loops += looped ? 1 : 0
      stable += out.success && !looped ? 1 : 0
    }
    return { success: success / n, loop: loops / n, stable: stable / n }
  }
  const m = run(epsilon, tieTol)
  const pure = run(0, 0)
  return { success_rate: m.success, loop_rate: m.loop, stable_success_rate: m.stable, pure_greedy_success_rate: pure.success }
}

export interface LocalTrainParams {
  maze: MazeGrid
  reward: RewardConfigForm
  episodes: number
  learning_rate: number
  gamma: number
  epsilon_start: number
  epsilon_end: number
  seed: number
}

function makeAgent(env: LocalMazeEnv, p: LocalTrainParams, rand: () => number) {
  env.shapingGamma = p.gamma
  return new LocalQAgent(env.nStates, p.gamma, p.learning_rate, p.epsilon_start, p.epsilon_end, Math.floor(p.episodes * 0.8), rand)
}

/** 跑一回合；collect=true 时记录轨迹与逐步决策解说 */
function runEpisode(env: LocalMazeEnv, agent: LocalQAgent, collect: boolean) {
  let s = env.reset()
  let total = 0, steps = 0, success = false, done = false
  const cells: [number, number][] = [[...env.start] as [number, number]]
  const decisions: StepDecision[] = []
  while (!done) {
    const d = agent.selectExplained(s)
    const q0 = agent.q[s][d.a]
    const alpha = agent.lr
    const out = env.step(d.a)
    const nextMax = out.terminated ? 0 : Math.max(...agent.q[out.state])
    const target = agent.update(s, d.a, out.reward, out.state, out.terminated)
    if (collect) {
      cells.push(out.pos)
      decisions.push({
        a: d.a, x: d.explore ? 1 : 0, u: round4(d.u), e: round4(d.eps), q: d.q.map(round4), b: d.best,
        p: round4(d.prob), r: round4(out.reward), q0: round4(q0), q1: round4(agent.q[s][d.a]), al: round4(alpha),
        nm: round4(nextMax), tt: round4(target), t: out.terminated ? 1 : 0,
      })
    }
    total += out.reward
    steps += 1
    s = out.state
    done = out.terminated || out.truncated
    success = out.success
  }
  agent.decay()
  return { total, steps, success, cells, decisions }
}

/**
 * 在浏览器中训练表格 Q-Learning，按与后端 WebSocket 相同的事件格式回调 emit。
 * 每 25 回合让出一次主线程，保证界面流畅；shouldStop() 返回 true 时提前结束。
 */
export async function trainLocal(p: LocalTrainParams, emit: (ev: TrainEvent) => void, shouldStop: () => boolean): Promise<void> {
  const t0 = performance.now()
  const env = new LocalMazeEnv(p.maze, p.reward)
  const rand = mulberry32(p.seed + 1)
  const agent = makeAgent(env, p, rand)
  emit({ type: "started", algo: "qlearning", seed: p.seed })
  const every = Math.max(1, Math.ceil(p.episodes / 40))
  let ran = 0
  for (let ep = 1; ep <= p.episodes; ep++) {
    if (shouldStop()) break
    const collect = ep % every === 0 || ep === 1
    const r = runEpisode(env, agent, collect)
    ran = ep
    emit({ type: "episode", episode: ep, reward: round4(r.total), steps: r.steps, success: r.success, epsilon: round4(agent.epsilon) })
    if (collect) {
      emit({ type: "trajectory", cells: r.cells, episode: ep, decisions: r.decisions, gamma: p.gamma, policy: agent.greedyPolicy(env) })
      for (const pos of r.cells) emit({ type: "step", pos, episode: ep })
    }
    if (ep % 25 === 0) await new Promise((res) => setTimeout(res, 0))
  }
  const m = evaluateLocal(env, agent, mulberry32(p.seed + 99))
  const summary: TrainSummary = {
    eval_success_rate: m.success_rate,
    eval_loop_rate: m.loop_rate,
    eval_stable_success_rate: m.stable_success_rate,
    eval_pure_greedy_success_rate: m.pure_greedy_success_rate,
    policy: { cells: agent.greedyPolicy(env) },
    episodes_run: ran,
    stopped: shouldStop(),
    elapsed_sec: Math.round(performance.now() - t0) / 1000,
  }
  emit({ type: "done", summary })
}

/** 人机 PK 对手（浏览器版）：与后端 /api/pk/prepare 返回格式相同 */
export function pkPrepareLocal(p: LocalTrainParams) {
  const env = new LocalMazeEnv(p.maze, p.reward)
  if (env.nStates > 20000) throw new Error(`迷宫状态数 ${env.nStates} 过多（奖励格太多），请减少奖励格后再试`)
  const episodes = Math.max(20, Math.min(5000, p.episodes))
  const agent = makeAgent(env, { ...p, episodes }, mulberry32(p.seed + 1))
  const checkpoints = new Map<number, string>([
    [Math.max(1, Math.round(episodes * 0.05)), "新手 RL"],
    [Math.max(2, Math.round(episodes * 0.3)), "进阶 RL"],
    [episodes, "训练完成的 RL"],
  ])
  const opponents: { label: string; episode: number; epsilon: number; q_table: number[][] }[] = []
  for (let ep = 1; ep <= episodes; ep++) {
    runEpisode(env, agent, false)
    const label = checkpoints.get(ep)
    if (label) opponents.push({ label, episode: ep, epsilon: round4(agent.epsilon), q_table: agent.q.map((row) => Array.from(row, round4)) })
  }
  return { opponents, bonus_cells: env.bonusCells, max_steps: env.maxSteps, gamma: p.gamma }
}
