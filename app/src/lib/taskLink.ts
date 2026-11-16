// taskLink.ts —— "实验任务链接"：把迷宫 + 训练参数 + 任务说明编码进 URL（#task=…），
// 教师一键分享，学生打开即得到同一实验设置。纯前端实现，无需服务器，在线体验版与完整版通用。
// 解析时做严格校验，格式不对的链接直接忽略，不会让页面进入非法状态。

import type { TrainConfig } from "@/sections/ControlPanel"
import type { MazeGrid } from "@/types"

export interface SharedTask {
  grid: MazeGrid
  config: TrainConfig
  note?: string
}

function toB64Url(s: string): string {
  const bytes = new TextEncoder().encode(s)
  let bin = ""
  bytes.forEach((b) => { bin += String.fromCharCode(b) })
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
}

function fromB64Url(s: string): string {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4)
  const bin = atob(b64)
  return new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)))
}

export function buildTaskUrl(task: SharedTask): string {
  const payload = { v: 1, g: task.grid, c: task.config, n: task.note?.trim() || undefined }
  return `${location.origin}${location.pathname}#task=${toB64Url(JSON.stringify(payload))}`
}

const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x)

function validGrid(g: unknown): g is MazeGrid {
  if (!Array.isArray(g) || g.length < 4 || g.length > 12) return false
  const w = Array.isArray(g[0]) ? g[0].length : 0
  if (w < 4 || w > 12) return false
  let starts = 0, goals = 0
  for (const row of g) {
    if (!Array.isArray(row) || row.length !== w) return false
    for (const v of row) {
      if (!Number.isInteger(v) || v < 0 || v > 5) return false
      if (v === 2) starts++
      if (v === 3) goals++
    }
  }
  return starts === 1 && goals === 1
}

function validConfig(c: unknown): c is TrainConfig {
  if (!c || typeof c !== "object") return false
  const o = c as Record<string, unknown>
  const h = o.hyper as Record<string, unknown> | undefined
  const r = o.reward as Record<string, unknown> | undefined
  return ["qlearning", "ppo", "dqn"].includes(o.algo as string) && isNum(o.seed) && !!h && !!r &&
    ["episodes", "total_timesteps", "learning_rate", "gamma", "epsilon_start", "epsilon_end"].every((k) => isNum(h[k])) &&
    ["step_penalty", "wall_penalty", "goal_reward", "bonus_reward", "trap_penalty"].every((k) => isNum(r[k])) &&
    typeof r.trap_terminal === "boolean" && typeof r.potential_shaping === "boolean"
}

/** 从当前地址栏读取分享的实验任务；没有或格式不合法时返回 null */
export function readTaskFromUrl(): SharedTask | null {
  const m = location.hash.match(/#task=([A-Za-z0-9_-]+)/)
  if (!m) return null
  try {
    const p = JSON.parse(fromB64Url(m[1]))
    if (p?.v !== 1 || !validGrid(p.g) || !validConfig(p.c)) return null
    const note = typeof p.n === "string" ? p.n.slice(0, 300) : undefined
    return { grid: p.g, config: p.c, note }
  } catch {
    return null
  }
}
