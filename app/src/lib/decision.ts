// decision.ts —— 决策解说的共享常量与统计（解说卡、步骤条、迷宫箭头、人机 PK 共用）

import type { StepDecision } from "@/types"

export const ACTION_ARROWS = ["↑", "↓", "←", "→"]
export const ACTION_NAMES = ["上", "下", "左", "右"]

/** 探索 / 寻优 的统一配色 */
export const MODE_STYLE = {
  explore: { label: "探索", text: "text-orange-700", bg: "bg-orange-500", soft: "bg-orange-50 border-orange-200" },
  exploit: { label: "寻优", text: "text-sky-800", bg: "bg-sky-700", soft: "bg-sky-50 border-sky-200" },
} as const

/** 回合级统计：探索 / 寻优步数占比 */
export function modeStats(decisions: StepDecision[]) {
  const n = decisions.length
  const explore = decisions.reduce((k, d) => k + d.x, 0)
  return { n, explore, exploit: n - explore, exploreRate: n ? explore / n : 0 }
}
