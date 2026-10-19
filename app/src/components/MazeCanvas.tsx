// MazeCanvas —— 迷宫网格渲染（编辑器与实时视图共用）

import { cn } from "@/lib/utils"
import { Cell, type MazeGrid, type PolicyCell } from "@/types"

const CELL_STYLE: Record<number, string> = {
  [Cell.EMPTY]: "bg-slate-800/60 hover:bg-slate-700/60",
  [Cell.WALL]: "bg-slate-500",
  [Cell.START]: "bg-sky-500",
  [Cell.GOAL]: "bg-emerald-500",
  [Cell.BONUS]: "bg-amber-400",
  [Cell.TRAP]: "bg-rose-600",
}

export const CELL_LEGEND = [
  { label: "起点", cls: "bg-sky-500" },
  { label: "终点", cls: "bg-emerald-500" },
  { label: "墙", cls: "bg-slate-500" },
  { label: "奖励格", cls: "bg-amber-400" },
  { label: "陷阱格", cls: "bg-rose-600" },
]

const ARROWS = ["↑", "↓", "←", "→"]

interface Props {
  grid: MazeGrid
  agentPos?: [number, number] | null
  trail?: [number, number][]
  policy?: PolicyCell[] | null
  showHeatmap?: boolean
  editable?: boolean
  onCell?: (r: number, c: number) => void
}

function heatColor(v: number, min: number, max: number): string {
  if (max - min < 1e-9) return "transparent"
  const t = (v - min) / (max - min) // 0..1，蓝(低)→红(高)
  const r = Math.round(37 + t * 200)
  const g = Math.round(99 + t * 40)
  const b = Math.round(235 - t * 160)
  return `rgba(${r},${g},${b},${0.25 + 0.55 * t})`
}

export function MazeCanvas({ grid, agentPos, trail, policy, showHeatmap, editable, onCell }: Props) {
  const h = grid.length
  const w = grid[0]?.length ?? 0
  const trailSet = new Set((trail ?? []).map(([r, c]) => `${r},${c}`))
  const policyMap = new Map<string, PolicyCell>()
  let qMin = Infinity
  let qMax = -Infinity
  for (const p of policy ?? []) {
    policyMap.set(`${p.pos[0]},${p.pos[1]}`, p)
    qMin = Math.min(qMin, p.q_max)
    qMax = Math.max(qMax, p.q_max)
  }

  return (
    <div
      className="inline-grid gap-[3px] rounded-lg bg-slate-900/80 p-2 select-none"
      style={{ gridTemplateColumns: `repeat(${w}, minmax(0,1fr))` }}
    >
      {grid.flatMap((row, r) =>
        row.map((cell, c) => {
          const key = `${r}-${c}`
          const isAgent = agentPos && agentPos[0] === r && agentPos[1] === c
          const inTrail = trailSet.has(`${r},${c}`)
          const p = policyMap.get(`${r},${c}`)
          return (
            <div
              key={key}
              onClick={() => editable && onCell?.(r, c)}
              className={cn(
                "relative flex aspect-square w-9 items-center justify-center rounded-[4px] text-xs font-bold transition-colors",
                CELL_STYLE[cell] ?? CELL_STYLE[Cell.EMPTY],
                editable && "cursor-crosshair",
              )}
              style={
                showHeatmap && p
                  ? { backgroundColor: heatColor(p.q_max, qMin, qMax) }
                  : undefined
              }
            >
              {inTrail && !isAgent && (
                <div className="absolute inset-[3px] rounded-[3px] bg-cyan-400/25" />
              )}
              {p && !isAgent && cell !== Cell.GOAL && (
                <span className={cn("absolute text-[13px] leading-none", showHeatmap ? "text-white/90" : "text-cyan-300")}>
                  {ARROWS[p.action]}
                </span>
              )}
              {cell === Cell.START && !isAgent && <span className="text-[10px] text-white/90">起</span>}
              {cell === Cell.GOAL && <span className="text-[10px] text-white">终</span>}
              {cell === Cell.BONUS && <span className="text-[10px] text-amber-900">+</span>}
              {cell === Cell.TRAP && <span className="text-[10px] text-white">×</span>}
              {isAgent && (
                <div className="absolute inset-[5px] rounded-full bg-fuchsia-400 shadow-[0_0_10px_2px_rgba(232,121,249,0.7)]" />
              )}
            </div>
          )
        }),
      )}
      {h === 0 && <div className="text-slate-400">空迷宫</div>}
    </div>
  )
}
