// MazeCanvas —— 迷宫网格渲染（编辑器与实时视图共用）

import { cn } from "@/lib/utils"
import { Cell, type MazeGrid, type PolicyCell } from "@/types"

const CELL_STYLE: Record<number, string> = {
  [Cell.EMPTY]: "bg-slate-100 hover:bg-slate-200",
  [Cell.WALL]: "bg-slate-600",
  [Cell.START]: "bg-sky-600",
  [Cell.GOAL]: "bg-emerald-600",
  [Cell.BONUS]: "bg-amber-400",
  [Cell.TRAP]: "bg-rose-600",
}

export const CELL_LEGEND = [
  { label: "起点", cls: "bg-sky-600" },
  { label: "终点", cls: "bg-emerald-600" },
  { label: "墙", cls: "bg-slate-600" },
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
  // 发散配色：橙(低) → 白 → 蓝(高)，与 TensorFlow Playground 的正负值配色一致
  const t = (v - min) / (max - min)
  const [lo, hi] = [[245, 147, 34], [8, 119, 189]]
  const [c, k] = t < 0.5 ? [lo, 1 - t * 2] : [hi, t * 2 - 1]
  const mix = (i: number) => Math.round(255 + (c[i] - 255) * k * 0.85)
  return `rgb(${mix(0)},${mix(1)},${mix(2)})`
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
      className="inline-grid gap-[2px] rounded-md border border-slate-200 bg-white p-2 select-none"
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
                <div className="absolute inset-[3px] rounded-[3px] bg-sky-500/20" />
              )}
              {p && !isAgent && cell !== Cell.GOAL && (
                <span className={cn("absolute text-[13px] leading-none", showHeatmap ? "text-slate-800" : "text-brand")}>
                  {ARROWS[p.action]}
                </span>
              )}
              {cell === Cell.START && !isAgent && <span className="text-[10px] text-white/90">起</span>}
              {cell === Cell.GOAL && <span className="text-[10px] text-white">终</span>}
              {cell === Cell.BONUS && <span className="text-[10px] text-amber-900">+</span>}
              {cell === Cell.TRAP && <span className="text-[10px] text-white">×</span>}
              {isAgent && (
                <div className="absolute inset-[6px] rounded-full border-2 border-white bg-ink shadow-sm" />
              )}
            </div>
          )
        }),
      )}
      {h === 0 && <div className="text-slate-500">空迷宫</div>}
    </div>
  )
}
