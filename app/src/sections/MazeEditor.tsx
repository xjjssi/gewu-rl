// MazeEditor —— 迷宫编辑器：笔刷绘制 / 尺寸切换 / 预设加载 / 随机生成

import { useEffect, useState } from "react"
import { Dices, Eraser, Flag, Gem, Skull, Square, Trash2 } from "lucide-react"

import { MazeCanvas, CELL_LEGEND } from "@/components/MazeCanvas"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { Cell, type MazeGrid, type Preset } from "@/types"

type Brush = number // Cell 值或 -1（擦除）

const BRUSHES: { id: Brush; label: string; icon: React.ReactNode; cls: string }[] = [
  { id: Cell.WALL, label: "墙", icon: <Square className="h-4 w-4" />, cls: "data-[on=true]:bg-slate-600 data-[on=true]:text-white" },
  { id: Cell.BONUS, label: "奖励格", icon: <Gem className="h-4 w-4" />, cls: "data-[on=true]:bg-amber-400 data-[on=true]:text-amber-950" },
  { id: Cell.TRAP, label: "陷阱格", icon: <Skull className="h-4 w-4" />, cls: "data-[on=true]:bg-rose-600 data-[on=true]:text-white" },
  { id: Cell.START, label: "起点", icon: <Flag className="h-4 w-4" />, cls: "data-[on=true]:bg-sky-600 data-[on=true]:text-white" },
  { id: Cell.GOAL, label: "终点", icon: <Flag className="h-4 w-4 rotate-180" />, cls: "data-[on=true]:bg-emerald-600 data-[on=true]:text-white" },
  { id: -1, label: "擦除", icon: <Eraser className="h-4 w-4" />, cls: "data-[on=true]:bg-white data-[on=true]:ring-2 data-[on=true]:ring-brand" },
]

const SIZES = [6, 8, 10, 12]

function blankGrid(size: number): MazeGrid {
  const g: MazeGrid = Array.from({ length: size }, () => Array(size).fill(Cell.EMPTY))
  g[0][0] = Cell.START
  g[size - 1][size - 1] = Cell.GOAL
  return g
}

/** 尺寸切换时尽量保留原图内容，并保证起终点存在 */
function resizeGrid(old: MazeGrid, size: number): MazeGrid {
  const g = blankGrid(size)
  let hasStart = false
  let hasGoal = false
  for (let r = 0; r < Math.min(old.length, size); r++) {
    for (let c = 0; c < Math.min(old[0].length, size); c++) {
      const v = old[r][c]
      if (v === Cell.START) { hasStart = true; g[r][c] = v }
      else if (v === Cell.GOAL) { hasGoal = true; g[r][c] = v }
      else g[r][c] = v
    }
  }
  if (!hasStart) g[0][0] = Cell.START
  if (!hasGoal) g[size - 1][size - 1] = Cell.GOAL
  return g
}

/** 客户端随机迷宫：随机撒墙 + 2 奖励格 + 2 陷阱格（不保证连通，教学上允许） */
function randomGrid(size: number): MazeGrid {
  const g = blankGrid(size)
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++)
      if (g[r][c] === Cell.EMPTY && Math.random() < 0.2) g[r][c] = Cell.WALL
  const empties: [number, number][] = []
  g.forEach((row, r) => row.forEach((v, c) => v === Cell.EMPTY && empties.push([r, c])))
  for (let i = 0; i < Math.min(4, empties.length); i++) {
    const idx = Math.floor(Math.random() * empties.length)
    const [r, c] = empties.splice(idx, 1)[0]
    g[r][c] = i % 2 === 0 ? Cell.BONUS : Cell.TRAP
  }
  return g
}

interface Props {
  grid: MazeGrid
  onChange: (g: MazeGrid) => void
}

export default function MazeEditor({ grid, onChange }: Props) {
  const [brush, setBrush] = useState<Brush>(Cell.WALL)
  const [presets, setPresets] = useState<Preset[]>([])
  const [presetsError, setPresetsError] = useState(false)
  // 加载预设后递增 nonce 强制重置下拉框，否则再选同一个预设不触发 onValueChange
  const [presetNonce, setPresetNonce] = useState(0)

  useEffect(() => {
    fetch("/api/presets")
      .then((r) => r.json())
      .then((d) => { setPresets(d.presets ?? []); setPresetsError(false) })
      .catch(() => { setPresets([]); setPresetsError(true) })
  }, [])

  const paint = (r: number, c: number) => {
    const g = grid.map((row) => [...row])
    if (brush === Cell.START || brush === Cell.GOAL) {
      // 起终点唯一：先清除旧的
      g.forEach((row, rr) => row.forEach((v, cc) => { if (v === brush) g[rr][cc] = Cell.EMPTY }))
      g[r][c] = brush
    } else {
      g[r][c] = brush === -1 ? Cell.EMPTY : brush
    }
    // 擦除/覆盖起终点后自动补到最近的空格，避免产生"无起点/无终点"的非法迷宫
    const ensure = (cell: number) => {
      if (g.some((row) => row.includes(cell))) return
      outer: for (let rr = 0; rr < g.length; rr++)
        for (let cc = 0; cc < g[rr].length; cc++)
          if (g[rr][cc] === Cell.EMPTY) { g[rr][cc] = cell; break outer }
    }
    ensure(Cell.START)
    ensure(Cell.GOAL)
    onChange(g)
  }

  return (
    <Card className="border-slate-200 bg-white">
      <CardHeader className="pb-3">
        <CardTitle className="text-base text-slate-900">迷宫编辑器</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {BRUSHES.map((b) => (
            <Button
              key={b.id}
              size="sm"
              variant="outline"
              data-on={brush === b.id}
              onClick={() => setBrush(b.id)}
              className={cn("border-slate-300 bg-slate-100 text-slate-800 hover:bg-slate-200", b.cls)}
            >
              {b.icon}
              {b.label}
            </Button>
          ))}
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <Label className="text-xs text-slate-500">网格尺寸</Label>
            <Select
              value={String(grid.length)}
              onValueChange={(v) => onChange(resizeGrid(grid, Number(v)))}
            >
              <SelectTrigger className="w-24 border-slate-300 bg-slate-100">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SIZES.map((s) => (
                  <SelectItem key={s} value={String(s)}>{s} × {s}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-slate-500">预设迷宫</Label>
            <Select key={presetNonce} onValueChange={(id) => {
              const p = presets.find((x) => x.id === id)
              if (p) {
                onChange(p.grid.map((row) => [...row]))
                setPresetNonce((n) => n + 1) // 重置为空占位，允许再次选择同一预设
              }
            }}>
              <SelectTrigger className="w-36 border-slate-300 bg-slate-100">
                <SelectValue placeholder="加载预设…" />
              </SelectTrigger>
              <SelectContent>
                {presets.map((p) => (
                  <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            {presetsError && (
              <p className="text-xs text-rose-600">
                预设加载失败，请先启动后端（uvicorn main:app --port 8000）
              </p>
            )}
          </div>
          <Button size="sm" variant="outline" className="border-slate-300 bg-slate-100 text-slate-800 hover:bg-slate-200"
            onClick={() => onChange(randomGrid(grid.length))}>
            <Dices className="h-4 w-4" /> 随机迷宫
          </Button>
          <Button size="sm" variant="outline" className="border-slate-300 bg-slate-100 text-slate-800 hover:bg-slate-200"
            onClick={() => onChange(blankGrid(grid.length))}>
            <Trash2 className="h-4 w-4" /> 清空
          </Button>
        </div>

        <MazeCanvas grid={grid} editable onCell={paint} />

        <div className="flex flex-wrap gap-3 text-xs text-slate-500">
          {CELL_LEGEND.map((l) => (
            <span key={l.label} className="flex items-center gap-1">
              <span className={cn("inline-block h-3 w-3 rounded-sm", l.cls)} /> {l.label}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
