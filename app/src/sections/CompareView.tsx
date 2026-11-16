// CompareView —— 实验对比：历史 run 表格 + 曲线叠加 + 课程学习对照

import { useCallback, useEffect, useMemo, useState } from "react"
import { FlaskConical, RefreshCw } from "lucide-react"
import { CartesianGrid, Legend, Line, LineChart, XAxis, YAxis } from "recharts"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { OfflineNotice } from "@/components/OfflineNotice"
import { useBackendMode } from "@/hooks/useBackendMode"
import type { CurriculumResult, RunDetail, RunMeta } from "@/types"

const COLORS = ["#0877bd", "#2e8b57", "#f59322", "#d1495b", "#e8590c", "#6a4c93"]

function movingAvg(xs: number[], w: number): number[] {
  const out: number[] = []
  let sum = 0
  for (let i = 0; i < xs.length; i++) {
    sum += xs[i]
    if (i >= w) sum -= xs[i - w]
    out.push(sum / Math.min(i + 1, w))
  }
  return out
}

const ALGO_NAME: Record<string, string> = { qlearning: "表格Q学习", ppo: "PPO", dqn: "DQN" }

/** 图例命名：算法 + 实验标签（如有）+ 种子编号，避免两条曲线同名 */
function runName(cfg: { algo: string; seed: number; label?: string | null }): string {
  const algo = ALGO_NAME[cfg.algo] ?? cfg.algo
  return cfg.label ? `${algo}·${cfg.label}#${cfg.seed}` : `${algo}#${cfg.seed}`
}

const PAGE_SIZE = 8 // run 列表每页条数

export default function CompareView({ active = true }: { active?: boolean }) {
  const mode = useBackendMode()
  const [runs, setRuns] = useState<RunMeta[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [details, setDetails] = useState<Record<string, RunDetail>>({})
  const [curriculum, setCurriculum] = useState<CurriculumResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [page, setPage] = useState(0)
  const [loadError, setLoadError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    fetch("/api/experiments")
      .then((r) => r.json())
      .then((d) => { setRuns(d.runs ?? []); setPage(0); setLoadError(null) })
      .catch(() => setLoadError("无法加载实验列表，请先启动后端（uvicorn main:app --port 8000）"))
  }, [])

  // 每次切换到本页时自动重新拉取列表（意图实验产生的新 run 才能立刻出现）
  useEffect(() => {
    if (active) refresh()
  }, [active, refresh])

  const totalPages = Math.max(1, Math.ceil(runs.length / PAGE_SIZE))
  const pageRuns = runs.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const toggle = (id: string) => {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setSelected(next)
    if (!details[id]) {
      fetch(`/api/experiments/${id}`)
        .then((r) => r.json())
        .then((d: RunDetail) => setDetails((prev) => ({ ...prev, [id]: d })))
        .catch(() => {})
    }
  }

  const runCurriculum = async () => {
    setBusy(true)
    try {
      const r = await fetch("/api/train/curriculum", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ episodes_per_stage: 150, seed: 0 }),
      })
      if (!r.ok) return // 请求失败时保持原状，避免把错误响应当成结果渲染
      setCurriculum(await r.json())
    } finally {
      setBusy(false)
    }
  }

  // 叠加曲线：不同 run 长度不同，按 episode 对齐合并
  const overlayData = useMemo(() => {
    const ids = [...selected].filter((id) => details[id])
    if (ids.length === 0) return []
    const series = ids.map((id, i) => ({
      key: `r${i}`,
      name: runName(details[id].config),
      curve: movingAvg(details[id].metrics.map((m) => m.reward), 20),
    }))
    const maxLen = Math.max(...series.map((s) => s.curve.length))
    const rows: Record<string, number | null>[] = []
    for (let i = 0; i < maxLen; i++) {
      const row: Record<string, number | null> = { episode: i + 1 }
      series.forEach((s) => { row[s.key] = i < s.curve.length ? Number(s.curve[i].toFixed(3)) : null })
      rows.push(row)
    }
    return rows.map((r) => ({ ...r, __names: series.map((s) => s.name) }))
  }, [selected, details])

  const overlaySeries = useMemo(
    () =>
      [...selected].filter((id) => details[id]).map((id, i) => ({
        key: `r${i}`,
        name: runName(details[id].config),
        color: COLORS[i % COLORS.length],
      })),
    [selected, details],
  )

  // 课程学习曲线：各级曲线按顺序拼接，与直接训练对照
  const curriculumData = useMemo(() => {
    if (!curriculum) return []
    const rows: { episode: number; curriculum: number | null; baseline: number | null }[] = []
    let offset = 0
    for (const st of curriculum.stages) {
      const c = movingAvg(st.curve.map((p) => p.reward), 20)
      c.forEach((v, i) => rows.push({ episode: offset + i + 1, curriculum: Number(v.toFixed(3)), baseline: null }))
      offset += c.length
    }
    const base = movingAvg(curriculum.baseline.curve.map((p) => p.reward), 20)
    base.forEach((v, i) => {
      if (i < rows.length) rows[i].baseline = Number(v.toFixed(3))
      else rows.push({ episode: i + 1, curriculum: null, baseline: Number(v.toFixed(3)) })
    })
    return rows
  }, [curriculum])

  const overlayConfig = Object.fromEntries(
    overlaySeries.map((s) => [s.key, { label: s.name, color: s.color }]),
  ) satisfies ChartConfig

  return (
    <div className="space-y-4">
      {mode === "offline" && <OfflineNotice feature="实验对比与课程学习（实验记录落盘、多次实验叠加、课程学习对照）" />}
      <Card className="border-slate-200 bg-white">
        <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
          <CardTitle className="text-base text-slate-900">历史实验（勾选 2 条以上叠加对比）</CardTitle>
          <Button size="sm" variant="outline" className="border-slate-300 bg-slate-100 text-slate-800 hover:bg-slate-200" onClick={refresh}>
            <RefreshCw className="h-4 w-4" /> 刷新
          </Button>
        </CardHeader>
        <CardContent>
          {loadError ? (
            <p className="text-sm text-rose-600">{loadError}</p>
          ) : runs.length === 0 ? (
            <p className="text-sm text-slate-500">暂无历史实验，请先在「训练实时视图」完成一次训练。</p>
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow className="border-slate-200">
                    <TableHead className="w-10"></TableHead>
                    <TableHead>时间</TableHead>
                    <TableHead>算法</TableHead>
                    <TableHead>标签</TableHead>
                    <TableHead>种子</TableHead>
                    <TableHead>回合数</TableHead>
                    <TableHead>评估成功率</TableHead>
                    <TableHead>状态</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pageRuns.map((r) => (
                    <TableRow key={r.id} className="border-slate-200">
                      <TableCell>
                        <Checkbox checked={selected.has(r.id)} onCheckedChange={() => toggle(r.id)} />
                      </TableCell>
                      <TableCell className="font-mono text-xs">{r.created_at}</TableCell>
                      <TableCell>{ALGO_NAME[r.algo] ?? r.algo}</TableCell>
                      <TableCell>{r.label ?? "—"}</TableCell>
                      <TableCell>{r.seed}</TableCell>
                      <TableCell>{r.episodes_run ?? "—"}</TableCell>
                      <TableCell>{r.eval_success_rate != null ? `${(r.eval_success_rate * 100).toFixed(0)}%` : "—"}</TableCell>
                      <TableCell>
                        <Badge className={r.stopped ? "bg-amber-600" : "bg-sky-600"}>
                          {r.stopped ? "已停止" : "完成"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {totalPages > 1 && (
                <div className="mt-3 flex items-center justify-end gap-2 text-xs text-slate-500">
                  <Button size="sm" variant="outline" disabled={page === 0}
                    className="border-slate-300 bg-slate-100 text-slate-800 hover:bg-slate-200"
                    onClick={() => setPage((p) => Math.max(0, p - 1))}>
                    上一页
                  </Button>
                  <span>第 {page + 1} / {totalPages} 页（共 {runs.length} 条）</span>
                  <Button size="sm" variant="outline" disabled={page >= totalPages - 1}
                    className="border-slate-300 bg-slate-100 text-slate-800 hover:bg-slate-200"
                    onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}>
                    下一页
                  </Button>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {overlaySeries.length > 0 && (
        <Card className="border-slate-200 bg-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-slate-900">回报曲线叠加（20 回合滑动平均）</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={overlayConfig} className="h-64 w-full">
              <LineChart data={overlayData} margin={{ left: 8, right: 8, top: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="episode" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} width={50} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Legend />
                {overlaySeries.map((s) => (
                  <Line key={s.key} type="monotone" dataKey={s.key} name={s.name}
                    stroke={s.color} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                ))}
              </LineChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}

      <Card className="border-slate-200 bg-white">
        <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-base text-slate-900">课程学习对照</CardTitle>
          <Button size="sm" className="bg-orange-600 hover:bg-orange-700" onClick={runCurriculum} disabled={busy}>
            <FlaskConical className="h-4 w-4" /> {busy ? "运行中…" : "运行课程学习对照"}
          </Button>
        </CardHeader>
        <CardContent>
          {!curriculum ? (
            <p className="text-sm text-slate-500">
              点击按钮后，后端将按「逆向课程」逐级训练表格 Q-Learning：同一张大迷宫，
              起点由近及远（终点隔壁 → 中场 → 中远程 → 全程），Q 表逐级迁移，
              并与「直接在全程起点训练同等总回合数」对照。
            </p>
          ) : (
            <div className="space-y-3">
              {/* 两组最终评估成功率（贪心策略在最终迷宫上各评 20 回合） */}
              <div className="flex flex-wrap gap-4 rounded-md border border-slate-200 bg-slate-50 px-4 py-2 text-sm">
                <span className="text-slate-700">
                  课程学习 评估成功率：
                  <span className="font-mono font-bold text-orange-700">
                    {(curriculum.eval.curriculum_success * 100).toFixed(0)}%
                  </span>
                </span>
                <span className="text-slate-700">
                  直接训练 评估成功率：
                  <span className="font-mono font-bold text-slate-900">
                    {(curriculum.eval.baseline_success * 100).toFixed(0)}%
                  </span>
                </span>
              </div>
              <ChartContainer
                config={{
                  curriculum: { label: "课程学习（逐级）", color: "#e8590c" },
                  baseline: { label: curriculum.baseline.name, color: "#64748b" },
                } satisfies ChartConfig}
                className="h-64 w-full"
              >
                <LineChart data={curriculumData} margin={{ left: 8, right: 8, top: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="episode" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} width={50} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Legend />
                  <Line type="monotone" dataKey="curriculum" name="课程学习（逐级）" stroke="var(--color-curriculum)" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="baseline" name="直接训练（最终迷宫）" stroke="var(--color-baseline)" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                </LineChart>
              </ChartContainer>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
