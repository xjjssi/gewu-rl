// useTrainingSocket —— 训练 WebSocket 连接与实时状态管理
//
// 流程：POST /api/train/start 拿到 session_id → 打开 /ws/train/{id} →
// 逐事件更新状态。step 事件进入 stepStream 供 LiveView 定速回放。

import { useCallback, useRef, useState } from "react"
import type {
  Algo,
  EpisodePoint,
  TrainEvent,
  TrainRequestBody,
  TrainSummary,
} from "@/types"

export type TrainStatus = "idle" | "running" | "done" | "error" | "stopped"

export interface TrajectoryRecord {
  episode: number
  cells: [number, number][]
}

export interface TrainingState {
  status: TrainStatus
  sessionId: string | null
  algo: Algo | null
  episodes: EpisodePoint[]
  /** 全部 step 关键帧（按到达顺序），LiveView 定速回放 */
  stepStream: [number, number][]
  /** 最近一条代表性轨迹 */
  trajectory: [number, number][]
  /** 全部代表性轨迹（按回合），训练结束后进入回放模式用 */
  trajectories: TrajectoryRecord[]
  currentEpisode: number
  epsilon: number | null
  summary: TrainSummary | null
  error: string | null
}

const initialState: TrainingState = {
  status: "idle",
  sessionId: null,
  algo: null,
  episodes: [],
  stepStream: [],
  trajectory: [],
  trajectories: [],
  currentEpisode: 0,
  epsilon: null,
  summary: null,
  error: null,
}

/** 后端未启动时的统一中文提示 */
const BACKEND_HINT = "无法连接后端服务，请先启动后端（uvicorn main:app --port 8000）"

export function useTrainingSocket() {
  const [state, setState] = useState<TrainingState>(initialState)
  const wsRef = useRef<WebSocket | null>(null)
  const sessionRef = useRef<string | null>(null)

  const stop = useCallback(async () => {
    if (sessionRef.current) {
      await fetch(`/api/train/${sessionRef.current}/stop`, { method: "POST" }).catch(() => {})
    }
  }, [])

  const reset = useCallback(() => {
    wsRef.current?.close()
    setState(initialState)
  }, [])

  const start = useCallback(async (body: TrainRequestBody) => {
    wsRef.current?.close()
    setState({ ...initialState, status: "running", algo: body.algo })
    try {
      const resp = await fetch("/api/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!resp.ok) throw new Error(`启动失败：HTTP ${resp.status} ${await resp.text()}`)
      const { session_id } = (await resp.json()) as { session_id: string }
      sessionRef.current = session_id
      setState((s) => ({ ...s, sessionId: session_id }))

      const proto = location.protocol === "https:" ? "wss" : "ws"
      const ws = new WebSocket(`${proto}://${location.host}/ws/train/${session_id}`)
      wsRef.current = ws

      ws.onmessage = (msg) => {
        const ev = JSON.parse(msg.data as string) as TrainEvent
        setState((s) => {
          switch (ev.type) {
            case "episode":
              return {
                ...s,
                episodes: [...s.episodes, ev],
                currentEpisode: ev.episode,
                epsilon: ev.epsilon,
              }
            case "step":
              return { ...s, stepStream: [...s.stepStream, ev.pos] }
            case "trajectory":
              return { ...s, trajectory: ev.cells, trajectories: [...s.trajectories, { episode: ev.episode, cells: ev.cells }] }
            case "done":
              return { ...s, status: ev.summary.stopped ? "stopped" : "done", summary: ev.summary }
            case "error":
              return { ...s, status: "error", error: ev.message }
            default:
              return s
          }
        })
      }
      ws.onerror = () =>
        setState((s) => (s.status === "running" ? { ...s, status: "error", error: `WebSocket 连接异常，${BACKEND_HINT}` } : s))
    } catch (e) {
      // fetch 网络层失败（后端未启动）抛 TypeError，给出明确的中文排查提示
      const msg = e instanceof TypeError ? BACKEND_HINT : e instanceof Error ? e.message : String(e)
      setState((s) => ({ ...s, status: "error", error: msg }))
    }
  }, [])

  return { state, start, stop, reset }
}
