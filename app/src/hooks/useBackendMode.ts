// useBackendMode —— 读取运行模式（完整版 / 在线体验版），检测结果全局缓存
import { useEffect, useState } from "react"

import { detectBackend, type BackendMode } from "@/lib/backend"

export function useBackendMode(): BackendMode | null {
  const [mode, setMode] = useState<BackendMode | null>(null)
  useEffect(() => {
    detectBackend().then(setMode)
  }, [])
  return mode
}
