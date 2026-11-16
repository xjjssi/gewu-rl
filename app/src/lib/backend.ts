// backend.ts —— 运行模式检测：完整版（已连接本地后端）/ 在线体验版（浏览器内运行）
//
// 静态部署（GitHub Pages 等）构建时设置 VITE_STATIC=1，直接进入浏览器模式；
// 否则探测 /api/presets：返回 JSON 即为完整版，网络失败或返回非 JSON（如静态站点的 404 页）即为浏览器模式。

export type BackendMode = "online" | "offline"

const STATIC = import.meta.env.VITE_STATIC === "1"
let probe: Promise<BackendMode> | null = null

export function detectBackend(): Promise<BackendMode> {
  if (STATIC) return Promise.resolve("offline")
  probe ??= fetch("/api/presets")
    .then((r) => (r.ok && (r.headers.get("content-type") ?? "").includes("json") ? "online" : "offline") as BackendMode)
    .catch(() => "offline" as BackendMode)
  return probe
}
