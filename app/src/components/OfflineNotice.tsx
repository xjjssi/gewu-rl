// OfflineNotice —— 在线体验版中，需要本地后端的功能给出说明与启动方式

import { Server } from "lucide-react"

export function OfflineNotice({ feature }: { feature: string }) {
  return (
    <div className="flex gap-3 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
      <Server className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="space-y-1">
        <p>
          <b>{feature}</b>需要本地后端（Python：多种子对照实验、规则引擎 / 大模型接口、实验记录落盘）。
          当前为<b>在线体验版</b>，训练、决策解说、价值传播和人机 PK 已在浏览器中运行。
        </p>
        <p className="text-xs text-amber-800">
          完整版：下载源码后执行 <code className="rounded bg-amber-100 px-1">start.bat</code>（Windows）或
          <code className="rounded bg-amber-100 px-1">./start.sh</code>（macOS / Linux），详见仓库 README。
        </p>
      </div>
    </div>
  )
}
