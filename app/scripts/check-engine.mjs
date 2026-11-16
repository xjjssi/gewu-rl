// check-engine.mjs —— 浏览器训练引擎（src/lib/localEngine.ts）的一致性检查
// 在 Node 中运行前端引擎：4 个预设 × 10 个种子 × 600 回合，统计统一口径评估成功率，
// 输出 JSON 供 tests/test_browser_engine.py 与后端 Python 实现对照。
// 用法：node scripts/check-engine.mjs   （CI 中也会运行）
import { build } from "esbuild"
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const outdir = mkdtempSync(join(tmpdir(), "gewu-engine-"))
await build({
  entryPoints: [join(here, "../src/lib/localEngine.ts"), join(here, "../src/lib/presets.ts")],
  outdir, bundle: true, format: "esm", platform: "node", logLevel: "error",
  alias: { "@": join(here, "../src") },
})
const { trainLocal } = await import(pathToFileURL(join(outdir, "localEngine.js")).href)
const { LOCAL_PRESETS } = await import(pathToFileURL(join(outdir, "presets.js")).href)

const reward = { step_penalty: -0.01, wall_penalty: -0.05, goal_reward: 1, bonus_reward: 0.5, trap_penalty: -1,
  trap_terminal: false, potential_shaping: false }
const result = {}
for (const p of LOCAL_PRESETS) {
  const rates = []
  for (let seed = 0; seed < 10; seed++) {
    let summary = null
    await trainLocal({ maze: p.grid, reward, episodes: 600, learning_rate: 0.1, gamma: 0.99, epsilon_start: 1,
      epsilon_end: 0.05, seed }, (ev) => { if (ev.type === "done") summary = ev.summary }, () => false)
    rates.push(summary.eval_success_rate)
  }
  result[p.id] = rates.reduce((a, b) => a + b, 0) / rates.length
}
console.log(JSON.stringify(result))
