"""test_browser_engine.py —— 在线体验版的浏览器训练引擎与后端 Python 实现的一致性。

运行 app/scripts/check-engine.mjs（Node 中执行前端 localEngine.ts）：4 个预设 × 10 个种子 × 600 回合，
统一口径评估成功率应与后端表格 Q-Learning 一致（正式实验 E1：四个预设 20 种子均为 100%）。
未安装 Node 或前端依赖时自动跳过。
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"


@pytest.mark.skipif(shutil.which("node") is None or not (APP / "node_modules" / "esbuild").exists(),
                    reason="需要 Node.js 与前端依赖（cd app && npm ci）")
def test_browser_engine_matches_backend_success():
    out = subprocess.run(["node", "scripts/check-engine.mjs"], cwd=APP, capture_output=True, text=True,
                         timeout=600, check=True).stdout
    rates = json.loads(out.strip().splitlines()[-1])
    assert set(rates) == {"empty_6x6", "single_wall_6x6", "multi_wall_8x8", "big_maze_10x10"}
    for preset, rate in rates.items():
        assert rate >= 0.95, (preset, rate)
