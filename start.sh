#!/usr/bin/env bash
# 格物RL 一键启动（macOS / Linux）：检查环境、首次安装依赖、启动后端与前端，Ctrl+C 一并停止
set -euo pipefail
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "[格物RL] 未找到 Python 3.10+：https://www.python.org/downloads/"; exit 1; }
command -v npm >/dev/null || { echo "[格物RL] 未找到 Node.js 20.19+：https://nodejs.org/"; exit 1; }
PY=python3
if ! python3 -c "import fastapi, uvicorn, gymnasium, stable_baselines3" 2>/dev/null; then
  if [ ! -x .venv/bin/python ]; then
    echo "[格物RL] 首次运行：创建虚拟环境并安装后端依赖（需联网，约 5-10 分钟）..."
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
  fi
  PY=.venv/bin/python
fi
if [ ! -d app/node_modules ]; then
  echo "[格物RL] 首次运行：安装前端依赖..."
  (cd app && npm ci)
fi
"$PY" -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 &
BACK=$!
(cd app && npm run dev -- --host 127.0.0.1 --port 3000) &
FRONT=$!
trap 'kill $BACK $FRONT 2>/dev/null' EXIT INT TERM
echo "[格物RL] 后端 http://127.0.0.1:8000 ，前端 http://localhost:3000 （Ctrl+C 停止）"
sleep 6
(command -v open >/dev/null && open http://localhost:3000) || (command -v xdg-open >/dev/null && xdg-open http://localhost:3000) || true
wait
