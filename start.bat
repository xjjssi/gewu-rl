@echo off
chcp 65001 >nul
rem 格物RL 一键启动（Windows）：自动检查环境、首次安装依赖、启动后端与前端并打开浏览器
cd /d "%~dp0"
where python >nul 2>nul || (echo [格物RL] 未找到 Python 3.10+，请先安装：https://www.python.org/downloads/ & pause & exit /b 1)
where npm >nul 2>nul || (echo [格物RL] 未找到 Node.js 20.19+，请先安装：https://nodejs.org/ & pause & exit /b 1)
set "PY=python"
python -c "import fastapi, uvicorn, gymnasium, stable_baselines3" >nul 2>nul && goto :frontend
if not exist ".venv\Scripts\python.exe" (
  echo [格物RL] 首次运行：创建虚拟环境并安装后端依赖（需联网，约 5-10 分钟）...
  python -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)
set "PY=%~dp0.venv\Scripts\python.exe"
:frontend
if not exist "app\node_modules" (
  echo [格物RL] 首次运行：安装前端依赖...
  pushd app
  call npm ci || (popd & goto :fail)
  popd
)
echo [格物RL] 启动后端 http://127.0.0.1:8000 与前端 http://localhost:3000 ...
start "格物RL 后端" cmd /k ""%PY%" -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000"
start "格物RL 前端" cmd /k "cd /d "%~dp0app" && npm run dev -- --host 127.0.0.1 --port 3000"
timeout /t 8 >nul
if not defined GEWU_NO_BROWSER start "" http://localhost:3000
echo [格物RL] 已在浏览器打开；关闭两个命令行窗口即可停止。
exit /b 0
:fail
echo [格物RL] 安装失败：请检查网络后重试，或按 README 手动安装。
pause
exit /b 1
