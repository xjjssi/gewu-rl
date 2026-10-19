# 格物RL · 前端（gewu-rl-web）

可视化强化学习教学实验平台的前端，基于 React 19 + TypeScript + Vite + Tailwind CSS + shadcn/ui。

## 功能

- **迷宫编辑器**：笔刷绘制墙 / 奖励格 / 陷阱格 / 起终点，支持预设与随机生成
- **训练控制台**：表格 Q-Learning（从零实现）与 PPO / DQN（Stable-Baselines3），超参按算法分别管理
- **训练实时视图**：实时曲线 + 智能体动画，训练结束后支持按回合回放（播放 / 暂停 / 倍速 / 进度拖动）
- **AI 奖励设计师**：自然语言教学意图 → 奖励函数 → 双智能体对照实验 → 意图达成度自动解读
- **实验对比**：历史 run 曲线叠加、课程学习 vs 直接训练对照

## 开发

```bash
npm install
npm run dev    # 默认端口 3000，/api 与 /ws 代理到 8000
```

需先启动后端（项目根目录的 `backend/`）：

```bash
cd backend && uvicorn main:app --port 8000
```

## 构建

```bash
npm run build
```
