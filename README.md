# 格物RL · 可视化强化学习教学实验平台

「格物RL」是一个面向强化学习教学的全栈实验平台：在浏览器里搭建网格迷宫，选择算法
（表格 Q-Learning / PPO / DQN）与奖励塑形配置，实时观看智能体的学习过程，
训练完成后叠加策略箭头与 Q 值热力图，并可对比多次实验曲线、演示课程学习对照。

> “格物致知”——通过亲手搭建环境、调整奖励、观察曲线，理解强化学习的核心概念：
> 探索与利用、折现因子、奖励塑形（PBRS）、函数逼近与表格法的差异。

## ✨ 核心创新：AI 奖励设计师

用**中文自然语言**描述教学目标（例如"让智能体学会绕开陷阱，宁愿绕远路也不冒险"），
系统自动完成「意图 → 奖励函数 → RL 训练 → 数据验证意图是否达成」的完整闭环：

1. **意图翻译**：LLM（OpenAI 兼容接口，如 Moonshot/Kimi）把意图翻译成结构化
   `RewardConfig` + 中文设计理由；未配置 Key 时自动切换到**内置规则引擎**
   （识别"陷阱/奖励格/快速/探索"等意图词），演示永不受网络影响；
   LLM 输出经过 JSON 鲁棒提取 + 逐字段 clamp 安全校验，任何失败自动降级。
2. **对照实验**：同一迷宫、同一算法、同一组种子（默认 5 个，可配置 1~8），分别训练
   **三组**智能体——「默认方案 / 默认+PBRS / AI 方案」（均落盘 `runs/`，可在「实验对比」页叠加曲线）。
   「默认+PBRS」组用于剥离势函数塑形的贡献：AI 的价值体现在第三组相对第二组的增量，
   而不是"PBRS 开 vs 关"。指标按多种子汇报**均值 ± 标准差**。
3. **意图达成度验证**：统一评估口径（ε=0.05 微小探索 + Q 值平局偏向终点 + 环路检测）跑 20 回合，
   对比成功率、平均步数、奖励格收集率、陷阱触发率、平均回合回报与绕圈率，
   并自动生成中文解读（含"过度约束警告"等教学提示）。

**LLM 配置方式（Key 绝不落盘）**：
- 前端「AI 奖励设计师」tab 的 LLM 设置区按次填写（仅存在于当次请求内存）；
- 或设置环境变量 `GEWU_LLM_BASE_URL` / `GEWU_LLM_API_KEY` / `GEWU_LLM_MODEL`；
- 两者都没有 → 直接使用规则引擎。

**演示推荐**：
- 在默认「单障碍 6×6」迷宫上输入收集类意图，可看到奖励格收集率与统一标尺回报的变化；
  输入"对陷阱零容忍"等强约束词会开启"踩陷阱即终止"。
- **最有冲击力的演示**：选「多障碍 8×8」、150 回合，输入"收集所有奖励格，哪怕绕路也要
  最终到达终点"——默认+PBRS 组直奔终点、奖励格收集率 0%；AI 方案学会"先绕路收集再到达"
  （20 种子正式数据：收集率 90.5%±15.6%，成功率 95.8%±8.3%）。单开 PBRS 做不到：
  势函数把智能体拉向终点方向，恰好抑制绕路收集——AI 的价值是把多目标权衡意图翻译成奖励配比。
- **另一个重要教学点**："默认奖励学不会"并不准确。有限预算下默认组训练曲线同样收敛，
  但纯贪心评估可能因 Q 值近似平局而绕圈（大迷宫 150 回合、20 种子：纯贪心成功率 35%±48%，
  统一口径评估 100%）。统一评估口径把"没学会"与"学会了但贪心未稳定"分开报告；
  PBRS 的真实收益是收敛更快（22.4 vs 29.2 回合）且贪心策略更稳定（纯贪心 100% vs 35%）。
  正式数据见 `backend/results/report/report_results.json`（20 种子）。

**相关接口**：`POST /api/reward-design`（仅生成方案）、`POST /api/intent-experiment`（完整闭环）。

> 实现细节：为了让"奖励格只领一次"的行为可被表格法正确学习，`GridMazeEnv` 的
> 离散状态 = 格子坐标 × 2^奖励格数 + 收集掩码（否则 Q 表会把已领走的奖励当成
> "幻影价值"，贪心策略会在奖励格旁打转——这本身也是一堂生动的 MDP 建模课）。


## 架构

```
┌─────────────────────────────┐        REST /api/*         ┌──────────────────────────────┐
│  前端 app/                   │  ────────────────────────▶ │  后端 backend/ (FastAPI)      │
│  React 19 + TS + Vite        │                            │  · 训练线程 + 事件缓冲         │
│  Tailwind + shadcn/ui        │  ◀──────────────────────── │  · runs/ 实验落盘             │
│  迷宫编辑器 / 实时动画        │   WebSocket /ws/train/{id} │  · 课程学习端点               │
│  实时曲线 / 实验对比          │   episode/trajectory/      │                              │
└─────────────────────────────┘   step/done 事件流          └──────────────┬───────────────┘
                                                                            │ import
                                                            ┌───────────────▼───────────────┐
                                                            │  教学包 gridworld_edu/ (MIT)   │
                                                            │  · GridMazeEnv (Gymnasium)     │
                                                            │  · 从零实现的表格 Q-Learning    │
                                                            │  · SB3 PPO/DQN 封装            │
                                                            │  · 预设迷宫 + 课程序列          │
                                                            └───────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装后端依赖（已按 requirements.txt 锁定）
pip install -r requirements.txt

# 2. 启动后端（项目根或 backend/ 下均可）
cd backend
uvicorn main:app --port 8000

# 3. 启动前端（另开一个终端）
cd app
npm install        # 首次
npm run dev        # 默认 http://localhost:3000，/api 与 /ws 已代理到 8000
```

打开浏览器访问 `http://localhost:3000`：
「环境搭建」画迷宫 → 左侧选算法与奖励塑形 → 「开始训练」→ 自动跳到「训练实时视图」
看动画与曲线 → 完成后叠加策略箭头 → 到「实验对比」勾选历史 run 叠加曲线或运行课程学习对照。

## 一键复现（无需启动服务）

```bash
cd backend
python reproduce.py            # 全量：3 预设环境 × 3 算法 × 3 seeds（约 15 分钟，PPO/DQN 占大头）
python reproduce.py --quick    # 快速自检：单 seed（约 5 分钟）
python reproduce.py --algos qlearning,dqn   # 只跑部分算法
```

输出：终端对比表 + `backend/results/compare_table.csv` + `backend/results/learning_curves.png`。

**预期结果**（统一评估口径：ε=0.05 + 平局偏向终点 + 环路检测；Q-Learning 列引自
`report_experiments.py` 正式数据 20 种子 × 600 回合，PPO/DQN 列引自
`report_experiments_sb3.py` 3 种子，预算见表注）：

| 环境 | Q-Learning（600 回合，20 种子） | PPO（20k 步，3 种子） | DQN（30k 步+塑形，3 种子） |
|---|---|---|---|
| 空场 6×6 | 100%±0 | 100%±0 | — |
| 单障碍 6×6 | 100%±0 | 无塑形 67%±47（1/3 种子未学成）；开塑形 100%±0 | 100%±0（开塑形） |
| 多障碍 8×8 | 100%±0 | ~0%（15k 预算不足，历史运行观察） | ~100%（历史运行观察） |

说明：
- DQN/PPO 的关键超参已针对小预算网格迷宫调整（见 `sb3_agents.py` 注释）：DQN 的
  `target_update_interval` 若用 SB3 默认的 10000，在 6k~15k 预算下目标网络从不更新，
  成功率为 0——这是函数逼近在小预算下的典型失效模式；现取 250。
- PPO 在奖励更稀疏的迷宫上对预算与种子都更敏感：单障碍 6×6 无塑形时 20k 与 40k 预算
  均有 1/3 种子学不通（`backend/results/report/sb3_results.json`），开启势函数塑形后
  全部收敛——这正是"奖励塑形缓解稀疏奖励样本效率"的教学案例。
- 表格中的成功率是**统一评估口径**结果；该口径与旧版"纯贪心 argmax"的差异
  （以及为什么旧口径会低估有限预算下的学习成果）见 `gridworld_edu/evaluation.py`。

## 目录结构

```
gewu-rl/
├── app/                      # 前端（React + Vite + Tailwind + shadcn/ui）
│   └── src/
│       ├── pages/Home.tsx           # 单页主界面
│       ├── sections/MazeEditor.tsx  # 迷宫编辑器（笔刷/预设/随机）
│       ├── sections/RewardDesigner.tsx # AI 奖励设计师（意图→配置→对照实验）
│       ├── sections/ControlPanel.tsx# 训练控制台（算法/超参/奖励塑形）
│       ├── sections/LiveView.tsx    # 实时视图（动画+曲线+策略叠加）
│       ├── sections/CompareView.tsx # 实验对比 + 课程学习对照
│       ├── hooks/useTrainingSocket.ts
│       ├── components/MazeCanvas.tsx
│       └── types/index.ts
├── backend/
│   ├── main.py               # FastAPI：REST + WebSocket + 训练线程 + 意图实验闭环
│   ├── llm_reward.py         # AI 奖励设计师：LLM 调用/鲁棒解析/规则回退引擎
│   ├── reproduce.py          # 一键复现脚本
│   ├── verify_shaping_gamma.py # PBRS 塑形 γ 一致性验证（价值迭代）
│   ├── report_experiments.py # 报告用正式实验（默认 20 种子，可 --seeds/--quick）
│   ├── runs/                 # 每次训练自动落盘（config/metrics/policy）
│   └── results/              # reproduce.py 输出
├── tests/                    # pytest 测试（环境/算法/API），根目录 `python -m pytest` 可跑
├── gridworld_edu/            # 教学核心包（可 pip install -e，MIT）
│   └── gridworld_edu/
│       ├── envs/grid_maze.py # GridMazeEnv + RewardConfig（PBRS）
│       ├── envs/presets.py   # 4 个预设迷宫 + curriculum()
│       └── agents/
│           ├── qlearning.py  # 从零实现的表格 Q-Learning（中文注释）
│           └── sb3_agents.py # SB3 PPO/DQN 封装与策略导出
├── requirements.txt
├── LICENSE                   # MIT
└── README.md
```

## 教学要点

- **表格法 vs 函数逼近**：Q-Learning 用 `state_index` 查 Q 表；PPO/DQN 用 4 维连续观测
  + 小型 MLP（64×64），同屏对比两者曲线是最直观的一课。
- **奖励塑形（PBRS）**：势函数塑形严格采用 F = γ·Φ(s') − Φ(s)，
  Φ(s) = −曼哈顿距离/最大距离。理论上最优策略不变（Ng et al., 1999），
  只改变学习速度——开关它、对比曲线即可验证。
- **课程学习**：`/api/train/curriculum` 采用逆向课程——同一个大迷宫，
  起点由近及远（曼哈顿距离 1 → 5 → 10 → 18），状态空间一致、Q 表可直接迁移，
  与直接从最远起点训练同等总回合数对照。
- **可复现性**：每次训练把 config.json / metrics.csv / policy.json 落盘到
  `backend/runs/`，实验对比页直接读取历史 run。

## 开放成果说明

- `gridworld_edu/` 为独立教学包，以 MIT 协议开放，可 `pip install -e gridworld_edu` 单独使用。
- 平台全部代码以 MIT 协议开放（见 `LICENSE`）。

## 致谢

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) —— 环境标准接口
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) —— PPO/DQN 参考实现
- [shadcn/ui](https://ui.shadcn.com/) + [Recharts](https://recharts.org/) —— 前端组件与图表
- Ng, A. Y., Harada, D., & Russell, S. J. (1999). *Policy invariance under reward
  transformations: Theory and application to reward shaping.* ICML.
