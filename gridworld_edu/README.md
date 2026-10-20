# gridworld_edu

面向强化学习教学的网格迷宫包：Gymnasium 标准环境、从零实现的表格 Q-Learning、
Stable-Baselines3（PPO/DQN）封装、预设迷宫与课程序列。MIT 协议开源。

## 安装

```bash
pip install -e .
```

依赖：gymnasium、numpy；使用 PPO/DQN 封装时另需 stable-baselines3 与 torch。

## 最小示例

```python
from gridworld_edu.envs import presets
from gridworld_edu.envs.grid_maze import GridMazeEnv
from gridworld_edu.agents.qlearning import QLearningAgent

env = GridMazeEnv(presets.single_wall_6x6())   # 预设迷宫
env.set_shaping_gamma(0.99)                    # PBRS 塑形 γ 与智能体一致
agent = QLearningAgent(n_states=env.n_states, seed=0)

for _ in range(300):
    _, info = env.reset()
    s = info["state_index"]                    # 表格法用离散状态编号
    done = False
    while not done:
        a = agent.select_action(s)
        _, r, terminated, truncated, info = env.step(a)
        agent.update(s, a, r, info["state_index"], terminated, truncated)
        s = info["state_index"]
        done = terminated or truncated
    agent.decay_epsilon()

policy = agent.greedy_policy(env)              # 每个可通行格子的贪心动作与 Q 值
```

> `env.step()` 返回的第一个值是给 PPO/DQN 用的连续观测向量；
> 表格法请使用 `info["state_index"]`。

## 内容

- `envs/grid_maze.py` — `GridMazeEnv`（Gymnasium 环境）与 `RewardConfig`（PBRS 奖励塑形）
- `envs/presets.py` — 4 个预设迷宫、随机迷宫生成、逆向课程序列 `curriculum()`
- `agents/qlearning.py` — 从零实现的表格 Q-Learning（全中文注释）
- `agents/sb3_agents.py` — Stable-Baselines3 PPO/DQN 封装与策略导出

本包是「格物RL」可视化教学实验平台的核心组件，可脱离平台单独使用。
