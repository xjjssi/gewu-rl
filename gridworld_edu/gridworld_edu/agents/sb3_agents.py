"""sb3_agents.py —— Stable-Baselines3 PPO / DQN 的教学封装

教学要点：
- PPO 与 DQN 都是"函数逼近"方法：不再维护一张 Q 表，而是用小型神经网络
  （默认 2 层 × 64 单元的 MLP）拟合价值/策略函数，因此可以处理连续观测。
- 这里刻意把网络做小、步数设少，让 CPU 上几十秒内即可看到学习效果。
- 固定 seed 保证实验可复现（课程讲解"随机性的影响"时可去掉）。
"""

from __future__ import annotations

from stable_baselines3 import DQN, PPO

from gridworld_edu.envs.grid_maze import GridMazeEnv


def create_sb3_agent(
    algo: str,
    env: GridMazeEnv,
    seed: int = 0,
    learning_rate: float | None = None,
    gamma: float = 0.99,
):
    """按算法名创建 SB3 智能体。

    参数
    ----
    algo : "ppo" | "dqn"
    env : GridMazeEnv
    seed : 随机种子
    learning_rate : 学习率；None 时按算法取默认（PPO 3e-4，DQN 1e-3——DQN 的
        TD 目标来自目标网络，网格迷宫奖励稀疏，1e-3 才能在小预算内拉动 Q 值）
    gamma : 折现因子，前端 slider 可调

    网格迷宫的教学预算只有几千~几万步，SB3 的部分默认值是为 Atari 级预算设计的，
    直接使用会导致"预算内目标网络从未更新"之类的失效：
    - DQN ``target_update_interval`` 默认 10000 步——6k 预算下目标网络一次都不更新，
      Q 值被随机初始化的目标网络拖住，表现为怎么训练成功率都是 0；
    - DQN ``train_freq`` 默认 4 步一更新、``learning_starts`` 默认 50000，
      对小迷宫过于保守；
    - PPO 默认 n_steps=2048 / n_epochs=10：2048 步的 rollout 在迷宫里包含太多回合，
      策略更新过慢，这里取 256。
    """
    env.set_shaping_gamma(gamma)  # PBRS 塑形 γ 与学习目标 γ 保持一致
    net_arch = [64, 64]  # 小型 MLP：教学够用，CPU 友好
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=3e-4 if learning_rate is None else learning_rate,
            gamma=gamma,
            n_steps=256,          # 小批量 rollout，加快回合反馈频率
            batch_size=64,
            n_epochs=10,
            ent_coef=0.02,        # 熵奖励：防止策略过早收敛到次优
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
        )
    if algo == "dqn":
        return DQN(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=1e-3 if learning_rate is None else learning_rate,
            gamma=gamma,
            buffer_size=20_000,
            learning_starts=100,
            batch_size=64,
            train_freq=1,              # 每步都更新，小预算下样本利用率优先
            target_update_interval=250,  # 目标网络 250 步同步一次（默认 10000 在本预算下从不更新）
            exploration_initial_eps=1.0,   # DQN 的 ε 同样先探索后利用
            exploration_final_eps=0.05,
            exploration_fraction=0.4,
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
        )
    raise ValueError(f"未知算法: {algo}")


def export_greedy_policy(model, env: GridMazeEnv) -> dict:
    """导出每个可通行格子的贪心动作与"类 Q 值"（动作概率），供前端画箭头热力图。

    对 PPO/DQN 没有显式 Q 表，这里用策略网络的动作概率分布作为可视化替代：
    概率越高表示网络越确信该动作，效果上等价于 Q 值的相对大小。
    """
    import numpy as np
    import torch

    from gridworld_edu.envs.grid_maze import CellType

    cells = []
    obs = env.reset()[0]
    for r in range(env.height):
        for c in range(env.width):
            if env.grid[r, c] == CellType.WALL:
                continue
            # 直接构造该格子的观测向量（与环境 _get_obs 逻辑一致，掩码=0 即未收集）
            gr, gc = env.goal_pos
            max_dist = (env.height - 1) + (env.width - 1)
            base = [r / (env.height - 1), c / (env.width - 1),
                    (gr - r) / max_dist, (gc - c) / max_dist]
            obs = np.array(base + [0.0] * len(env.bonus_cells), dtype=np.float32)
            with torch.no_grad():
                t = torch.as_tensor(obs).unsqueeze(0).to(model.device)
                if hasattr(model.policy, "q_net"):
                    # DQN：直接读取 Q 网络输出的真实 Q 值
                    vals = model.policy.q_net(t).detach().cpu().numpy().flatten()
                else:
                    # PPO：用策略网络的动作概率作为"类 Q 值"可视化
                    dist = model.policy.get_distribution(t)
                    vals = dist.distribution.probs.detach().cpu().numpy().flatten()
            cells.append(
                {
                    "pos": [r, c],
                    "action": int(np.argmax(vals)),
                    "q_values": [round(float(p), 4) for p in vals],
                    "q_max": round(float(vals.max()), 4),
                }
            )
    return {"cells": cells}
