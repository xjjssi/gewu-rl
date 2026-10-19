"""qlearning.py —— 从零实现的表格 Q-Learning（教学版）

不依赖任何 RL 库，只用 NumPy。每一段公式都对应课堂板书：

    Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
                                └────────── TD 目标 ──────────┘
                                └──────────── TD 误差 ────────────┘

其中：
- α (learning_rate)：学习率，控制新信息覆盖旧信息的速度；
- γ (gamma)：折现因子，权衡眼前奖励与长期回报；
- ε (epsilon)：探索率，ε-greedy 策略以 ε 的概率随机探索，否则选当前最优动作；
- ε 随训练线性衰减（epsilon_start → epsilon_end），实现"先探索、后利用"。

终止 (terminated) 与截断 (truncated) 的区别：
- 终止（到达终点 / 陷阱终结）后没有后继状态，TD 目标只有 r；
- 截断（步数上限）只是人为中断，理论上仍可 bootstrap，因此保留 max Q(s',·) 项。
"""

from __future__ import annotations

import numpy as np

from gridworld_edu.envs.grid_maze import CellType, GridMazeEnv


class QLearningAgent:
    """表格 Q-Learning 智能体。"""

    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 400,
        seed: int = 0,
    ):
        self.lr = learning_rate
        self.lr_start = learning_rate
        # 学习率线性衰减到初始值的 1/10：
        # 训练后期 Q 值需要"沉淀"——若 α 恒定，TD 噪声会让相近动作的 Q 排名抖动，
        # 贪心策略可能在原地打转（课堂上可演示"恒定步长 vs 衰减步长"的差别）。
        self.lr_end = learning_rate * 0.1
        self._lr_step = (self.lr_start - self.lr_end) / max(1, epsilon_decay_episodes)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        # ε 线性衰减的每回合步长
        self._eps_step = (epsilon_start - epsilon_end) / max(1, epsilon_decay_episodes)
        self.rng = np.random.default_rng(seed)
        # Q 表：行是状态（格子序号），列是动作（上下左右）
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.n_actions = n_actions

    # ------------------------------------------------------------------
    def select_action(self, state: int, greedy: bool = False) -> int:
        """ε-greedy 选动作。greedy=True 时纯利用（用于评估）。"""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))  # 探索：随机动作
        # 利用：选 Q 值最大的动作；并列时随机打破平局，避免偏向固定方向
        q = self.q_table[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Q-Learning 单步更新，返回 TD 误差绝对值（可用于课堂可视化）。"""
        # 终止状态没有后继：TD 目标 = r；否则 = r + γ·max Q(s',·)
        if terminated:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state].max()
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return abs(td_error)

    def decay_epsilon(self) -> None:
        """每回合结束后调用一次，线性衰减探索率与学习率。"""
        self.epsilon = max(self.epsilon_end, self.epsilon - self._eps_step)
        self.lr = max(self.lr_end, self.lr - self._lr_step)

    # ------------------------------------------------------------------
    def greedy_policy(self, env: GridMazeEnv) -> dict:
        """导出每个可通行格子的贪心策略与 Q 值，供前端画箭头和热力图。

        状态含奖励格收集掩码；可视化统一展示"尚未收集任何奖励格"时的策略。
        """
        cells = []
        for r in range(env.height):
            for c in range(env.width):
                if env.grid[r, c] == CellType.WALL:
                    continue
                s = (r * env.width + c) * env._bonus_states  # 掩码=0 的状态
                q = self.q_table[s]
                cells.append(
                    {
                        "pos": [r, c],
                        "action": int(q.argmax()),
                        "q_values": [round(float(v), 4) for v in q],
                        "q_max": round(float(q.max()), 4),
                    }
                )
        return {"cells": cells}
