"""grid_maze.py —— 网格迷宫环境（Gymnasium 兼容）

这是「格物RL」平台的核心教学环境。设计目标：

1. 规则简单透明，方便课堂讲解 MDP 五元组 (S, A, P, R, γ) 的每一部分；
2. 观测同时支持两种形式：
   - 低维连续向量（默认 observation_space，供 PPO/DQN 等函数逼近算法使用）；
   - 离散格子序号 ``state_index``（供表格 Q-Learning 使用）；
3. 奖励完全由 ``RewardConfig`` 配置，方便做"奖励塑形"对照实验；
4. 势函数奖励塑形（Potential-Based Reward Shaping, PBRS）严格满足
   Ng, Harada & Russell (1999) 的 γ-一致形式：

       F(s, s') = γ · Φ(s') − Φ(s),   Φ(s) = −dist(s, goal) / dist_max

   理论保证：对任意满足该形式的 F，新 MDP 与原 MDP 的最优策略集合完全相同
   （只是 Q 函数整体平移了一个与状态有关的势）。因此打开塑形只会改变
   "学习曲线形状"，不会改变"最终学到什么"——这是课堂上非常重要的一课。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CellType(IntEnum):
    """格子类型编码，与前端/序列化格式保持一致。"""

    EMPTY = 0   # 空地
    WALL = 1    # 墙（不可通行）
    START = 2   # 起点
    GOAL = 3    # 终点
    BONUS = 4   # 奖励格（经过时给一次性正奖励）
    TRAP = 5    # 陷阱格（给负奖励，可选是否终止回合）


# 四个动作：上、下、左、右。对应的 (Δrow, Δcol) 偏移。
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["上", "下", "左", "右"]


@dataclass
class RewardConfig:
    """奖励塑形配置。所有项都可以在前端"奖励塑形面板"中调整。"""

    step_penalty: float = -0.01    # 每走一步的代价，鼓励尽快到达终点
    wall_penalty: float = -0.05    # 撞墙（原地不动）的代价
    goal_reward: float = 1.0       # 到达终点的奖励
    bonus_reward: float = 0.5      # 踩到奖励格的一次性奖励
    trap_penalty: float = -1.0     # 踩到陷阱格的惩罚
    trap_terminal: bool = False    # 踩到陷阱是否立即结束回合
    potential_shaping: bool = False  # 是否启用势函数奖励塑形（PBRS）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RewardConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class GridMazeEnv(gym.Env):
    """网格迷宫环境。

    参数
    ----
    grid : list[list[int]]
        二维格子矩阵，取值见 ``CellType``。行数列数均在 6~12 之间。
        必须恰好包含一个 START 和一个 GOAL。
    reward_config : RewardConfig
        奖励塑形配置。
    max_steps : int | None
        单回合最大步数（截断，不算终止）。默认 4 × 格子数。
    """

    metadata = {"render_modes": ["dict"]}

    MIN_SIZE, MAX_SIZE = 6, 12

    def __init__(
        self,
        grid: list[list[int]],
        reward_config: RewardConfig | None = None,
        max_steps: int | None = None,
    ):
        super().__init__()
        g = np.asarray(grid, dtype=np.int64)
        if g.ndim != 2:
            raise ValueError("grid 必须是二维矩阵")
        h, w = g.shape
        if not (self.MIN_SIZE <= h <= self.MAX_SIZE and self.MIN_SIZE <= w <= self.MAX_SIZE):
            raise ValueError(f"网格尺寸须在 {self.MIN_SIZE}~{self.MAX_SIZE} 之间，收到 {h}x{w}")
        starts = np.argwhere(g == CellType.START)
        goals = np.argwhere(g == CellType.GOAL)
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("迷宫必须恰好包含一个起点和一个终点")

        self.grid = g
        self.height, self.width = h, w
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps or 4 * h * w

        self.start_pos = (int(starts[0][0]), int(starts[0][1]))
        self.goal_pos = (int(goals[0][0]), int(goals[0][1]))

        # 奖励格列表与"已收集"掩码。
        # 教学要点：奖励格只能领一次，因此"哪些奖励格已收集"是 MDP 状态的一部分。
        # 若状态只包含坐标，Q(奖励格旁) 会把已领走的奖励当成仍可领的"幻影价值"，
        # 贪心策略可能在奖励格附近来回打转（2-环）。把收集掩码编码进状态即可修复。
        self.bonus_cells: list[tuple[int, int]] = [
            (int(r), int(c)) for r, c in np.argwhere(g == CellType.BONUS)
        ]
        self._bonus_index = {cell: i for i, cell in enumerate(self.bonus_cells)}
        self._n_bonus_bits = len(self.bonus_cells)
        self._bonus_states = 1 << self._n_bonus_bits  # 2^n（无奖励格时为 1，退化为纯坐标状态）

        # ---- MDP 的动作空间与观测空间 ----
        self.action_space = spaces.Discrete(4)
        # 观测 = [行归一化, 列归一化, 到终点的相对行向量, 到终点的相对列向量, 各奖励格是否已收集(0/1)...]
        # 低维连续向量，便于 MlpPolicy 的 PPO/DQN 直接消费。
        self._max_dist = (h - 1) + (w - 1)  # 曼哈顿距离上界，用于归一化
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4 + self._n_bonus_bits,), dtype=np.float32
        )

        self._pos = self.start_pos
        self._steps = 0
        self._bonus_mask = 0  # 已收集奖励格的位掩码
        # 行为统计钩子（供"AI 奖励设计师"做意图达成度验证）：
        # 统计本回合踩过的奖励格/陷阱格次数，reset 时清零。
        self.episode_stats = {"bonus_collected": 0, "trap_triggered": 0}

        # PBRS 塑形用的折现因子。理论要求它与智能体学习目标所用的 γ 一致：
        # F = γ·Φ(s') − Φ(s) 的"最优策略不变"证明（Ng et al., 1999 定理 1）
        # 依赖于塑形 γ 等于 MDP 的折现因子；若智能体用 γ=0.8 而塑形按 0.99 计算，
        # 塑形项就不再是严格的势函数差，策略不变性失去理论保证。
        # 默认 0.99（与多数算法默认 γ 相同），训练入口应按实际 γ 调用 set_shaping_gamma()。
        self._shaping_gamma = 0.99

    def set_shaping_gamma(self, gamma: float) -> None:
        """把 PBRS 塑形的折现因子设为智能体实际使用的 γ。"""
        self._shaping_gamma = float(gamma)

    # ------------------------------------------------------------------
    # 供表格法使用的离散状态
    # ------------------------------------------------------------------
    @property
    def state_index(self) -> int:
        """离散状态序号 = 格子序号 × 2^奖励格数 + 收集掩码，供表格 Q-Learning 作 Q 表行下标。"""
        cell = self._pos[0] * self.width + self._pos[1]
        return cell * self._bonus_states + self._bonus_mask

    @property
    def n_states(self) -> int:
        return self.height * self.width * self._bonus_states

    # ------------------------------------------------------------------
    # 势函数：Φ(s) = −曼哈顿距离 / 最大距离
    # ------------------------------------------------------------------
    def potential(self, pos: tuple[int, int]) -> float:
        dist = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
        return -dist / self._max_dist

    def _get_obs(self) -> np.ndarray:
        r, c = self._pos
        gr, gc = self.goal_pos
        base = [
            r / (self.height - 1),
            c / (self.width - 1),
            (gr - r) / self._max_dist,
            (gc - c) / self._max_dist,
        ]
        # 每个奖励格追加一位 0/1 标志：是否已被收集
        flags = [
            1.0 if (self._bonus_mask >> i) & 1 else 0.0
            for i in range(self._n_bonus_bits)
        ]
        return np.array(base + flags, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium 标准接口
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._pos = self.start_pos
        self._steps = 0
        self._bonus_mask = 0
        self.episode_stats = {"bonus_collected": 0, "trap_triggered": 0}
        return self._get_obs(), {"state_index": self.state_index}

    def step(self, action: int):
        assert self.action_space.contains(action), f"非法动作 {action}"
        cfg = self.reward_config
        self._steps += 1

        dr, dc = ACTIONS[action]
        old_pos = self._pos
        nr, nc = old_pos[0] + dr, old_pos[1] + dc

        reward = cfg.step_penalty
        terminated = False
        truncated = False

        # 越界或撞墙：原地不动，给撞墙惩罚
        if not (0 <= nr < self.height and 0 <= nc < self.width) or self.grid[nr, nc] == CellType.WALL:
            reward += cfg.wall_penalty
        else:
            self._pos = (nr, nc)
            cell = self.grid[nr, nc]
            if cell == CellType.GOAL:
                reward += cfg.goal_reward
                terminated = True
            elif cell == CellType.BONUS:
                bit = 1 << self._bonus_index[(nr, nc)]
                if not (self._bonus_mask & bit):  # 每个奖励格每回合只发一次奖励
                    reward += cfg.bonus_reward
                    self._bonus_mask |= bit
                    self.episode_stats["bonus_collected"] += 1
            elif cell == CellType.TRAP:
                reward += cfg.trap_penalty
                self.episode_stats["trap_triggered"] += 1
                if cfg.trap_terminal:
                    terminated = True

        # ---- 势函数奖励塑形（PBRS）----
        # F = γ·Φ(s') − Φ(s)，γ 取智能体学习目标实际使用的折现因子
        # （由训练入口通过 set_shaping_gamma 传入，默认 0.99）。
        # 塑形 γ 与 MDP 的 γ 一致时，"最优策略不变"才有理论保证
        # （证明见 Ng et al. 1999 定理 1）。
        # 注意：终止状态的势必须定义为 0（Φ(terminal)=0），终止转移同样要计入塑形。
        # 终点处 Φ 本来就等于 0，但陷阱终结（trap_terminal=True）时若跳过塑形，
        # 回合回报的塑形总和会依赖终止位置，破坏"最优策略不变"的理论保证。
        if cfg.potential_shaping:
            phi_next = 0.0 if terminated else self.potential(self._pos)
            reward += self._shaping_gamma * phi_next - self.potential(old_pos)

        if not terminated and self._steps >= self.max_steps:
            truncated = True  # 步数上限：截断而非终止，bootstrapping 时两者须区分

        info = {
            "state_index": self.state_index,
            "pos": list(self._pos),
            "is_success": terminated and self._pos == self.goal_pos,
        }
        if terminated or truncated:
            # 回合结束时附带上行为统计，供意图达成度验证读取
            info["episode_stats"] = dict(self.episode_stats)
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> dict[str, Any]:
        """返回可 JSON 序列化的网格字典，供前端绘制。"""
        return {
            "grid": self.grid.tolist(),
            "agent_pos": list(self._pos),
            "height": self.height,
            "width": self.width,
        }
