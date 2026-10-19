"""test_env.py —— GridMazeEnv 核心正确性：预设合法性、连通性、奖励发放、状态编码、PBRS 公式。"""

from collections import deque

import pytest

from gridworld_edu.envs.grid_maze import CellType, GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS, curriculum

from verify_shaping_gamma import check_policy_invariance

# 一个带奖励格的最小自定义迷宫（6×6 是环境允许的最小尺寸）
CUSTOM_BONUS_MAZE = [
    [2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3],
]


def _bfs_reachable(grid: list[list[int]]) -> set[tuple[int, int]]:
    """从起点 BFS，返回所有可通行格子的集合。"""
    h, w = len(grid), len(grid[0])
    start = next((r, c) for r in range(h) for c in range(w) if grid[r][c] == CellType.START)
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != CellType.WALL and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def test_presets_and_curriculum_valid_and_connected():
    """四个预设 + 四个课程关卡：构造合法、起点 BFS 可达终点及全部奖励/陷阱格。"""
    grids = [p["grid"] for p in PRESETS.values()] + [s["grid"] for s in curriculum()]
    assert len(PRESETS) == 4 and len(curriculum()) == 4
    for grid in grids:
        GridMazeEnv(grid)  # 构造即校验：尺寸、唯一起终点
        reach = _bfs_reachable(grid)
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if v in (CellType.GOAL, CellType.BONUS, CellType.TRAP):
                    assert (r, c) in reach, f"格子 ({r},{c}) 类型 {v} 不可达"


def test_bonus_awarded_once():
    """奖励格只发一次：第一次踩上有 bonus_reward，离开后返回不再发。"""
    cfg = RewardConfig()
    env = GridMazeEnv(CUSTOM_BONUS_MAZE, cfg)
    env.reset()
    for a in [1, 1, 3, 3]:  # 下、下、右、右 → 踩到 (2,2) 奖励格
        _, r, _, _, _ = env.step(a)
    assert r == pytest.approx(cfg.step_penalty + cfg.bonus_reward)
    assert env.episode_stats["bonus_collected"] == 1
    env.step(2)  # 左：离开奖励格
    env.step(3)  # 右：回到奖励格，不再发奖
    assert env.episode_stats["bonus_collected"] == 1


def test_bonus_awarded_once_reward_value():
    """第二次踩同一奖励格：奖励回落为纯步数代价。"""
    cfg = RewardConfig()
    env = GridMazeEnv(CUSTOM_BONUS_MAZE, cfg)
    env.reset()
    for a in [1, 1, 3, 3]:
        env.step(a)
    env.step(2)  # 离开
    _, r2, _, _, _ = env.step(3)  # 返回
    assert r2 == pytest.approx(cfg.step_penalty)


def test_mask_state_encoding():
    """收集掩码是状态的一部分：收集前后 state_index 相差 1，n_states 翻倍。"""
    env = GridMazeEnv(CUSTOM_BONUS_MAZE)
    assert env.n_states == 6 * 6 * 2  # 1 个奖励格 → 掩码 2 种取值
    _, info = env.reset()
    s0 = info["state_index"]
    assert s0 % 2 == 0  # 掩码为 0
    for a in [1, 1, 3, 3]:
        _, _, _, _, info = env.step(a)
    assert info["state_index"] % 2 == 1  # 掩码位被置 1
    # reset 后掩码清零
    _, info = env.reset()
    assert info["state_index"] % 2 == 0


def test_wall_bump_stays_put():
    """撞墙/越界：原地不动，奖励 = 步数代价 + 撞墙惩罚。"""
    cfg = RewardConfig()
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], cfg)
    _, info = env.reset()
    _, r, terminated, truncated, info = env.step(0)  # 起点 (0,0) 向上 = 越界
    assert info["pos"] == [0, 0]
    assert r == pytest.approx(cfg.step_penalty + cfg.wall_penalty)
    assert not terminated and not truncated


def test_pbrs_exact_formula_nonterminal():
    """非终止转移的塑形项严格等于 γ·Φ(s') − Φ(s)。"""
    gamma = 0.8
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig(potential_shaping=True))
    env.set_shaping_gamma(gamma)
    env.reset()
    old = env.start_pos
    _, r, terminated, _, info2 = env.step(3)  # 右：(0,0) → (0,1)
    assert not terminated
    expected = -0.01 + gamma * env.potential(tuple(info2["pos"])) - env.potential(old)
    assert r == pytest.approx(expected)


def test_pbrs_terminal_transition_phi_zero():
    """终止转移同样计入塑形且 Φ(terminal)=0：走进终点的奖励含 −Φ(s) 项。"""
    gamma = 0.8
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig(potential_shaping=True))
    env.set_shaping_gamma(gamma)
    env.reset()
    env._pos = (5, 4)  # 终点 (5,5) 左侧一格（测试专用：直接设置内部状态）
    env._steps = 0
    _, r, terminated, _, info = env.step(3)  # 右 → 终点
    assert terminated and info["is_success"]
    # Φ(terminal)=0：F = γ·0 − Φ((5,4)) = −(−1/10) = 0.1
    expected = -0.01 + 1.0 + (gamma * 0.0 - (-1 / 10))
    assert r == pytest.approx(expected)


def test_shaping_gamma_must_match_agent_gamma():
    """默认塑形 γ=0.99；set_shaping_gamma 后按新 γ 计算。"""
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig(potential_shaping=True))
    env.reset()
    _, r_default, _, _, _ = env.step(3)
    env.reset()
    env.set_shaping_gamma(0.5)
    _, r_custom, _, _, _ = env.step(3)
    # F = γ·Φ(s') − Φ(s) = γ·(−0.9) − (−1) = 1 − 0.9γ，随 γ 变化
    assert r_default == pytest.approx(-0.01 + (1 - 0.9 * 0.99))
    assert r_custom == pytest.approx(-0.01 + (1 - 0.9 * 0.5))


def test_shaping_policy_invariance_small_gamma():
    """γ=0.8（刻意远离默认 0.99）时，有无 PBRS 的最优策略完全一致（价值迭代验证）。"""
    ok, n_bad, n_all = check_policy_invariance(PRESETS["single_wall_6x6"]["grid"], gamma=0.8)
    assert ok, f"{n_bad}/{n_all} 个状态的最优动作集合不一致"
