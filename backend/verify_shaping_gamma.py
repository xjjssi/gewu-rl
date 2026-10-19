"""verify_shaping_gamma.py —— 验证 PBRS 塑形 γ 与智能体 γ 一致时的策略不变性

理论（Ng, Harada & Russell, 1999 定理 1）：当塑形项满足 F(s,s') = γ·Φ(s') − Φ(s)
且 γ 与 MDP 的折现因子一致时，塑形前后 MDP 的最优策略集合完全相同。
若塑形 γ 写死为 0.99 而智能体用 γ=0.8，该保证不成立。

本脚本在小迷宫上用价值迭代分别求"无塑形"与"有塑形（γ=0.8）"两个 MDP 的
最优策略，逐状态比较最优动作集合是否一致。

用法：
    python verify_shaping_gamma.py            # 默认 γ=0.8，单障碍 6×6
    python verify_shaping_gamma.py --gamma 0.9 --env multi_wall_8x8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gridworld_edu"))

from gridworld_edu.envs.grid_maze import CellType, GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS

TOL = 1e-8  # 价值迭代收敛阈值 / 最优动作集合判定的数值容差


def _all_states(env: GridMazeEnv) -> list[tuple[tuple[int, int], int]]:
    """枚举全部可达状态：(格子坐标, 奖励格收集掩码)。掩码任意子集都可能出现。"""
    cells = [
        (int(r), int(c))
        for r in range(env.height)
        for c in range(env.width)
        if env.grid[r, c] != CellType.WALL
    ]
    return [(pos, mask) for pos in cells for mask in range(env._bonus_states)]


def _set_state(env: GridMazeEnv, pos: tuple[int, int], mask: int) -> None:
    """把环境内部状态强制设为 (pos, mask)，用于转移采样（仅离线规划用）。"""
    env._pos = pos
    env._bonus_mask = mask
    env._steps = 0  # 避免触发步数截断，保证转移只由 MDP 规则决定


def value_iteration(
    grid: list[list[int]],
    reward_config: RewardConfig,
    gamma: float,
) -> dict[tuple, set[int]]:
    """对给定奖励配置做价值迭代，返回每个状态的最优动作集合。"""
    env = GridMazeEnv(grid, reward_config)
    env.set_shaping_gamma(gamma)  # 塑形 γ 与价值迭代所用 γ 一致
    states = _all_states(env)
    idx = {s: i for i, s in enumerate(states)}
    v = np.zeros(len(states))

    # 预计算转移：(state, action) -> (reward, next_state_index | None)
    transitions: dict[tuple[int, int], tuple[float, int | None]] = {}
    for si, (pos, mask) in enumerate(states):
        if env.grid[pos] == CellType.GOAL:
            continue  # 终点是吸收态
        for a in range(env.action_space.n):
            _set_state(env, pos, mask)
            _, r, terminated, truncated, info = env.step(a)
            assert not truncated, "离线转移采样不应触发截断"
            if terminated:
                transitions[(si, a)] = (r, None)  # 终止转移：无后继状态（Φ(terminal)=0 已在 r 中体现）
            else:
                npos = (info["pos"][0], info["pos"][1])
                nmask = info["state_index"] % env._bonus_states
                transitions[(si, a)] = (r, idx[(npos, nmask)])

    # 价值迭代至收敛
    while True:
        delta = 0.0
        for si, (pos, _) in enumerate(states):
            if env.grid[pos] == CellType.GOAL:
                continue
            qs = [r if ns is None else r + gamma * v[ns]
                  for (r, ns) in (transitions[(si, a)] for a in range(env.action_space.n))]
            nv = max(qs)
            delta = max(delta, abs(nv - v[si]))
            v[si] = nv
        if delta < TOL:
            break

    # 提取最优动作集合（并列最优都算，避免数值平局造成伪差异）
    policy: dict[tuple, set[int]] = {}
    for si, (pos, mask) in enumerate(states):
        if env.grid[pos] == CellType.GOAL:
            continue
        qs = np.array([r if ns is None else r + gamma * v[ns]
                       for (r, ns) in (transitions[(si, a)] for a in range(env.action_space.n))])
        policy[(pos, mask)] = set(np.flatnonzero(qs >= qs.max() - 1e-6).tolist())
    return policy


def check_policy_invariance(grid: list[list[int]], gamma: float) -> tuple[bool, int, int]:
    """比较无塑形 / 有塑形两个 MDP 的最优策略，返回 (是否一致, 不一致状态数, 总状态数)。"""
    pi_plain = value_iteration(grid, RewardConfig(potential_shaping=False), gamma)
    pi_shaped = value_iteration(grid, RewardConfig(potential_shaping=True), gamma)
    mismatched = [s for s in pi_plain if pi_plain[s] != pi_shaped.get(s, set())]
    return len(mismatched) == 0, len(mismatched), len(pi_plain)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.8, help="折现因子（默认 0.8，刻意远离 0.99）")
    parser.add_argument("--env", default="single_wall_6x6", choices=list(PRESETS), help="预设迷宫")
    args = parser.parse_args()

    grid = PRESETS[args.env]["grid"]
    ok, n_bad, n_all = check_policy_invariance(grid, args.gamma)
    print(f"环境: {args.env}（{PRESETS[args.env]['name']}），γ = {args.gamma}")
    print(f"共比较 {n_all} 个状态，最优动作集合不一致的状态数: {n_bad}")
    if ok:
        print("结论：一致 ✅ —— 塑形 γ 取智能体实际 γ 时，有无 PBRS 的最优策略完全相同。")
        sys.exit(0)
    print("结论：不一致 ✗ —— 塑形改变了最优策略，违背 PBRS 理论，请检查实现。")
    sys.exit(1)


if __name__ == "__main__":
    main()
