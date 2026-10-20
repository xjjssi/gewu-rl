"""evaluation.py —— 统一策略评估口径（后端服务 / 复现脚本 / 报告实验共用）

背景：纯贪心评估（argmax + 随机平局）在有限训练预算下会系统性低估学习成果——
当若干动作的 Q 值几乎相等时（例如大迷宫起点附近），贪心策略会在平局动作间
绕圈直到步数上限，出现"训练后期成功率 100%、贪心评估 0%"的自相矛盾。

统一口径（backend/main.py、reproduce.py、report_experiments.py 三处一致）：

1. **ε-评估**：评估时保留很小的探索率（默认 ε=0.05），模拟"基本学会、偶尔探索"，
   避免单一平局把整局锁死在环路里；
2. **平局偏向终点**：Q 值在容差内并列时，优先选择执行后到终点曼哈顿距离更小的
   动作（方向与 PBRS 势函数一致；仅作平局裁决，不修改 Q 值本身）；
3. **环路检测**：单回合内某状态重复到访超过阈值（默认第 3 次到访起）记为
   "绕圈回合"，单独报告 loop_rate。

由此可区分两种截然不同的情形：
- "完全没学会"：ε-评估成功率依然接近随机基线；
- "学会了但贪心未稳定"：ε-评估成功率高，但 loop_rate 高 / 纯贪心成功率低
  （Q 值已学到正确方向，只是平局裁决不稳）。

这本身也是一堂教学课：评估协议是实验结论的一部分。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from gridworld_edu.envs.grid_maze import ACTIONS, CellType, GridMazeEnv

# 默认评估超参（三处调用方统一使用，不各自为政）
DEFAULT_EVAL_EPSILON = 0.05   # 评估时保留的微小探索率
DEFAULT_TIE_TOL = 0.05        # Q 值平局相对容差：q >= qmax - tol*|qmax| 视为并列
DEFAULT_LOOP_THRESHOLD = 2    # 单回合内某状态到访次数超过该值 → 记为绕圈回合


def qlearning_eval_action(
    env: GridMazeEnv,
    q_table: np.ndarray,
    state: int,
    rng: np.random.Generator,
    epsilon: float = DEFAULT_EVAL_EPSILON,
    tie_tol: float = DEFAULT_TIE_TOL,
) -> int:
    """Q-Learning 评估动作：ε 随机探索 + 容差平局 + 平局偏向缩短终点距离。

    tie_tol=0 时退化为"精确 argmax + 随机平局"（旧口径）；epsilon=0 且
    tie_tol=0 即完全等价于旧的纯贪心评估。
    """
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    q = q_table[state]
    qmax = q.max()
    if tie_tol <= 0.0:
        # 旧口径：精确 argmax + 随机平局打破（用于"纯贪心"对照组）
        return int(rng.choice(np.flatnonzero(q == qmax)))
    tol = max(1e-9, tie_tol * abs(qmax))
    cands = np.flatnonzero(q >= qmax - tol)
    if len(cands) == 1:
        return int(cands[0])
    # 平局裁决：优先选择执行后到终点曼哈顿距离更小的动作；仍并列则随机。
    # 由 state_index 反推格子坐标（state = 格子序号 × 2^奖励格数 + 收集掩码），
    # 与 reward_config 无关，因此同一智能体在不同奖励配置下评估也成立。
    cell = state // env._bonus_states
    r, c = divmod(cell, env.width)
    best: list[int] = []
    best_d: int | None = None
    for a in cands:
        dr, dc = ACTIONS[int(a)]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < env.height and 0 <= nc < env.width) or env.grid[nr, nc] == CellType.WALL:
            nr, nc = r, c  # 撞墙原地不动
        d = abs(nr - env.goal_pos[0]) + abs(nc - env.goal_pos[1])
        if best_d is None or d < best_d:
            best_d, best = d, [int(a)]
        elif d == best_d:
            best.append(int(a))
    return int(rng.choice(best))


def evaluate_policy(
    env: GridMazeEnv,
    act_fn: Callable[[np.ndarray, dict], int],
    n: int = 20,
    loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
) -> dict[str, Any]:
    """用给定策略跑 n 回合，返回统一口径的行为指标。

    act_fn(obs, info) -> action，由各算法入口构造（内部可含 ε 探索与平局裁决）。
    环路检测：单回合内某状态到访次数 > loop_threshold 即记为绕圈回合
    （正常最优路径上每个状态至多到访一次；撞墙原地不动也会累计到访次数，
    恰好能捕捉"贪心卡墙角"这类不稳定行为）。
    """
    success = steps = 0
    total_reward = 0.0
    loops = stable_success = 0
    bonus_eps = trap_eps = 0
    bonus_cnt = trap_cnt = 0
    for _ in range(n):
        obs, info = env.reset()
        visits = {info["state_index"]: 1}
        looped = False
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            obs, r, terminated, truncated, info = env.step(act_fn(obs, info))
            ep_reward += r
            ep_steps += 1
            done = terminated or truncated
            si = info["state_index"]
            visits[si] = visits.get(si, 0) + 1
            if visits[si] > loop_threshold:
                looped = True
        ok = int(info["is_success"])
        success += ok
        loops += int(looped)
        stable_success += int(ok and not looped)
        steps += ep_steps
        total_reward += ep_reward
        st = info.get("episode_stats", {})
        bonus_cnt += st.get("bonus_collected", 0)
        trap_cnt += st.get("trap_triggered", 0)
        bonus_eps += int(st.get("bonus_collected", 0) > 0)
        trap_eps += int(st.get("trap_triggered", 0) > 0)
    return {
        "success_rate": round(success / n, 3),
        "avg_steps": round(steps / n, 1),
        "avg_reward": round(total_reward / n, 3),
        "loop_rate": round(loops / n, 3),                # 绕圈回合占比（贪心不稳定的信号）
        "stable_success_rate": round(stable_success / n, 3),  # 成功且未绕圈的回合占比
        "bonus_collect_rate": round(bonus_eps / n, 3),   # 至少收集 1 个奖励格的回合占比
        "trap_trigger_rate": round(trap_eps / n, 3),     # 至少踩 1 次陷阱的回合占比
        "avg_bonus_count": round(bonus_cnt / n, 2),
        "avg_trap_count": round(trap_cnt / n, 2),
        "eval_episodes": n,
    }


def evaluate_qlearning(
    env: GridMazeEnv,
    agent,
    n: int = 20,
    epsilon: float = DEFAULT_EVAL_EPSILON,
    tie_tol: float = DEFAULT_TIE_TOL,
    loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
    seed: int = 0,
) -> dict[str, Any]:
    """表格 Q-Learning 的统一评估：ε-评估 + 平局偏向终点 + 环路检测。

    同时附跑一组旧口径（纯贪心 argmax、随机平局、ε=0），返回
    pure_greedy_success_rate / pure_greedy_loop_rate——两者的差距正是
    "学会了但贪心未稳定"的量化证据，供报告与解读文案使用。
    """
    rng = np.random.default_rng(seed)
    act = lambda o, i: qlearning_eval_action(  # noqa: E731
        env, agent.q_table, i["state_index"], rng, epsilon=epsilon, tie_tol=tie_tol)
    m = evaluate_policy(env, act, n=n, loop_threshold=loop_threshold)
    m["epsilon"] = epsilon

    rng_pure = np.random.default_rng(seed + 10_000)
    act_pure = lambda o, i: qlearning_eval_action(  # noqa: E731
        env, agent.q_table, i["state_index"], rng_pure, epsilon=0.0, tie_tol=0.0)
    mp = evaluate_policy(env, act_pure, n=n, loop_threshold=loop_threshold)
    m["pure_greedy_success_rate"] = mp["success_rate"]
    m["pure_greedy_loop_rate"] = mp["loop_rate"]
    return m


def evaluate_sb3(
    env: GridMazeEnv,
    model,
    n: int = 20,
    epsilon: float = DEFAULT_EVAL_EPSILON,
    loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
    seed: int = 0,
) -> dict[str, Any]:
    """SB3（PPO/DQN）的统一评估：deterministic predict + ε 随机覆盖 + 环路检测。"""
    rng = np.random.default_rng(seed)

    def act(obs, info):
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.integers(env.action_space.n))
        return int(model.predict(obs, deterministic=True)[0])

    m = evaluate_policy(env, act, n=n, loop_threshold=loop_threshold)
    m["epsilon"] = epsilon
    return m
