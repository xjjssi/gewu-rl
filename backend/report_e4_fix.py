"""report_e4_fix.py —— 课程学习迁移修正实验：跨关卡迁移时对"新墙动作"做掩码重置。

发现的问题：上一级学到的 Q 值会让贪心策略试图穿过本级新增的墙，
导致"训练显示收敛、贪心评估 0%"——这是课程学习迁移的经典陷阱，
修正方法：进入新关卡时，把所有指向墙格的动作 Q 值重置为负值。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gridworld_edu"))

from gridworld_edu.envs.grid_maze import ACTIONS, CellType, GridMazeEnv
from gridworld_edu.envs.presets import big_maze_10x10, curriculum
from gridworld_edu.agents.qlearning import QLearningAgent


def train_episodes(env, agent, episodes):
    rows = []
    for _ in range(episodes):
        _, info = env.reset()
        s = info["state_index"]
        done = False
        while not done:
            a = agent.select_action(s)
            _, r, term, trunc, info2 = env.step(a)
            agent.update(s, a, r, info2["state_index"], term, trunc)
            s = info2["state_index"]
            done = term or trunc
        agent.decay_epsilon()
        # 以 is_success（到达终点）为准：trap_terminal=True 时 term 也可能是踩陷阱死亡
        rows.append(int(info2["is_success"]))
    return rows


def eval_greedy(env, agent, n=20):
    ok = 0
    for _ in range(n):
        _, info = env.reset()
        s = info["state_index"]
        done = False
        while not done:
            a = agent.select_action(s, greedy=True)
            _, _, term, trunc, info2 = env.step(a)
            s = info2["state_index"]
            done = term or trunc
        ok += int(info2["is_success"])
    return ok / n


def convergence_episode(succ_list, window=20, threshold=0.8):
    for i in range(len(succ_list)):
        lo = max(0, i - window + 1)
        if i + 1 >= window and sum(succ_list[lo:i + 1]) / (i - lo + 1) >= threshold:
            return i + 1
    return None


def mask_wall_actions(env: GridMazeEnv, agent: QLearningAgent):
    """把指向墙格的动作 Q 值重置为负（对所有奖励掩码变体生效）。"""
    n_bonus = int(np.sum(env.grid == CellType.BONUS))
    stride = 2 ** n_bonus
    for s in range(agent.q_table.shape[0]):
        cell = s // stride
        r, c = divmod(cell, env.width)
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.height and 0 <= nc < env.width) \
                    or env.grid[nr, nc] == CellType.WALL:
                agent.q_table[s, a] = min(agent.q_table[s, a], -1.0)


print("E4-fixed: 课程学习(动作掩码迁移) vs 直接训练, 600 ep, 5 seeds")
for mode in ("curriculum_masked", "direct"):
    srs, convs = [], []
    for seed in range(5):
        final_env = GridMazeEnv(big_maze_10x10())
        agent = QLearningAgent(final_env.n_states, seed=seed,
                               epsilon_decay_episodes=int(600 * 0.8))
        if mode == "curriculum_masked":
            all_succ = []
            for st in curriculum():
                env = GridMazeEnv(st["grid"])
                mask_wall_actions(env, agent)   # 关键修正：迁移时重置新墙动作
                all_succ += train_episodes(env, agent, 150)
            convs.append(convergence_episode(all_succ) or 601)
            srs.append(eval_greedy(final_env, agent))
        else:
            succ = train_episodes(final_env, agent, 600)
            convs.append(convergence_episode(succ) or 601)
            srs.append(eval_greedy(final_env, agent))
    print(f"  {mode}: succ={np.mean(srs):.3f}±{np.std(srs):.3f} conv={np.mean(convs):.1f}")
