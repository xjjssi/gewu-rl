"""test_evaluation.py —— 统一评估口径（gridworld_edu.evaluation）的行为契约。

覆盖：ε-评估、Q 值平局偏向终点、环路检测，以及
"完全没学会" vs "学会了但贪心未稳定" 的区分能力。
"""

import numpy as np
import pytest

from gridworld_edu.agents.qlearning import QLearningAgent
from gridworld_edu.envs.grid_maze import GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS
from gridworld_edu.evaluation import (
    evaluate_policy,
    evaluate_qlearning,
    qlearning_eval_action,
)


def _train(agent, env, episodes):
    for _ in range(episodes):
        _, info = env.reset()
        s = info["state_index"]
        done = False
        while not done:
            a = agent.select_action(s)
            _, r, terminated, truncated, info = env.step(a)
            agent.update(s, a, r, info["state_index"], terminated, truncated)
            s = info["state_index"]
            done = terminated or truncated
        agent.decay_epsilon()


def test_tie_break_prefers_goal_closer_action():
    """平局裁决：Q 值在容差内并列时，应选择执行后离终点更近的动作。"""
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig())
    agent = QLearningAgent(env.n_states, epsilon_start=0.0, seed=0)
    # 起点 (0,0)（state_index=0）：右（动作3）与下（动作1）都接近终点，
    # 左（2）/上（0）会撞墙原地不动（距离不变，更远）。四个动作 Q 值近似并列。
    agent.q_table[0] = [-0.07, -0.07, -0.07, -0.0701]
    rng = np.random.default_rng(0)
    chosen = {qlearning_eval_action(env, agent.q_table, 0, rng, epsilon=0.0)
              for _ in range(50)}
    assert chosen <= {1, 3}  # 只会选"下"或"右"，绝不选撞墙动作
    # tie_tol=0（旧口径）：在精确平局集 {0,1,2} 中随机打破（动作 3 的 Q 严格更小，不会被选）
    rng2 = np.random.default_rng(0)
    chosen2 = {qlearning_eval_action(env, agent.q_table, 0, rng2, epsilon=0.0, tie_tol=0.0)
               for _ in range(200)}
    assert chosen2 == {0, 1, 2}


def test_loop_detection_marks_circling_episode():
    """环路检测：在两个格子间反复横跳的策略应被记为绕圈回合、成功率为 0。"""
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig())
    # 策略：在起点 (0,0) 与 (0,1) 之间左右振荡
    def oscillate(obs, info):
        r, c = divmod(info["state_index"], env.width)
        return 0 if c == 1 else 3  # 在 (0,1) 向左，在 (0,0) 向右
    m = evaluate_policy(env, oscillate, n=5)
    assert m["success_rate"] == 0.0
    assert m["loop_rate"] == 1.0
    assert m["stable_success_rate"] == 0.0


def test_straight_policy_has_no_loop():
    """直达终点的确定性策略不应被误判为绕圈。"""
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig())

    def to_goal(obs, info):
        r, c = divmod(info["state_index"], env.width)
        return 3 if c < env.width - 1 else 1  # 先右到底，再向下
    m = evaluate_policy(env, to_goal, n=5)
    assert m["success_rate"] == 1.0
    assert m["loop_rate"] == 0.0
    assert m["stable_success_rate"] == 1.0


def test_epsilon_exploration_in_eval():
    """ε=1.0 时评估动作应完全随机（四个动作都可能出现）。"""
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig())
    agent = QLearningAgent(env.n_states, seed=0)
    rng = np.random.default_rng(0)
    chosen = {qlearning_eval_action(env, agent.q_table, 0, rng, epsilon=1.0)
              for _ in range(400)}
    assert chosen == {0, 1, 2, 3}


def test_evaluate_qlearning_trained_agent():
    """训练后的智能体：ε-评估成功率高、绕圈率低；返回字段完整（含纯贪心对照）。"""
    episodes = 200
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig())
    env.set_shaping_gamma(0.99)
    agent = QLearningAgent(env.n_states, seed=0,
                           epsilon_decay_episodes=int(episodes * 0.8))
    _train(agent, env, episodes)
    m = evaluate_qlearning(env, agent, n=20, seed=0)
    assert m["success_rate"] >= 0.9
    assert m["loop_rate"] <= 0.1
    for k in ("stable_success_rate", "pure_greedy_success_rate",
              "pure_greedy_loop_rate", "eval_episodes"):
        assert k in m
    assert m["eval_episodes"] == 20


def test_evaluate_qlearning_distinguishes_unstable_from_unlearned():
    """核心区分能力：构造"学会了但贪心未稳定"的 Q 表——
    ε-评估（含平局偏向终点）应成功，旧口径纯贪心（随机平局）可能绕圈；
    两组指标必须分开报告，而不是混为一个成功率。"""
    env = GridMazeEnv(PRESETS["big_maze_10x10"]["grid"], RewardConfig())
    env.set_shaping_gamma(0.99)
    agent = QLearningAgent(env.n_states, seed=1,
                           epsilon_decay_episodes=int(150 * 0.8))
    _train(agent, env, 150)
    m = evaluate_qlearning(env, agent, n=20, seed=1)
    # 该种子下旧口径纯贪心 0%（审查者实测复现），ε-评估应接近 100%
    assert m["success_rate"] >= 0.9
    assert m["pure_greedy_success_rate"] <= m["success_rate"]
    # loop_rate 与 pure_greedy_loop_rate 给出了"未稳定"的量化证据
    assert 0.0 <= m["loop_rate"] <= 1.0
    assert 0.0 <= m["pure_greedy_loop_rate"] <= 1.0
