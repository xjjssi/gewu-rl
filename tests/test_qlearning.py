"""test_qlearning.py —— 表格 Q-Learning：小预算训练后贪心策略应在空场达到高成功率。"""

import numpy as np
import pytest

from gridworld_edu.agents.qlearning import QLearningAgent
from gridworld_edu.envs.grid_maze import GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS


def test_qlearning_converges_on_empty_field():
    """空场 6×6、200 回合小预算：贪心评估成功率应 ≥ 90%。"""
    episodes = 200
    env = GridMazeEnv(PRESETS["empty_6x6"]["grid"], RewardConfig())
    env.set_shaping_gamma(0.99)
    agent = QLearningAgent(env.n_states, seed=0,
                           epsilon_decay_episodes=int(episodes * 0.8))
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

    success = 0
    for _ in range(20):
        _, info = env.reset()
        s = info["state_index"]
        done = False
        while not done:
            a = agent.select_action(s, greedy=True)
            _, _, terminated, truncated, info = env.step(a)
            s = info["state_index"]
            done = terminated or truncated
        success += int(info["is_success"])
    assert success / 20 >= 0.9


def test_epsilon_decay_floor_and_schedule():
    """ε 线性衰减：按步长单调下降，衰减到下界 epsilon_end 后不再继续下降。"""
    agent = QLearningAgent(n_states=4, epsilon_start=1.0, epsilon_end=0.05,
                           epsilon_decay_episodes=10, seed=0)
    step = (1.0 - 0.05) / 10
    prev = agent.epsilon
    for i in range(1, 12):  # 多衰减 2 次，验证下界不被击穿
        agent.decay_epsilon()
        if i <= 10:
            assert agent.epsilon == pytest.approx(max(0.05, 1.0 - step * i))
        assert agent.epsilon <= prev
        prev = agent.epsilon
    assert agent.epsilon == pytest.approx(0.05)  # 停在下界
    # 学习率同样衰减到下界（初始值的 1/10）
    assert agent.lr == pytest.approx(agent.lr_end)


def test_q_update_formula_numeric():
    """Q 更新公式数值验证：Q ← Q + α·(r + γ·maxQ(s′) − Q)；终止时 TD 目标只有 r。"""
    agent = QLearningAgent(n_states=3, learning_rate=0.5, gamma=0.9,
                           epsilon_start=0.0, seed=0)
    agent.q_table[1, 2] = 4.0  # 后继状态 s′=1 的最大 Q 值
    # 非终止：td_target = 1.0 + 0.9*4.0 = 4.6；Q(0,1) = 0 + 0.5*(4.6-0) = 2.3
    td = agent.update(0, 1, reward=1.0, next_state=1, terminated=False, truncated=False)
    assert agent.q_table[0, 1] == pytest.approx(2.3)
    assert td == pytest.approx(4.6)
    # 终止：td_target = r = -1.0；Q(2,0) = 0 + 0.5*(-1.0-0) = -0.5
    agent.update(2, 0, reward=-1.0, next_state=0, terminated=True, truncated=False)
    assert agent.q_table[2, 0] == pytest.approx(-0.5)
    # 截断仍 bootstrap（与终止不同）：td_target = 0 + 0.9*4.0 = 3.6
    agent.update(2, 3, reward=0.0, next_state=1, terminated=False, truncated=True)
    assert agent.q_table[2, 3] == pytest.approx(1.8)


def test_greedy_tie_break_is_random():
    """贪心平局打破：两个动作 Q 值并列最大时，多次选择应覆盖两个动作（不偏向固定方向）。"""
    agent = QLearningAgent(n_states=1, epsilon_start=0.0, seed=0)
    agent.q_table[0] = [1.0, 2.0, 2.0, 0.0]  # 动作 1 与 2 并列最大
    chosen = {agent.select_action(0, greedy=True) for _ in range(200)}
    assert chosen == {1, 2}
    # 无平局时贪心唯一确定
    agent.q_table[0] = [0.0, 1.0, 2.0, 1.5]
    assert all(agent.select_action(0, greedy=True) == 2 for _ in range(50))
    # 探索路径：ε=1 时四个动作都可能被选到
    agent.epsilon = 1.0
    explored = {agent.select_action(0) for _ in range(400)}
    assert explored == {0, 1, 2, 3}
