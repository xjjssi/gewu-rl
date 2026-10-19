"""test_qlearning.py —— 表格 Q-Learning：小预算训练后贪心策略应在空场达到高成功率。"""

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
