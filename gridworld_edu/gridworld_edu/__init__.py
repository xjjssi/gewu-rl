"""gridworld_edu —— 格物RL 教学用网格迷宫强化学习核心包。

包含：
- envs.grid_maze.GridMazeEnv：Gymnasium 兼容的网格迷宫环境
- envs.presets：预设迷宫与课程学习序列
- agents.qlearning.QLearningAgent：从零实现的表格 Q-Learning（教学用）
- agents.sb3_agents：Stable-Baselines3 PPO/DQN 封装
"""

from gridworld_edu.envs.grid_maze import GridMazeEnv, RewardConfig, CellType

__all__ = ["GridMazeEnv", "RewardConfig", "CellType"]
__version__ = "0.1.0"
