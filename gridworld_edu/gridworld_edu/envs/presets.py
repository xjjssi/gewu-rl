"""presets.py —— 预设迷宫与课程学习序列

每个预设返回 (名称, grid 矩阵)。格子编码见 CellType：
0 空地 / 1 墙 / 2 起点 / 3 终点 / 4 奖励格 / 5 陷阱格。
"""

from __future__ import annotations

E, W, S, G, B, T = 0, 1, 2, 3, 4, 5


def empty_6x6() -> list[list[int]]:
    """空场 6×6：没有任何障碍，用于观察最朴素的 Q-Learning 收敛。"""
    return [
        [S, E, E, E, E, E],
        [E, E, E, E, E, E],
        [E, E, E, E, E, E],
        [E, E, E, E, E, E],
        [E, E, E, E, E, E],
        [E, E, E, E, E, G],
    ]


def single_wall_6x6() -> list[list[int]]:
    """单障碍 6×6：中间一堵竖墙留一个缺口，观察智能体学会绕行。"""
    return [
        [S, E, W, E, E, E],
        [E, E, W, E, B, E],
        [E, E, W, E, E, E],
        [E, E, E, E, E, E],
        [E, T, W, E, E, E],
        [E, E, W, E, E, G],
    ]


def multi_wall_8x8() -> list[list[int]]:
    """多障碍 8×8：含奖励格与陷阱格，适合演示奖励塑形的影响。"""
    return [
        [S, E, E, W, E, E, E, E],
        [E, W, E, W, E, W, E, E],
        [E, W, E, E, E, W, E, B],
        [E, W, W, W, E, W, E, W],
        [E, E, E, E, E, E, E, E],
        [E, W, E, T, W, W, W, E],
        [E, W, E, E, E, E, E, E],
        [E, W, E, W, E, E, E, G],
    ]


def big_maze_10x10() -> list[list[int]]:
    """大迷宫 10×10：较长路径 + 奖励/陷阱混合，适合课程学习的最后一级。"""
    return [
        [S, E, W, E, E, E, W, E, E, E],
        [E, E, W, E, W, E, W, E, W, E],
        [E, E, W, E, W, E, E, E, W, E],
        [E, E, W, E, W, W, W, E, W, E],
        [E, E, E, E, E, E, W, E, W, E],
        [W, W, W, W, W, E, W, E, E, E],
        [E, E, E, E, W, E, E, E, W, E],
        [E, B, W, E, W, W, W, E, W, E],
        [E, E, W, E, E, E, E, E, W, T],
        [E, W, W, W, W, E, W, E, E, G],
    ]


PRESETS: dict[str, dict] = {
    "empty_6x6": {"name": "空场 6×6", "grid": empty_6x6()},
    "single_wall_6x6": {"name": "单障碍 6×6", "grid": single_wall_6x6()},
    "multi_wall_8x8": {"name": "多障碍 8×8", "grid": multi_wall_8x8()},
    "big_maze_10x10": {"name": "大迷宫 10×10", "grid": big_maze_10x10()},
}


def random_maze(size: int = 8, wall_ratio: float = 0.2, seed: int | None = None) -> list[list[int]]:
    """随机迷宫生成：随机撒墙，起点左上、终点右下，附带少量奖励/陷阱格。

    注意：不保证连通，教学中可以让学生观察"无解迷宫"导致的训练失败。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=int)
    mask = rng.random((size, size)) < wall_ratio
    grid[mask] = W
    grid[0, 0] = S
    grid[size - 1, size - 1] = G
    # 在空地上随机放 2 个奖励格、2 个陷阱格
    empties = np.argwhere(grid == E)
    rng.shuffle(empties)
    for i, (r, c) in enumerate(empties[:4]):
        grid[r, c] = B if i % 2 == 0 else T
    return grid.tolist()


def _stage_with_start(start: tuple[int, int]) -> list[list[int]]:
    """完整大迷宫墙体布局，仅改变起点位置（逆向课程：起点由近及远）。

    所有关卡的墙体、奖励格、陷阱格完全一致，只移动起点——
    这样 Q 表迁移永远合法（不会像"逐关加墙"那样把旧策略指向新墙
    造成负迁移），且近起点关卡快速建立终点附近的准确价值，
    价值迭代会自然地向远处状态传播（动态规划的反向传播特性）。
    """
    grid = [row[:] for row in big_maze_10x10()]
    for r in range(10):
        for c in range(10):
            if grid[r][c] == S:
                grid[r][c] = E
    grid[start[0]][start[1]] = S
    return grid


def curriculum() -> list[dict]:
    """课程学习序列：完整迷宫布局，起点由近及远（逆向课程）。

    关卡 1~4 起点到终点的曼哈顿距离约为 1 → 5 → 10 → 18。
    状态空间完全一致，Q 表可直接迁移。
    """
    return [
        {"stage": 1, "name": "第一课：终点隔壁", "grid": _stage_with_start((9, 8))},
        {"stage": 2, "name": "第二课：中场", "grid": _stage_with_start((6, 7))},
        {"stage": 3, "name": "第三课：中远程", "grid": _stage_with_start((4, 4))},
        {"stage": 4, "name": "第四课：全程", "grid": _stage_with_start((0, 0))},
    ]
