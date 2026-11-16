"""teaching_observations.py —— 复现《实验指导书》中"预期现象"所用的数据

用法（项目根目录）：python backend/teaching_observations.py
输出各实验配置在 5 个种子上的平均：评估成功率、收敛回合（滑动 20 回合成功率 ≥80% 的首达回合）、
最后一条代表性轨迹中探索步的占比、训练进行到 1/4 时价值快照中正价值格子数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402
from gridworld_edu.envs.presets import PRESETS  # noqa: E402

CASES = [
    ("实验一 · 基线（大迷宫 10×10，500 回合）", "big_maze_10x10", {}, {}),
    ("实验一 · ε 终止值 0.2", "big_maze_10x10", {"epsilon_end": 0.2}, {}),
    ("实验一 · ε 初始值 0.1（几乎不探索）", "big_maze_10x10", {"epsilon_start": 0.1}, {}),
    ("实验二 · 150 回合，无塑形", "big_maze_10x10", {"episodes": 150}, {}),
    ("实验二 · 150 回合，开启 PBRS", "big_maze_10x10", {"episodes": 150}, {"potential_shaping": True}),
    ("实验二 · γ = 0.8", "big_maze_10x10", {"gamma": 0.8}, {}),
]


def observe(preset: str, hp: dict, reward: dict, seeds=range(5)) -> dict:
    rows = []
    for seed in seeds:
        req = main.TrainRequest(maze=PRESETS[preset]["grid"], hyperparams={"episodes": 500, **hp},
                                reward_config=reward, seed=seed)
        session = main.TrainSession(f"obs-{seed}", req)
        summary, _ = main.run_qlearning(session)
        trajs = [e for e in session.buffer if e["type"] == "trajectory"]
        last = trajs[-1]["decisions"]
        quarter = trajs[len(trajs) // 4]["policy"]
        rows.append((summary["eval_success_rate"],
                     main._convergence_episode(session.metrics_rows) or np.nan,
                     sum(d["x"] for d in last) / len(last),
                     sum(c["q_max"] > 0 for c in quarter)))
    a = np.array(rows, dtype=float)
    return {"success": a[:, 0].mean(), "converge": np.nanmean(a[:, 1]), "late_explore": a[:, 2].mean(),
            "positive_cells_at_quarter": a[:, 3].mean(), "cells": len(trajs[-1]["policy"])}


if __name__ == "__main__":
    for name, preset, hp, reward in CASES:
        o = observe(preset, hp, reward)
        print(f"{name:<28} 成功率 {o['success']:.0%} | 收敛回合 {o['converge']:.1f} | "
              f"末段探索占比 {o['late_explore']:.0%} | 1/4 处正价值格 {o['positive_cells_at_quarter']:.1f}/{o['cells']}")
