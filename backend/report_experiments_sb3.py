"""report_experiments_sb3.py —— 报告第五章用 SB3 组实验（PPO/DQN，CPU，分两段运行）。

用法：
  python report_experiments_sb3.py ppo   # PPO 组（约 4 分钟）
  python report_experiments_sb3.py dqn   # DQN 组（约 2 分钟）
结果追加写入 backend/results/report/sb3_results.json

评估口径与 Q-Learning 组统一（gridworld_edu.evaluation.evaluate_sb3）：
deterministic predict + ε=0.05 随机覆盖 + 环路检测。
实际种子数：3（seeds=(0,1,2)，见 run_group 默认值；SB3 组 CPU 训练较慢，
未扩到 20 种子，引用时请以此为准）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gridworld_edu"))

from gridworld_edu.envs.grid_maze import GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS
from gridworld_edu.agents.sb3_agents import create_sb3_agent
from gridworld_edu.evaluation import evaluate_sb3

OUT = Path(__file__).resolve().parent / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "sb3_results.json"


def eval_sb3(model, grid, reward_config, n=20, seed=0):
    env = GridMazeEnv(grid, RewardConfig.from_dict(reward_config))
    return evaluate_sb3(env, model, n=n, seed=seed)["success_rate"]


def run_group(tag, env_id, algo, timesteps, reward_config, seeds=(0, 1, 2)):
    grid = PRESETS[env_id]["grid"]
    srs = []
    for seed in seeds:
        t0 = time.time()
        env = GridMazeEnv(grid, RewardConfig.from_dict(reward_config))
        model = create_sb3_agent(algo, env, seed=seed)
        model.learn(total_timesteps=timesteps)
        sr = eval_sb3(model, grid, reward_config, seed=seed)
        dt = time.time() - t0
        srs.append(sr)
        print(f"  {tag} seed={seed}: succ={sr:.2f} ({dt:.0f}s)", flush=True)
    return {"env": env_id, "algo": algo, "timesteps": timesteps,
            "reward_config": reward_config,
            "success_mean": round(float(np.mean(srs)), 3),
            "success_std": round(float(np.std(srs)), 3), "seeds": len(seeds)}


which = sys.argv[1] if len(sys.argv) > 1 else "ppo"
data = json.loads(OUT_FILE.read_text(encoding="utf-8")) if OUT_FILE.exists() else {}

# 各实验组定义：键 = 结果 JSON 中的键，值 = (环境, 算法, 步数, 奖励配置)
GROUPS = {
    "ppo_empty_20k": ("empty_6x6", "ppo", 20_000, {}),
    "ppo_wall_shaping_20k": ("single_wall_6x6", "ppo", 20_000, {"potential_shaping": True}),
    "ppo_wall_noshaping_20k": ("single_wall_6x6", "ppo", 20_000, {}),
    "ppo_wall_shaping_40k": ("single_wall_6x6", "ppo", 40_000, {"potential_shaping": True}),
    "ppo_wall_noshaping_40k": ("single_wall_6x6", "ppo", 40_000, {}),
    "dqn_wall_shaping_30k": ("single_wall_6x6", "dqn", 30_000, {"potential_shaping": True}),
}
BUNDLES = {"ppo": ["ppo_empty_20k", "ppo_wall_shaping_20k", "ppo_wall_noshaping_20k"],
           "dqn": ["dqn_wall_shaping_30k"]}

keys = BUNDLES.get(which, [which])
for key in keys:
    env_id, algo, ts, rc = GROUPS[key]
    data[key] = run_group(key, env_id, algo, ts, rc)

OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已写入 {OUT_FILE}")
