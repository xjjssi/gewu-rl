"""report_experiments_sb3.py —— 报告第五章用 SB3 组实验（PPO/DQN，CPU，分两段运行）。

用法：
  python report_experiments_sb3.py ppo   # PPO 组（约 4 分钟）
  python report_experiments_sb3.py dqn   # DQN 组（约 2 分钟）
结果追加写入 backend/results/report/sb3_results.json
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

OUT = Path(__file__).resolve().parent / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "sb3_results.json"


def eval_sb3(model, grid, reward_config, n=20):
    env = GridMazeEnv(grid, RewardConfig.from_dict(reward_config))
    ok = 0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            a = int(model.predict(obs, deterministic=True)[0])
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        # 以 is_success（到达终点）为准：trap_terminal=True 时 term 也可能是踩陷阱死亡
        ok += int(info["is_success"])
    return ok / n


def run_group(tag, env_id, algo, timesteps, reward_config, seeds=(0, 1, 2)):
    grid = PRESETS[env_id]["grid"]
    srs = []
    for seed in seeds:
        t0 = time.time()
        env = GridMazeEnv(grid, RewardConfig.from_dict(reward_config))
        model = create_sb3_agent(algo, env, seed=seed)
        model.learn(total_timesteps=timesteps)
        sr = eval_sb3(model, grid, reward_config)
        dt = time.time() - t0
        srs.append(sr)
        print(f"  {tag} seed={seed}: succ={sr:.2f} ({dt:.0f}s)", flush=True)
    return {"env": env_id, "algo": algo, "timesteps": timesteps,
            "reward_config": reward_config,
            "success_mean": round(float(np.mean(srs)), 3),
            "success_std": round(float(np.std(srs)), 3), "seeds": len(seeds)}


which = sys.argv[1] if len(sys.argv) > 1 else "ppo"
data = json.loads(OUT_FILE.read_text(encoding="utf-8")) if OUT_FILE.exists() else {}

if which == "ppo":
    data["ppo_empty_20k"] = run_group("PPO@empty", "empty_6x6", "ppo", 20_000, {})
    data["ppo_wall_shaping_20k"] = run_group(
        "PPO@wall+shaping", "single_wall_6x6", "ppo", 20_000, {"potential_shaping": True})
    data["ppo_wall_noshaping_20k"] = run_group(
        "PPO@wall", "single_wall_6x6", "ppo", 20_000, {})
elif which == "dqn":
    data["dqn_wall_shaping_30k"] = run_group(
        "DQN@wall+shaping", "single_wall_6x6", "dqn", 30_000, {"potential_shaping": True})

OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已写入 {OUT_FILE}")
