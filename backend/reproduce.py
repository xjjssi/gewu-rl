"""reproduce.py —— 一键复现脚本

不启动任何服务，直接在 3 个预设环境上运行 qlearning / ppo / dqn 各若干个 seed，
输出对比表（终端打印 + CSV）和 matplotlib 学习曲线图到 backend/results/。

用法：
    python reproduce.py                # 全量：3 环境 × 3 算法 × 3 seeds（约 15 分钟，PPO/DQN 占大头）
    python reproduce.py --quick        # 快速自检：单 seed（约 5 分钟）
    python reproduce.py --quick --algos qlearning,dqn   # 只跑部分算法

预期结果（默认预算：Q-Learning 600 回合，PPO/DQN 15k 步）：
    - Q-Learning：三个环境均 ~100% 成功率；
    - DQN：三个环境均 ~100%（需要 target_update_interval=250 等修正参数，
      SB3 默认的 10000 在小预算下目标网络从不更新，成功率为 0）；
    - PPO：空场/单障碍 ~100%；多障碍 8×8 在 15k 预算内通常学不通（~0%），
      需 ≥30k 步或开启势函数塑形（奖励塑形本来就用于缓解稀疏奖励下的样本效率问题）。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gridworld_edu"))

import numpy as np

from gridworld_edu.agents.qlearning import QLearningAgent
from gridworld_edu.agents.sb3_agents import create_sb3_agent
from gridworld_edu.envs.grid_maze import GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]
ENV_IDS = ["empty_6x6", "single_wall_6x6", "multi_wall_8x8"]


def moving_avg(xs: list[float], w: int = 20) -> list[float]:
    return [float(np.mean(xs[max(0, i - w + 1): i + 1])) for i in range(len(xs))]


def eval_greedy(env: GridMazeEnv, act_fn, n: int = 20) -> float:
    """用贪心策略跑 n 回合，返回成功率。"""
    success = 0
    for _ in range(n):
        obs, info = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(act_fn(obs, info))
            done = terminated or truncated
        success += int(info["is_success"])
    return success / n


def run_qlearning(env: GridMazeEnv, episodes: int, seed: int, gamma: float = 0.99) -> tuple[list[float], float]:
    agent = QLearningAgent(n_states=env.n_states, seed=seed, gamma=gamma,
                           epsilon_decay_episodes=int(episodes * 0.8))
    env.set_shaping_gamma(gamma)  # PBRS 塑形 γ 与智能体学习目标 γ 一致
    rewards = []
    for _ in range(episodes):
        _, info = env.reset()
        s = info["state_index"]
        total, done = 0.0, False
        while not done:
            a = agent.select_action(s)
            _, r, terminated, truncated, info = env.step(a)
            ns = info["state_index"]
            agent.update(s, a, r, ns, terminated, truncated)
            total += r
            s = ns
            done = terminated or truncated
        agent.decay_epsilon()
        rewards.append(total)
    sr = eval_greedy(env, lambda o, i: agent.select_action(i["state_index"], greedy=True))
    return rewards, sr


def run_sb3(algo: str, env: GridMazeEnv, timesteps: int, seed: int) -> tuple[list[float], float]:
    from stable_baselines3.common.monitor import Monitor

    model = create_sb3_agent(algo, env, seed=seed)
    monitor = Monitor(env)
    model.set_env(monitor)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    # 从 Monitor 日志重建回合回报曲线
    rewards = [float(r) for r in monitor.get_episode_rewards()]
    sr = eval_greedy(env, lambda o, i: int(model.predict(o, deterministic=True)[0]))
    return rewards, sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="快速自检模式：默认只跑 1 个 seed（可用 --seeds 覆盖）")
    parser.add_argument("--episodes", type=int, default=None, help="Q-Learning 回合数（覆盖默认值）")
    parser.add_argument("--timesteps", type=int, default=None, help="PPO/DQN 步数（覆盖默认值）")
    parser.add_argument("--seeds", type=int, default=None, help="每种配置的 seed 个数（默认全量 3 / quick 1）")
    parser.add_argument("--algos", type=str, default=None,
                        help="只跑部分算法，逗号分隔，如 qlearning,dqn")
    args = parser.parse_args()
    episodes = args.episodes or (300 if args.quick else 600)
    # 15k 步是 DQN/PPO 在单障碍迷宫上稳定学成的最小预算量级；
    # 多障碍 8×8 的 PPO 需要更多步数或塑形辅助（见文件 docstring 的预期结果表）
    timesteps = args.timesteps or 15_000
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 3)
    seeds = SEEDS[: n_seeds]
    algos = [a.strip() for a in args.algos.split(",")] if args.algos else ["qlearning", "ppo", "dqn"]

    rows = []           # 对比表行
    curves: dict[tuple, list[float]] = {}  # (env, algo) -> 平均曲线
    for env_id in ENV_IDS:
        grid = PRESETS[env_id]["grid"]
        for algo in algos:
            srs, curve_list = [], []
            for seed in seeds:
                env = GridMazeEnv(grid, RewardConfig())
                t0 = time.time()
                if algo == "qlearning":
                    rewards, sr = run_qlearning(env, episodes, seed)
                else:
                    rewards, sr = run_sb3(algo, env, timesteps, seed)
                srs.append(sr)
                curve_list.append(moving_avg(rewards))
                rows.append([env_id, algo, seed, round(sr, 3),
                             round(float(np.mean(rewards[-50:])), 3), round(time.time() - t0, 1)])
                print(f"[{env_id:>16}] {algo:>9} seed={seed}  成功率={sr:.2f}  "
                      f"末50回合均回报={np.mean(rewards[-50:]):+.3f}  耗时={time.time()-t0:.1f}s")
            # 对齐长度后求平均曲线
            min_len = min(len(c) for c in curve_list)
            curves[(env_id, algo)] = np.mean([c[:min_len] for c in curve_list], axis=0).tolist()

    # ---- 对比表 CSV ----
    csv_path = RESULTS_DIR / "compare_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["环境", "算法", "seed", "评估成功率", "末50回合均回报", "耗时(秒)"])
        w.writerows(rows)

    print(f"\n========== 汇总（{len(seeds)} 个 seed 平均成功率） ==========")
    print(f"{'环境':<18}{''.join(f'{a:<12}' for a in algos)}")
    for env_id in ENV_IDS:
        cells = []
        for algo in algos:
            vals = [r[3] for r in rows if r[0] == env_id and r[1] == algo]
            cells.append(f"{np.mean(vals):.2f}±{np.std(vals):.2f}")
        print(f"{PRESETS[env_id]['name']:<18}{''.join(f'{c:<12}' for c in cells)}")

    # ---- 学习曲线图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 中文字体兜底：依次尝试常见 Windows 中文字体
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, len(ENV_IDS), figsize=(6 * len(ENV_IDS), 4), sharey=False)
    colors = {"qlearning": "#2563eb", "ppo": "#16a34a", "dqn": "#dc2626"}
    for ax, env_id in zip(axes, ENV_IDS):
        for algo in algos:
            c = curves[(env_id, algo)]
            ax.plot(range(1, len(c) + 1), c, label=algo, color=colors.get(algo), linewidth=1.5)
        ax.set_title(PRESETS[env_id]["name"])
        ax.set_xlabel("回合")
        ax.set_ylabel("回报（20 回合滑动平均）")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("格物RL 复现实验：三种算法 × 三个预设环境（多种子平均）")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "learning_curves.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n对比表已保存: {csv_path}")
    print(f"曲线图已保存: {fig_path}")


if __name__ == "__main__":
    main()
