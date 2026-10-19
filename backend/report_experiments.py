"""report_experiments.py —— 技术报告第五章用正式实验（快速组：表格法与消融、意图、课程学习）

产出 JSON + CSV 到 backend/results/report/，供技术报告引用。
PPO/DQN 组见 report_experiments_sb3.py（耗时较长，单独运行）。

用法：
    python report_experiments.py                 # 完整重跑：默认 20 个种子（配合
                                                 # report_experiments_sb3.py 约需半天，请预留时间）
    python report_experiments.py --quick         # 快速自检：3 种子 + 削减预算，几分钟内跑通
    python report_experiments.py --seeds 8       # 自定义种子数
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gridworld_edu"))

from gridworld_edu.envs.grid_maze import GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS, big_maze_10x10, curriculum
from gridworld_edu.agents.qlearning import QLearningAgent

OUT = Path(__file__).resolve().parent / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)

_parser = argparse.ArgumentParser()
_parser.add_argument("--seeds", type=int, default=20, help="每组实验的种子数（默认 20）")
_parser.add_argument("--quick", action="store_true", help="快速自检：3 种子 + 削减预算")
_args = _parser.parse_args()
N_SEEDS = 3 if _args.quick else _args.seeds
E1_EPISODES = 300 if _args.quick else 600      # E1 每环境每种子回合数
E2_EPISODES = 120 if _args.quick else 150      # E2 消融预算
E3_EPISODES = 120 if _args.quick else 150      # E3 意图实验预算
E4_BUDGET = 160 if _args.quick else 200        # E4 课程学习总预算


def train_qlearning(grid, reward_config=None, episodes=600, seed=0, gamma=0.99):
    """训练表格 Q-Learning，返回 (agent, env, metrics_rows)。"""
    env = GridMazeEnv(grid, RewardConfig.from_dict(reward_config or {}))
    env.set_shaping_gamma(gamma)  # PBRS 塑形 γ 与智能体学习目标 γ 一致
    agent = QLearningAgent(env.n_states, seed=seed, gamma=gamma,
                           epsilon_decay_episodes=int(episodes * 0.8))
    rows = []
    for ep in range(1, episodes + 1):
        _, info = env.reset()
        s = info["state_index"]
        done = False
        total, steps, success = 0.0, 0, False
        while not done:
            a = agent.select_action(s)
            _, r, term, trunc, info2 = env.step(a)
            agent.update(s, a, r, info2["state_index"], term, trunc)
            s = info2["state_index"]
            total += r
            steps += 1
            done = term or trunc
            # 以 is_success（到达终点）为准：trap_terminal=True 时 term 也可能是踩陷阱死亡
            success = info2["is_success"]
        agent.decay_epsilon()
        rows.append((ep, round(total, 4), steps, int(success)))
    return agent, env, rows


def eval_greedy(env, agent, n=20):
    ok = 0
    steps_list = []
    for _ in range(n):
        _, info = env.reset()
        s = info["state_index"]
        done = False
        steps = 0
        while not done:
            a = agent.select_action(s, greedy=True)
            _, _, term, trunc, info2 = env.step(a)
            s = info2["state_index"]
            steps += 1
            done = term or trunc
        ok += int(info2["is_success"])
        steps_list.append(steps)
    return {"success_rate": round(ok / n, 3),
            "avg_steps": round(float(np.mean(steps_list)), 1)}


def convergence_episode(rows, window=20, threshold=0.8):
    succ = [r[3] for r in rows]
    for i in range(len(succ)):
        lo = max(0, i - window + 1)
        if i + 1 >= window and sum(succ[lo:i + 1]) / (i - lo + 1) >= threshold:
            return i + 1
    return None


results = {}

# ---------- E1：Q-Learning 三环境 × N_SEEDS 种子 ----------
print(f"E1: Q-Learning 三环境 x {N_SEEDS} 种子, {E1_EPISODES} ep")
e1 = []
for env_id, p in PRESETS.items():
    srs, convs = [], []
    for seed in range(N_SEEDS):
        agent, env, rows = train_qlearning(p["grid"], episodes=E1_EPISODES, seed=seed)
        m = eval_greedy(env, agent)
        srs.append(m["success_rate"])
        convs.append(convergence_episode(rows))
    e1.append({"env": env_id, "name": p["name"], "algo": "qlearning",
               "success_mean": round(float(np.mean(srs)), 3),
               "success_std": round(float(np.std(srs)), 3),
               "converge_mean": round(float(np.mean([c for c in convs if c])), 1),
               "seeds": N_SEEDS})
    # 随机基线
    rng = np.random.default_rng(0)
    env = GridMazeEnv(p["grid"])
    rand_ok = 0
    for _ in range(20):
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, rinfo = env.step(int(rng.integers(4)))
            done = term or trunc
        rand_ok += int(rinfo["is_success"])
    e1.append({"env": env_id, "name": p["name"], "algo": "random",
               "success_mean": round(rand_ok / 20, 3), "success_std": 0,
               "converge_mean": None, "seeds": 1})
    print(f"  {env_id}: ql={e1[-2]['success_mean']}±{e1[-2]['success_std']} rand={e1[-1]['success_mean']}")
results["E1_baselines"] = {"seeds": N_SEEDS, "episodes": E1_EPISODES, "rows": e1}

# ---------- E2：势函数塑形消融（大迷宫，有限预算）----------
print(f"E2: PBRS 消融 @ big_maze_10x10, {E2_EPISODES} ep, {N_SEEDS} seeds")
e2 = []
for shaping in (True, False):
    srs, convs, steps_l = [], [], []
    for seed in range(N_SEEDS):
        agent, env, rows = train_qlearning(
            big_maze_10x10(), {"potential_shaping": shaping}, episodes=E2_EPISODES, seed=seed)
        m = eval_greedy(env, agent)
        srs.append(m["success_rate"])
        steps_l.append(m["avg_steps"])
        c = convergence_episode(rows)
        convs.append(c if c else E2_EPISODES + 1)
    e2.append({"potential_shaping": shaping,
               "success_mean": round(float(np.mean(srs)), 3),
               "success_std": round(float(np.std(srs)), 3),
               "converge_mean": round(float(np.mean(convs)), 1),
               "avg_steps": round(float(np.mean(steps_l)), 1)})
    print(f"  shaping={shaping}: succ={e2[-1]['success_mean']}±{e2[-1]['success_std']} conv={e2[-1]['converge_mean']}")
results["E2_pbrs_ablation"] = {"seeds": N_SEEDS, "episodes": E2_EPISODES, "rows": e2}

# ---------- E3：AI 奖励设计师意图实验（规则引擎路径，三组对照）----------
# 三组：默认方案 / 默认+PBRS / AI 方案。AI 的价值体现在第三组相对第二组的增量，
# 而不是"PBRS 开 vs 关"（否则图 5-2b 实质上只是 PBRS 消融的重复）。
print(f"E3: 意图实验（三组对照）@ big_maze_10x10, {E3_EPISODES} ep, {N_SEEDS} seeds")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_reward import design_reward  # noqa: E402

design = design_reward("让智能体尽快到达终点，步数越少越好", big_maze_10x10(), None)
e3 = []
for tag, rc in [("默认", {}), ("默认+PBRS", {"potential_shaping": True}),
                ("AI生成", design["reward_config"])]:
    srs, convs = [], []
    for seed in range(N_SEEDS):
        agent, env, rows = train_qlearning(big_maze_10x10(), rc, episodes=E3_EPISODES, seed=seed)
        m = eval_greedy(env, agent)
        srs.append(m["success_rate"])
        c = convergence_episode(rows)
        convs.append(c if c else E3_EPISODES + 1)
    e3.append({"group": tag, "reward_config": rc,
               "success_mean": round(float(np.mean(srs)), 3),
               "success_std": round(float(np.std(srs)), 3),
               "converge_mean": round(float(np.mean(convs)), 1)})
    print(f"  {tag}: succ={e3[-1]['success_mean']}±{e3[-1]['success_std']} conv={e3[-1]['converge_mean']}")
results["E3_intent_experiment"] = {"instruction": "让智能体尽快到达终点，步数越少越好",
                                   "seeds": N_SEEDS, "episodes": E3_EPISODES,
                                   "design": design, "groups": e3}

# ---------- E4：逆向课程学习 vs 直接训练 ----------
print(f"E4: 逆向课程学习 vs 直接训练, 总预算 {E4_BUDGET} ep, {N_SEEDS} seeds")


def train_episodes(env, agent, episodes, restart_sched=False):
    if restart_sched:
        # 阶段内探索率与学习率重启：否则到后期关卡 ε/α 已衰减殆尽，
        # 无法学习"新起点到已学会区域"之间的路段（课程迁移的经典陷阱）
        agent.epsilon = agent.epsilon_start
        agent.lr = agent.lr_start
        agent._eps_step = (agent.epsilon_start - agent.epsilon_end) / max(1, int(episodes * 0.8))
        agent._lr_step = (agent.lr_start - agent.lr_end) / max(1, int(episodes * 0.8))
    rows = []
    for ep in range(1, episodes + 1):
        _, info = env.reset()
        s = info["state_index"]
        done = False
        total, steps, success = 0.0, 0, False
        while not done:
            a = agent.select_action(s)
            _, r, term, trunc, info2 = env.step(a)
            agent.update(s, a, r, info2["state_index"], term, trunc)
            s = info2["state_index"]
            total += r
            steps += 1
            done = term or trunc
            # 同上：以 is_success 为准，避免陷阱终止被误计为成功
            success = info2["is_success"]
        agent.decay_epsilon()
        rows.append((ep, round(total, 4), steps, int(success)))
    return rows


e4 = []
for mode in ("curriculum", "direct"):
    srs, convs = [], []
    for seed in range(N_SEEDS):
        final_env = GridMazeEnv(big_maze_10x10())
        agent = QLearningAgent(final_env.n_states, seed=seed,
                               epsilon_decay_episodes=int(E4_BUDGET * 0.8))
        if mode == "curriculum":
            # 逆向课程：同一 Q 表在"起点由近及远"的 4 个关卡上逐级迁移
            all_rows = []
            for st in curriculum():
                env = GridMazeEnv(st["grid"])
                all_rows += train_episodes(env, agent, E4_BUDGET // 4, restart_sched=True)
        else:
            all_rows = train_episodes(final_env, agent, E4_BUDGET)
        convs.append(convergence_episode(all_rows) or E4_BUDGET + 1)
        srs.append(eval_greedy(final_env, agent)["success_rate"])
    e4.append({"mode": mode, "success_mean": round(float(np.mean(srs)), 3),
               "success_std": round(float(np.std(srs)), 3),
               "converge_mean": round(float(np.mean(convs)), 1)})
    print(f"  {mode}: succ={e4[-1]['success_mean']}±{e4[-1]['success_std']} conv={e4[-1]['converge_mean']}")
results["E4_curriculum"] = {"note": f"逆向课程（起点距终点 1→5→10→18 曼哈顿距离）{E4_BUDGET // 4} 回合×4 级，Q 表迁移 + 阶段内探索率重启；直接组 {E4_BUDGET} 回合。最终迷宫贪心评估 20 回合。",
                            "budget": E4_BUDGET, "seeds": N_SEEDS, "groups": e4}

with open(OUT / "report_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已写入 {OUT / 'report_results.json'}")
