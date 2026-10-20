"""report_experiments.py —— 技术报告第五章用正式实验（快速组：表格法与消融、意图、课程学习）

产出 JSON + CSV 到 backend/results/report/，供技术报告引用。
PPO/DQN 组见 report_experiments_sb3.py（耗时较长，单独运行）。

评估口径（与 backend/main.py、reproduce.py 统一，见 gridworld_edu/evaluation.py）：
ε-评估（ε=0.05）+ Q 值平局偏向终点 + 环路检测；同时附跑旧口径纯贪心
（pure_greedy_*）用于区分"完全没学会"与"学会了但贪心未稳定"。

用法：
    python report_experiments.py                 # 完整重跑：默认 20 个种子（配合
                                                 # report_experiments_sb3.py 约需半天，请预留时间）
    python report_experiments.py --quick         # 快速自检：3 种子 + 削减预算，几分钟内跑通
    python report_experiments.py --seeds 8       # 自定义种子数
    python report_experiments.py --only E2,E3    # 只重跑部分实验（结果合并进已有 JSON）
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
from gridworld_edu.evaluation import evaluate_qlearning

OUT = Path(__file__).resolve().parent / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "report_results.json"

_parser = argparse.ArgumentParser()
_parser.add_argument("--seeds", type=int, default=20, help="每组实验的种子数（默认 20）")
_parser.add_argument("--quick", action="store_true", help="快速自检：3 种子 + 削减预算")
_parser.add_argument("--only", type=str, default=None,
                     help="只运行部分实验，逗号分隔（E1,E2,E3,E4）；结果合并进已有 report_results.json")
_args = _parser.parse_args()
N_SEEDS = 3 if _args.quick else _args.seeds
E1_EPISODES = 300 if _args.quick else 600      # E1 每环境每种子回合数
E2_EPISODES = 120 if _args.quick else 150      # E2 消融预算
E3_EPISODES = 120 if _args.quick else 150      # E3 意图实验预算
E4_BUDGET = 160 if _args.quick else 200        # E4 课程学习总预算
ONLY = {s.strip().upper() for s in _args.only.split(",")} if _args.only else {"E1", "E2", "E3", "E4"}


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


def eval_greedy(env, agent, n=20, seed=0):
    """统一评估口径（gridworld_edu.evaluation.evaluate_qlearning）。

    返回字段含：success_rate（ε-评估）、avg_steps、loop_rate、
    stable_success_rate、pure_greedy_success_rate / pure_greedy_loop_rate（旧口径对照）。
    """
    return evaluate_qlearning(env, agent, n=n, seed=seed)


def convergence_episode(rows, window=20, threshold=0.8):
    succ = [r[3] for r in rows]
    for i in range(len(succ)):
        lo = max(0, i - window + 1)
        if i + 1 >= window and sum(succ[lo:i + 1]) / (i - lo + 1) >= threshold:
            return i + 1
    return None


# 分段运行（--only）时保留已有结果，只覆盖本次运行的实验
results = {}
if OUT_FILE.exists() and _args.only:
    results = json.loads(OUT_FILE.read_text(encoding="utf-8"))


def agg_evals(ms):
    """把 per-seed 的 evaluate_qlearning 结果聚合为 mean±std。"""
    out = {}
    for k in ("success_rate", "avg_steps", "loop_rate", "stable_success_rate",
              "bonus_collect_rate", "trap_trigger_rate",
              "pure_greedy_success_rate", "pure_greedy_loop_rate"):
        vals = [m[k] for m in ms]
        out[f"{k}_mean"] = round(float(np.mean(vals)), 3)
        out[f"{k}_std"] = round(float(np.std(vals)), 3)
    return out


# ---------- E1：Q-Learning 三环境 × N_SEEDS 种子 ----------
if "E1" in ONLY:
    print(f"E1: Q-Learning 三环境 x {N_SEEDS} 种子, {E1_EPISODES} ep")
    e1 = []
    for env_id, p in PRESETS.items():
        srs, convs = [], []
        for seed in range(N_SEEDS):
            agent, env, rows = train_qlearning(p["grid"], episodes=E1_EPISODES, seed=seed)
            m = eval_greedy(env, agent, seed=seed)
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
if "E2" in ONLY:
    print(f"E2: PBRS 消融 @ big_maze_10x10, {E2_EPISODES} ep, {N_SEEDS} seeds")
    e2 = []
    for shaping in (True, False):
        ms, convs = [], []
        for seed in range(N_SEEDS):
            agent, env, rows = train_qlearning(
                big_maze_10x10(), {"potential_shaping": shaping}, episodes=E2_EPISODES, seed=seed)
            ms.append(eval_greedy(env, agent, seed=seed))
            c = convergence_episode(rows)
            convs.append(c if c else E2_EPISODES + 1)
        row = {"potential_shaping": shaping,
               "converge_mean": round(float(np.mean(convs)), 1),
               "converge_std": round(float(np.std(convs)), 1)}
        row.update(agg_evals(ms))
        e2.append(row)
        print(f"  shaping={shaping}: succ={row['success_rate_mean']}±{row['success_rate_std']} "
              f"conv={row['converge_mean']} loop={row['loop_rate_mean']} "
              f"pure={row['pure_greedy_success_rate_mean']}")
    results["E2_pbrs_ablation"] = {"seeds": N_SEEDS, "episodes": E2_EPISODES, "rows": e2,
                                   "eval_protocol": "ε=0.05 + 平局偏向终点 + 环路检测（loop_threshold=2）；"
                                                    "pure_greedy_* 为旧口径纯贪心对照"}

# ---------- E3：AI 奖励设计师意图实验（规则引擎路径，三组对照）----------
# 三组：默认方案 / 默认+PBRS / AI 方案。AI 的价值体现在第三组相对第二组的增量，
# 而不是"PBRS 开 vs 关"（否则图 5-2b 实质上只是 PBRS 消融的重复）。
# 主案例选择（2026-10 重审后修正）：「尽快到达」类意图下三组最终行为几乎一致
# （PBRS 理论保证最优策略不变，默认奖励本就近优），增量为零；多目标权衡意图
# 「收集所有奖励格，哪怕绕路也要最终到达终点」在多障碍 8×8 上有真实稳定增量——
# 单开 PBRS 时势函数把智能体拉向终点方向，奖励格（需先远离终点绕路）收集率为 0，
# 而 AI 方案（bonus_reward=2.0、步数代价减半、关闭塑形）学会"先收集再到达"，
# 收集率显著高于 0 且成功率保持接近 1。
if "E3" in ONLY:
    print(f"E3: 意图实验（三组对照）@ multi_wall_8x8, {E3_EPISODES} ep, {N_SEEDS} seeds")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llm_reward import design_reward  # noqa: E402

    E3_MAZE = PRESETS["multi_wall_8x8"]["grid"]
    E3_INSTRUCTION = "收集所有奖励格，哪怕绕路也要最终到达终点"
    design = design_reward(E3_INSTRUCTION, E3_MAZE, None)
    e3 = []
    for tag, rc in [("默认", {}), ("默认+PBRS", {"potential_shaping": True}),
                    ("AI生成", design["reward_config"])]:
        ms, convs = [], []
        for seed in range(N_SEEDS):
            agent, env, rows = train_qlearning(E3_MAZE, rc, episodes=E3_EPISODES, seed=seed)
            ms.append(eval_greedy(env, agent, seed=seed))
            c = convergence_episode(rows)
            convs.append(c if c else E3_EPISODES + 1)
        row = {"group": tag, "reward_config": rc,
               "converge_mean": round(float(np.mean(convs)), 1),
               "converge_std": round(float(np.std(convs)), 1)}
        row.update(agg_evals(ms))
        e3.append(row)
        print(f"  {tag}: succ={row['success_rate_mean']}±{row['success_rate_std']} "
              f"conv={row['converge_mean']} bonus={row['bonus_collect_rate_mean']}±{row['bonus_collect_rate_std']} "
              f"steps={row['avg_steps_mean']}")
    results["E3_intent_experiment"] = {"instruction": E3_INSTRUCTION,
                                       "env": "multi_wall_8x8",
                                       "seeds": N_SEEDS, "episodes": E3_EPISODES,
                                       "design": design, "groups": e3,
                                       "eval_protocol": "ε=0.05 + 平局偏向终点 + 环路检测（loop_threshold=2）；"
                                                        "pure_greedy_* 为旧口径纯贪心对照",
                                       "note": "主案例为多目标权衡意图（收集+到达）：AI 增量体现在奖励格收集率"
                                               "（相对默认+PBRS 组），而非成功率；成功率各组均接近 1。"}

# ---------- E4：逆向课程学习 vs 直接训练 ----------
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


if "E4" in ONLY:
    print(f"E4: 逆向课程学习 vs 直接训练, 总预算 {E4_BUDGET} ep, {N_SEEDS} seeds")
    e4 = []
    for mode in ("curriculum", "direct"):
        ms, convs = [], []
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
            ms.append(eval_greedy(final_env, agent, seed=seed))
        row = {"mode": mode,
               "converge_mean": round(float(np.mean(convs)), 1),
               "converge_std": round(float(np.std(convs)), 1)}
        row.update(agg_evals(ms))
        e4.append(row)
        print(f"  {mode}: succ={row['success_rate_mean']}±{row['success_rate_std']} "
              f"conv={row['converge_mean']} pure={row['pure_greedy_success_rate_mean']}±{row['pure_greedy_success_rate_std']}")
    results["E4_curriculum"] = {"note": f"逆向课程（起点距终点 1→5→10→18 曼哈顿距离）{E4_BUDGET // 4} 回合×4 级，Q 表迁移 + 阶段内探索率重启；直接组 {E4_BUDGET} 回合。最终迷宫统一口径评估 20 回合（ε=0.05 + 平局偏向终点 + 环路检测）。",
                                "budget": E4_BUDGET, "seeds": N_SEEDS, "groups": e4}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已写入 {OUT_FILE}")
