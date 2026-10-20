"""report_figures.py —— 由 report_results.json / sb3_results.json 重新生成技术报告插图。

保证图与落盘数据严格一致（不重画旧数据）。输出到 backend/results/report/，
并同步复制到报告构建目录 <workspace>/report_figs/（build_report_pdf.py 从那里读图）。

用法：python report_figures.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "report"
WORKSPACE_FIGS = HERE.parent.parent / "report_figs"  # <workspace>/report_figs

for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

rep = json.loads((DATA / "report_results.json").read_text(encoding="utf-8"))
sb3 = json.loads((DATA / "sb3_results.json").read_text(encoding="utf-8"))

C_QL, C_RAND, C_PBRS, C_NOPBRS, C_AI, C_DEF = "#2563eb", "#9ca3af", "#16a34a", "#dc2626", "#7c3aed", "#d97706"


def save(fig, name):
    p1 = DATA / name
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    WORKSPACE_FIGS.mkdir(exist_ok=True)
    shutil.copy2(p1, WORKSPACE_FIGS / name)
    print(f"  {name} -> {p1} + {WORKSPACE_FIGS / name}")


# ---------- 图 5-1：E1 算法有效性 ----------
e1 = rep["E1_baselines"]
rows = e1["rows"]
envs = [r for r in rows if r["algo"] == "qlearning"]
rands = {r["env"]: r for r in rows if r["algo"] == "random"}
fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(len(envs))
ql = [r["success_mean"] * 100 for r in envs]
rd = [rands[r["env"]]["success_mean"] * 100 for r in envs]
b1 = ax.bar(x - 0.18, ql, 0.36, label=f"Q-Learning（{e1['seeds']} 种子，统一口径）", color=C_QL)
b2 = ax.bar(x + 0.18, rd, 0.36, label="随机策略基线", color=C_RAND)
ax.bar_label(b1, fmt="%.0f%%")
ax.bar_label(b2, fmt="%.0f%%")
ax.set_xticks(x, [r["name"] for r in envs])
ax.set_ylabel("评估成功率（%）")
ax.set_ylim(0, 115)
ax.set_title(f"图 5-1  算法有效性（{e1['episodes']} 回合预算 × {e1['seeds']} 种子）")
ax.legend()
ax.grid(axis="y", alpha=0.3)
save(fig, "fig_e1_baselines.png")

# ---------- 图 5-2：E2 消融 / E3 意图 / E4 课程 ----------
e2 = {r["potential_shaping"]: r for r in rep["E2_pbrs_ablation"]["rows"]}
e3 = {g["group"]: g for g in rep["E3_intent_experiment"]["groups"]}
e4 = {g["mode"]: g for g in rep["E4_curriculum"]["groups"]}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

# (a) E2：统一口径成功率 + 纯贪心对照 + 收敛回合
ax = axes[0]
labels = ["开 PBRS", "关 PBRS"]
uni = [e2[True]["success_rate_mean"] * 100, e2[False]["success_rate_mean"] * 100]
pure = [e2[True]["pure_greedy_success_rate_mean"] * 100,
        e2[False]["pure_greedy_success_rate_mean"] * 100]
pure_std = [e2[True]["pure_greedy_success_rate_std"] * 100,
            e2[False]["pure_greedy_success_rate_std"] * 100]
x = np.arange(2)
b1 = ax.bar(x - 0.18, uni, 0.36, label="统一口径（ε=0.05）", color=C_PBRS)
b2 = ax.bar(x + 0.18, pure, 0.36, yerr=pure_std, capsize=3,
            label="旧口径纯贪心", color=C_NOPBRS)
ax.bar_label(b1, fmt="%.0f%%")
ax.bar_label(b2, fmt="%.0f%%")
ax.set_xticks(x, labels)
ax.set_ylabel("评估成功率（%）")
ax.set_ylim(0, 125)
conv_txt = f"收敛回合：{e2[True]['converge_mean']} vs {e2[False]['converge_mean']}"
ax.set_title(f"图 5-2a  PBRS 消融（大迷宫 150 回合 × 20 种子）\n{conv_txt}", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# (b) E3：三组 成功率 + 收集率
ax = axes[1]
order = ["默认", "默认+PBRS", "AI生成"]
colors = [C_DEF, C_PBRS, C_AI]
succ = [e3[g]["success_rate_mean"] * 100 for g in order]
succ_std = [e3[g]["success_rate_std"] * 100 for g in order]
bonus = [e3[g]["bonus_collect_rate_mean"] * 100 for g in order]
x = np.arange(3)
b1 = ax.bar(x - 0.18, succ, 0.36, yerr=succ_std, capsize=3, label="成功率", color=colors)
b2 = ax.bar(x + 0.18, bonus, 0.36, label="奖励格收集率", color=colors, alpha=0.45, hatch="//")
for b, v in zip(b1, succ):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}%", ha="center", fontsize=8)
for b, v in zip(b2, bonus):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x, order)
ax.set_ylabel("指标（%）")
ax.set_ylim(0, 125)
ax.set_title("图 5-2b  意图实验：「收集奖励格+到达终点」\n（多障碍 8×8，150 回合 × 20 种子）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# (c) E4：课程 vs 直接
ax = axes[2]
labels = ["逆向课程", "直接训练"]
keys = ["curriculum", "direct"]
uni = [e4[k]["success_rate_mean"] * 100 for k in keys]
pure = [e4[k]["pure_greedy_success_rate_mean"] * 100 for k in keys]
pure_std = [e4[k]["pure_greedy_success_rate_std"] * 100 for k in keys]
conv = [e4[k]["converge_mean"] for k in keys]
x = np.arange(2)
b1 = ax.bar(x - 0.18, uni, 0.36, label="统一口径成功率", color=C_QL)
b2 = ax.bar(x + 0.18, pure, 0.36, yerr=pure_std, capsize=3,
            label="旧口径纯贪心", color=C_RAND)
ax.bar_label(b1, fmt="%.0f%%")
ax.bar_label(b2, fmt="%.0f%%")
for i, c in enumerate(conv):
    ax.text(i, 108, f"收敛 {c:.0f} 回合", ha="center", fontsize=8, color="#374151")
ax.set_xticks(x, labels)
ax.set_ylabel("评估成功率（%）")
ax.set_ylim(0, 125)
ax.set_title(f"图 5-2c  课程学习 vs 直接训练（总预算 {rep['E4_curriculum']['budget']} 回合 × 20 种子）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
save(fig, "fig_ablations.png")

# ---------- 图 5-3：SB3 采样效率 ----------
order = ["ppo_empty_20k", "ppo_wall_noshaping_20k", "ppo_wall_shaping_20k",
         "ppo_wall_noshaping_40k", "ppo_wall_shaping_40k", "dqn_wall_shaping_30k"]
labels = ["PPO 空场\n20k", "PPO 单障碍\n20k 无塑形", "PPO 单障碍\n20k+塑形",
          "PPO 单障碍\n40k 无塑形", "PPO 单障碍\n40k+塑形", "DQN 单障碍\n30k+塑形"]
vals = [sb3[k]["success_mean"] * 100 for k in order]
stds = [sb3[k]["success_std"] * 100 for k in order]
colors = [C_NOPBRS if "noshaping" in k else (C_PBRS if "shaping" in k else C_QL) for k in order]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
b = ax.bar(np.arange(len(order)), vals, 0.6, yerr=stds, capsize=3, color=colors)
for bi, v in zip(b, vals):
    ax.text(bi.get_x() + bi.get_width() / 2, v + 3, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_xticks(np.arange(len(order)), labels, fontsize=8)
ax.set_ylabel("评估成功率（%）")
ax.set_ylim(0, 125)
ax.set_title(f"图 5-3  深度 RL 采样效率与塑形收益（3 种子，统一口径）", fontsize=10)
ax.grid(axis="y", alpha=0.3)
save(fig, "fig_sb3.png")

print("全部图已按当前 JSON 数据重绘。")
