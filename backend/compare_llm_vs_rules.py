"""compare_llm_vs_rules.py —— LLM 路径 vs 规则引擎：固定意图集上的配置差异对比

用途：为报告提供"AI 设计师相对纯规则引擎的增量"证据。
- 规则引擎路径：无条件运行（确定性）；
- LLM 路径：读取环境变量 GEWU_LLM_API_KEY（及 GEWU_LLM_BASE_URL / GEWU_LLM_MODEL），
  无 Key 时自动跳过并标注 SKIPPED，不会中断脚本。

运行：cd backend && python compare_llm_vs_rules.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gridworld_edu"))
sys.path.insert(0, str(ROOT))

from gridworld_edu.envs.presets import PRESETS  # noqa: E402
from llm_reward import _fallback_design, design_reward, resolve_llm_config  # noqa: E402

# 固定意图集：覆盖四类意图 + 多意图组合 + 否定句边界
INTENTS = [
    ("让智能体学会绕开陷阱，宁愿绕远路也不冒险", "multi_wall_8x8"),
    ("鼓励智能体收集奖励格，哪怕多花几步", "single_wall_6x6"),
    ("让智能体尽快到达终点，走最短路径，步数越少越好", "big_maze_10x10"),
    ("让智能体充分探索整个迷宫，多尝试不同路线", "empty_6x6"),
    ("绕开陷阱，同时尽快到达终点", "multi_wall_8x8"),
    ("特别快地冲到终点", "empty_6x6"),
]

FIELDS = ["step_penalty", "wall_penalty", "goal_reward", "bonus_reward",
          "trap_penalty", "trap_terminal", "potential_shaping"]


def diff_fields(a: dict, b: dict) -> list[str]:
    return [f for f in FIELDS if a.get(f) != b.get(f)]


def main() -> None:
    llm_cfg = resolve_llm_config(None)
    print(f"LLM 路径：{'启用（' + str(llm_cfg.get('model')) + '）' if llm_cfg else 'SKIPPED（未配置 GEWU_LLM_API_KEY）'}\n")

    for instruction, preset in INTENTS:
        maze = PRESETS[preset]["grid"]
        rule = _fallback_design(instruction, maze)
        print(f"意图：「{instruction}」（迷宫 {preset}）")
        print(f"  [规则引擎] {json.dumps(rule['reward_config'], ensure_ascii=False)}")
        if llm_cfg:
            llm = design_reward(instruction, maze, llm_cfg)
            tag = "LLM" if llm["source"] == "llm" else "LLM 失败→降级规则"
            print(f"  [{tag}] {json.dumps(llm['reward_config'], ensure_ascii=False)}")
            if llm["source"] == "llm":
                d = diff_fields(rule["reward_config"], llm["reward_config"])
                print(f"  差异字段：{', '.join(d) if d else '（无差异）'}")
                print(f"  LLM 设计理由：{llm['explanation'][:120]}")
        else:
            print("  [LLM] SKIPPED")
        print()


if __name__ == "__main__":
    main()
