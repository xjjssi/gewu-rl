"""analyze_survey.py —— 课堂试用问卷统计（概念前后测 + SUS 可用性量表 + 教学功能评分）

用法：python tools/analyze_survey.py 数据.csv
CSV 表头见 tools/survey_template.csv，题目见 docs/教学/课堂试用问卷.md。只使用标准库。
"""

from __future__ import annotations

import csv
import statistics as st
import sys

ANSWERS = {"q1": "C", "q2": "B", "q3": "B"}


def sus_score(row: dict) -> float:
    """SUS 标准算法：奇数题 (得分−1)，偶数题 (5−得分)，求和 × 2.5，范围 0–100。"""
    total = 0
    for i in range(1, 11):
        v = int(row[f"sus{i}"])
        if not 1 <= v <= 5:
            raise ValueError(f"sus{i} 应为 1–5，收到 {v}（id={row.get('id')}）")
        total += (v - 1) if i % 2 == 1 else (5 - v)
    return total * 2.5


def correct(row: dict, phase: str) -> int:
    return sum(row[f"{phase}_{q}"].strip().upper() == a for q, a in ANSWERS.items())


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("没有数据行")
    pre = [correct(r, "pre") for r in rows]
    post = [correct(r, "post") for r in rows]
    sus = [sus_score(r) for r in rows]
    per_q = {q: (sum(r[f"pre_{q}"].strip().upper() == a for r in rows) / n,
                 sum(r[f"post_{q}"].strip().upper() == a for r in rows) / n) for q, a in ANSWERS.items()}
    dims = {d: st.mean(int(r[d]) for r in rows) for d in ("d1", "d2", "d3")}
    return {
        "n": n,
        "pre_mean": st.mean(pre), "post_mean": st.mean(post),
        "gain": st.mean(post) - st.mean(pre),
        "improved_ratio": sum(b > a for a, b in zip(pre, post)) / n,
        "per_question": per_q,
        "sus_mean": st.mean(sus), "sus_sd": st.stdev(sus) if n > 1 else 0.0,
        "dims": dims,
    }


def main(path: str) -> None:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
    r = analyze(rows)
    names = {"q1": "ε-greedy 概率", "q2": "价值传播方向", "q3": "PBRS 性质"}
    dim_names = {"d1": "决策解说", "d2": "价值传播动画", "d3": "人机 PK"}
    print(f"有效问卷 {r['n']} 份")
    print(f"概念测试（满分 3）：前测 {r['pre_mean']:.2f} → 后测 {r['post_mean']:.2f}，平均提升 {r['gain']:+.2f}；"
          f"{r['improved_ratio']:.0%} 的同学后测得分提高")
    for q, (a, b) in r["per_question"].items():
        print(f"  {names[q]}：正确率 {a:.0%} → {b:.0%}")
    print(f"SUS 可用性：{r['sus_mean']:.1f} ± {r['sus_sd']:.1f}（68 为行业平均水平）")
    print("教学功能评分（1–5）：" + "，".join(f"{dim_names[k]} {v:.2f}" for k, v in r["dims"].items()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
