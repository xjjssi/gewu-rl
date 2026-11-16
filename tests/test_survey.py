"""test_survey.py —— 问卷统计脚本：SUS 标准算法与前后测计分（使用构造的样例数据，非真实问卷）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from analyze_survey import analyze, sus_score  # noqa: E402


def _row(**kw):
    base = {"id": "1", "pre_q1": "A", "pre_q2": "A", "pre_q3": "A", "post_q1": "C", "post_q2": "B", "post_q3": "B",
            "d1": "4", "d2": "5", "d3": "3", **{f"sus{i}": "3" for i in range(1, 11)}}
    base.update(kw)
    return base


def test_sus_extremes_and_neutral():
    best = {f"sus{i}": ("5" if i % 2 else "1") for i in range(1, 11)}
    worst = {f"sus{i}": ("1" if i % 2 else "5") for i in range(1, 11)}
    assert sus_score(best) == 100
    assert sus_score(worst) == 0
    assert sus_score(_row()) == 50


def test_pre_post_gain_and_dims():
    r = analyze([_row(), _row(id="2", post_q3="A")])
    assert r["pre_mean"] == 0 and r["post_mean"] == 2.5 and r["gain"] == 2.5
    assert r["per_question"]["q3"] == (0.0, 0.5)
    assert r["dims"]["d2"] == 5


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        sus_score(_row(sus1="7"))
