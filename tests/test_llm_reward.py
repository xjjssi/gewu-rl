"""test_llm_reward.py —— AI 奖励设计师（llm_reward.py）的单元测试

直接测试规则引擎与 LLM 解析函数，不依赖真实 API：
- C7：否定词窗口在标点处截断，不跨句抑制；
- C8：单字"别"不误伤"特别"（改为词组匹配）；
- C9：多意图按文档化规则合并（安全字段最严优先），解释与最终配置一致；
- C10：分字段截断（惩罚 ≤ 0，奖励 ≥ 0，goal_reward 必须为正）；
- C4a：LLM 输出 {"reason", "config"} 时 reason 被提取进解释；
- C4b：闭环迭代修正（mock LLM）能走通，规则引擎路径跳过迭代。
"""

import json

import pytest

import llm_reward
from llm_reward import (
    _fallback_design,
    design_reward,
    detect_intents,
    sanitize_reward_config,
)
from gridworld_edu.envs.presets import PRESETS

MAZE8 = PRESETS["multi_wall_8x8"]["grid"]  # 含陷阱
WALL6 = PRESETS["single_wall_6x6"]["grid"]


def _names(instruction: str) -> list[str]:
    return [r["name"] for r, _ in detect_intents(instruction)]


# ----------------------------------------------------------------------
# C7：否定词窗口在标点处截断
# ----------------------------------------------------------------------


def test_negation_window_stops_at_chinese_comma():
    """「不要踩陷阱，尽快到达终点」：否定词只管到"，"为止，避陷与快速都应命中。"""
    names = _names("不要踩陷阱，尽快到达终点")
    assert "规避陷阱" in names and "快速到达" in names


def test_negation_window_stops_at_punctuation_variant():
    """「别碰陷阱，快点到终点」同样应同时识别出避陷与快速。"""
    names = _names("别碰陷阱，快点到终点")
    assert "规避陷阱" in names and "快速到达" in names


def test_negation_within_same_clause_still_works():
    """同一分句内的否定仍然生效：「不需要收集奖励格」不触发收集意图。"""
    assert "收集奖励" not in _names("不需要收集奖励格")


# ----------------------------------------------------------------------
# C8："特别"中的"别"不算否定词
# ----------------------------------------------------------------------


def test_bie_in_tebie_not_negation():
    """「特别快地冲到终点」应识别出快速意图（"特别"里的"别"不误伤）。"""
    assert "快速到达" in _names("特别快地冲到终点")
    cfg = _fallback_design("特别快地冲到终点", WALL6)["reward_config"]
    assert cfg["step_penalty"] == pytest.approx(-0.08)


def test_bie_phrase_still_negates():
    """「别踩陷阱」仍是避陷意图（词组否定不影响避陷规则自身的语义识别）。"""
    cfg = _fallback_design("别踩陷阱", MAZE8)["reward_config"]
    assert cfg["trap_penalty"] == pytest.approx(-1.8)


# ----------------------------------------------------------------------
# C9：多意图合并 + 解释与最终配置一致
# ----------------------------------------------------------------------


def test_multi_intent_merge_keeps_safety_and_speed():
    """「绕开陷阱，同时尽快到达终点」：trap_penalty=-1.8（避陷）与 step_penalty=-0.08/塑形开（快速）都生效。"""
    r = _fallback_design("绕开陷阱，同时尽快到达终点", MAZE8)
    cfg = r["reward_config"]
    assert cfg["trap_penalty"] == pytest.approx(-1.8)   # 安全字段不被后续意图覆盖
    assert cfg["step_penalty"] == pytest.approx(-0.08)  # 快速意图的时间压力生效
    assert cfg["potential_shaping"] is True             # 快速意图的塑形生效


def test_multi_intent_safety_strictest_wins():
    """强约束避陷 + 快速：trap_terminal 取逻辑或（True 不被快速意图重置为 False）。"""
    cfg = _fallback_design("绝不能踩陷阱，同时尽快到达终点", MAZE8)["reward_config"]
    assert cfg["trap_terminal"] is True
    assert cfg["step_penalty"] == pytest.approx(-0.08)


def test_explanation_consistent_with_final_config():
    """解释文字必须与最终生效配置一致：塑形开启时不允许出现"刻意关闭塑形"的矛盾描述。"""
    r = _fallback_design("绕开陷阱，同时尽快到达终点", MAZE8)
    cfg, exp = r["reward_config"], r["explanation"]
    if cfg["potential_shaping"]:
        assert "刻意关闭了势函数塑形" not in exp
    else:
        assert "开启势函数塑形" not in exp
    # 解释中宣称的数值必须与最终配置一致
    assert str(cfg["trap_penalty"]) in exp
    assert str(cfg["step_penalty"]) in exp


# ----------------------------------------------------------------------
# C10：分字段截断
# ----------------------------------------------------------------------


def test_clamp_penalty_fields_capped_at_zero():
    """惩罚类字段上限为 0：step_penalty=3、trap_penalty=2 都被钳到 0，防止"挨罚刷分"。"""
    cfg = sanitize_reward_config({"step_penalty": 3, "trap_penalty": 2, "wall_penalty": 1})
    assert cfg["step_penalty"] == pytest.approx(0.0)
    assert cfg["trap_penalty"] == pytest.approx(0.0)
    assert cfg["wall_penalty"] == pytest.approx(0.0)


def test_clamp_reward_fields_floored_and_goal_positive():
    """奖励类字段下限为 0；goal_reward=-1 被钳到最小正值 GOAL_MIN。"""
    cfg = sanitize_reward_config({"goal_reward": -1, "bonus_reward": -0.5})
    assert cfg["goal_reward"] == pytest.approx(llm_reward.GOAL_MIN)
    assert cfg["bonus_reward"] == pytest.approx(0.0)
    # 整体量级约束仍在：不超过 [-2, 5]
    cfg2 = sanitize_reward_config({"trap_penalty": -99, "goal_reward": 99})
    assert cfg2["trap_penalty"] == pytest.approx(-2.0)
    assert cfg2["goal_reward"] == pytest.approx(5.0)


# ----------------------------------------------------------------------
# C4a：LLM 同包输出设计理由
# ----------------------------------------------------------------------


def _mock_post_chat(responses: list[str]):
    """构造 _post_chat 的 mock：按调用顺序依次返回 responses。"""
    it = iter(responses)
    return lambda base_url, api_key, payload: next(it)


def test_llm_reason_extracted(monkeypatch):
    """LLM 输出 {"reason","config"} 时：source=llm，reason 进入解释，config 被清洗后生效。"""
    monkeypatch.setattr(llm_reward, "_post_chat", _mock_post_chat([
        json.dumps({"reason": "加深陷阱惩罚让智能体绕路，同时保留终点吸引力。",
                    "config": {"trap_penalty": -1.5, "step_penalty": -0.02, "goal_reward": 1.5}},
                   ensure_ascii=False)
    ]))
    r = design_reward("绕开陷阱", MAZE8, {"api_key": "fake-key", "base_url": "http://x", "model": "m"})
    assert r["source"] == "llm"
    assert "加深陷阱惩罚让智能体绕路" in r["explanation"]
    assert r["reward_config"]["trap_penalty"] == pytest.approx(-1.5)
    assert r["reward_config"]["goal_reward"] == pytest.approx(1.5)


def test_llm_bad_output_falls_back(monkeypatch):
    """LLM 输出无法解析时仍降级规则引擎，不抛异常。"""
    monkeypatch.setattr(llm_reward, "_post_chat", lambda *a: "这不是 JSON")
    r = design_reward("绕开陷阱", MAZE8, {"api_key": "fake-key"})
    assert r["source"] == "fallback"


# ----------------------------------------------------------------------
# C4b：闭环迭代修正（走 /api/intent-experiment 全流程，mock LLM）
# ----------------------------------------------------------------------


def test_intent_experiment_iterates_with_mock_llm(monkeypatch):
    """mock LLM：首轮设计 → 指标反馈 → 第 1 轮修正（重训 AI 组）→ LLM 判定达成，停止。"""
    from fastapi.testclient import TestClient
    from main import app

    monkeypatch.setattr(llm_reward, "_post_chat", _mock_post_chat([
        # 1) 首轮设计
        json.dumps({"reason": "初版方案",
                    "config": {"step_penalty": -0.02, "goal_reward": 1.0}}, ensure_ascii=False),
        # 2) 第 1 轮修正：未达成，给出修正版
        json.dumps({"achieved": False, "reason": "步数压力不足，加大每步代价。",
                    "config": {"step_penalty": -0.08, "goal_reward": 2.0,
                               "potential_shaping": True}}, ensure_ascii=False),
        # 3) 第 2 轮反馈：判定达成，停止
        json.dumps({"achieved": True, "reason": "平均步数已显著低于默认方案，意图达成。",
                    "config": {"step_penalty": -0.08, "goal_reward": 2.0,
                               "potential_shaping": True}}, ensure_ascii=False),
    ]))
    client = TestClient(app)
    r = client.post("/api/intent-experiment", json={
        "instruction": "让智能体尽快到达终点",
        "maze": PRESETS["empty_6x6"]["grid"],
        "algo": "qlearning", "episodes": 30, "seed": 3, "n_seeds": 1,
        "llm_config": {"api_key": "fake-key", "base_url": "http://x", "model": "m"},
    })
    assert r.status_code == 200
    data = r.json()
    assert data["design"]["source"] == "llm"
    its = data["iterations"]
    assert [i["round"] for i in its] == [0, 1]          # 首轮 + 一轮修正后判定达成
    assert its[1]["reward_config"]["step_penalty"] == pytest.approx(-0.08)
    assert "步数压力不足" in its[1]["note"]
    assert "意图达成" in (its[0].get("judgement", "") + its[1].get("judgement", ""))
    # AI 组最终指标来自修正轮配置
    assert data["metrics_a"] == its[1]["metrics"]
    # 修正轮重训的 run 也落盘（首轮 1 个 + 修正轮 1 个）
    assert len(data["run_ids"]["ai"]) == 2


def test_intent_experiment_rule_path_skips_iteration():
    """无 LLM Key：规则引擎路径跳过迭代，iterations 只有 round 0，行为与之前一致。"""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    r = client.post("/api/intent-experiment", json={
        "instruction": "让智能体尽快到达终点",
        "maze": PRESETS["empty_6x6"]["grid"],
        "algo": "qlearning", "episodes": 30, "seed": 3, "n_seeds": 1,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["design"]["source"] == "fallback"
    assert len(data["iterations"]) == 1
    assert data["iterations"][0]["round"] == 0
    assert len(data["run_ids"]["ai"]) == 1
