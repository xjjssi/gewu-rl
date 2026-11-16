"""test_api.py —— 后端 REST 端点测试（fastapi TestClient，不开真实服务）。"""

import pytest
from fastapi.testclient import TestClient

from main import app
from gridworld_edu.envs.presets import PRESETS

client = TestClient(app)

WALL6 = PRESETS["single_wall_6x6"]["grid"]


def test_presets_endpoint():
    r = client.get("/api/presets")
    assert r.status_code == 200
    data = r.json()
    assert len(data["presets"]) == 4
    assert len(data["curriculum"]) == 4
    ids = {p["id"] for p in data["presets"]}
    assert {"empty_6x6", "single_wall_6x6", "multi_wall_8x8", "big_maze_10x10"} == ids


def test_experiments_list():
    r = client.get("/api/experiments")
    assert r.status_code == 200
    assert "runs" in r.json()


def test_experiment_run_id_validation():
    """F1：非法 run_id 返回 400（含 %2e%2e 编码穿越），不存在的返回 404。"""
    assert client.get("/api/experiments/%2e%2e").status_code == 400
    assert client.get("/api/experiments/...").status_code == 400  # ".." 会被 HTTP 客户端规范化，用 "..." 代替
    assert client.get("/api/experiments/bad id!").status_code == 400
    assert client.get("/api/experiments/" + "a" * 65).status_code == 400
    assert client.get("/api/experiments/no_such_run_999").status_code == 404


def test_compare_run_id_validation():
    assert client.post("/api/compare", json=["bad id!"]).status_code == 400
    assert client.post("/api/compare", json=["no_such_run_999"]).status_code == 404


def test_reward_design_rule_engine_path():
    """无 LLM Key 时走规则引擎：识别避陷意图并加深陷阱惩罚。"""
    r = client.post("/api/reward-design",
                    json={"instruction": "让智能体学会绕开陷阱，宁愿绕远路也不冒险", "maze": WALL6})
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "fallback"
    assert data["reward_config"]["trap_penalty"] <= -1.0


def test_reward_design_400_branches():
    """空意图 / 非法迷宫 → 400，而不是 500。"""
    assert client.post("/api/reward-design", json={"instruction": "  ", "maze": WALL6}).status_code == 400
    bad_maze = [[0] * 6 for _ in range(6)]  # 没有起点终点
    assert client.post("/api/reward-design", json={"instruction": "快点走", "maze": bad_maze}).status_code == 400


def test_rule_engine_negation_chinese():
    """C6：「不需要收集奖励格」是否定句，不应触发收集意图（bonus_reward 保持默认）。"""
    r = client.post("/api/reward-design",
                    json={"instruction": "不需要收集奖励格", "maze": WALL6})
    assert r.status_code == 200
    data = r.json()
    assert data["reward_config"]["bonus_reward"] == pytest.approx(0.5)
    assert "收集奖励" not in data["explanation"]


def test_rule_engine_english_intents():
    """C6：英文意图词 avoid/trap、collect、fast 应被规则引擎识别。"""
    r = client.post("/api/reward-design", json={"instruction": "avoid traps", "maze": WALL6})
    assert r.status_code == 200
    assert r.json()["reward_config"]["trap_penalty"] == pytest.approx(-1.8)

    r = client.post("/api/reward-design", json={"instruction": "collect all gems", "maze": WALL6})
    assert r.json()["reward_config"]["bonus_reward"] == pytest.approx(2.0)

    r = client.post("/api/reward-design", json={"instruction": "reach the goal fast", "maze": WALL6})
    cfg = r.json()["reward_config"]
    assert cfg["potential_shaping"] is True
    assert cfg["step_penalty"] == pytest.approx(-0.08)


def test_intent_experiment_three_groups_smoke():
    """C1+C3：意图实验返回三组（默认/默认+PBRS/AI）× 多种子的均值±标准差。"""
    r = client.post("/api/intent-experiment", json={
        "instruction": "让智能体尽快到达终点",
        "maze": PRESETS["empty_6x6"]["grid"],
        "algo": "qlearning", "episodes": 40, "seed": 7, "n_seeds": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["n_seeds"] == 2 and data["seeds"] == [7, 8]
    labels = [g["label"] for g in data["groups"]]
    assert labels == ["默认方案", "默认+PBRS", "AI 方案"]
    for g in data["groups"]:
        m = g["metrics"]
        assert m["n_seeds"] == 2
        assert 0.0 <= m["success_rate"] <= 1.0
        assert "success_rate_std" in m
    assert len(data["run_ids"]["ai"]) == 2
    assert "意图" in data["interpretation"]


def test_intent_experiment_400_branches():
    assert client.post("/api/intent-experiment", json={
        "instruction": "  ", "maze": WALL6}).status_code == 400
    assert client.post("/api/intent-experiment", json={
        "instruction": "快点走", "maze": WALL6, "n_seeds": 99}).status_code == 400
    assert client.post("/api/intent-experiment", json={
        "instruction": "快点走", "maze": WALL6, "algo": "dpo"}).status_code == 400


# ---------------------------------------------------------------------------
# 人机 PK（方案二）
# ---------------------------------------------------------------------------

def test_pk_prepare_snapshots_match_live_training():
    """三档对手按训练进度排列；"训练完成"档与训练实时视图的智能体 Q 表逐位一致。"""
    import numpy as np
    import main

    hp = {"episodes": 120}
    r = client.post("/api/pk/prepare", json={"maze": WALL6, "hyperparams": hp, "seed": 1})
    assert r.status_code == 200
    data = r.json()
    ops = data["opponents"]
    assert [o["label"] for o in ops] == ["新手 RL", "进阶 RL", "训练完成的 RL"]
    assert [o["episode"] for o in ops] == [6, 36, 120]
    assert ops[0]["epsilon"] > ops[1]["epsilon"] > ops[2]["epsilon"]
    assert data["bonus_cells"] == [[1, 4]]

    session = main.TrainSession("t-pk", main.TrainRequest(maze=WALL6, hyperparams=hp, seed=1))
    _, agent = main.run_qlearning(session)
    assert np.allclose(np.array(ops[2]["q_table"]), np.round(agent.q_table, 4))


def test_pk_prepare_rejects_invalid_maze():
    r = client.post("/api/pk/prepare", json={"maze": [[0, 0], [0, 0]]})
    assert r.status_code == 400
