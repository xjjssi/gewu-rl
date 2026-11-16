"""main.py —— 格物RL 后端服务（FastAPI）

职责：
- REST：预设迷宫、训练启停、历史实验查询、课程学习便捷端点
- WebSocket：训练过程实时事件流（episode / trajectory / step / done / error）
- 训练在独立线程中运行，事件写入线程安全的环形缓冲，WS 协程轮询转发，
  避免训练线程与事件循环互相阻塞；stop 标志随时可中断训练。
- 每个 run 落盘到 backend/runs/，作为"一键复现"的数据基础。
"""

from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 允许从项目根直接以 `uvicorn main:app` 启动（无需 pip install -e gridworld_edu）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gridworld_edu"))

from gridworld_edu.agents.qlearning import QLearningAgent
from gridworld_edu.agents.sb3_agents import create_sb3_agent, export_greedy_policy
from gridworld_edu.envs.grid_maze import CellType, GridMazeEnv, RewardConfig
from gridworld_edu.envs.presets import PRESETS, curriculum
from gridworld_edu.evaluation import evaluate_policy, evaluate_qlearning, evaluate_sb3
from llm_reward import design_reward, detect_intents, refine_reward, resolve_llm_config

RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="格物RL 后端", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 教学项目：允许 localhost 任意端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# 请求模型
# ----------------------------------------------------------------------


class TrainRequest(BaseModel):
    session_id: str | None = None
    maze: list[list[int]]
    algo: str = "qlearning"  # qlearning | ppo | dqn
    hyperparams: dict[str, Any] = {}
    reward_config: dict[str, Any] = {}
    seed: int = 0
    label: str | None = None  # 实验标签（如"AI 方案"/"默认方案"），用于对比图例区分


class CurriculumRequest(BaseModel):
    algo: str = "qlearning"
    episodes_per_stage: int = 200
    hyperparams: dict[str, Any] = {}
    reward_config: dict[str, Any] = {}
    seed: int = 0


class RewardDesignRequest(BaseModel):
    instruction: str
    maze: list[list[int]]
    llm_config: dict[str, Any] | None = None  # {base_url, api_key, model}，仅当次请求使用，不落盘


class IntentExperimentRequest(BaseModel):
    instruction: str
    maze: list[list[int]]
    algo: str = "qlearning"
    episodes: int = 200
    seed: int = 0          # 基准种子：实际使用 seed, seed+1, ..., seed+n_seeds-1
    n_seeds: int = 5       # 每组训练的种子数（1~8），多 seed 取均值±标准差，避免单种子偶然性
    llm_config: dict[str, Any] | None = None


# ----------------------------------------------------------------------
# 训练会话：线程 + 事件缓冲 + 停止标志
# ----------------------------------------------------------------------


class TrainSession:
    """一次训练任务。事件先进 buffer，WS 协程按偏移量消费。"""

    def __init__(self, session_id: str, req: TrainRequest):
        self.session_id = session_id
        self.req = req
        # 事件缓冲（含逐步关键帧，量级可达数万）：一次性训练内全量保留
        self.buffer: deque[dict] = deque(maxlen=200_000)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.finished = threading.Event()
        self.finished_at: float | None = None  # 训练结束时刻，供会话 GC 判断存活期
        self.active_ws = 0                     # 正在回放/订阅的 WebSocket 数，>0 时不可清理
        self.metrics_rows: list[tuple] = []  # (episode, reward, steps, success)
        self.thread = threading.Thread(target=self._run_safe, daemon=True)

    def emit(self, event: dict) -> None:
        with self.lock:
            self.buffer.append(event)

    def drain_from(self, offset: int) -> tuple[list[dict], int]:
        """取出 buffer 中 offset 之后的新事件。事件总量有限（episode + 轨迹 + 逐步
        关键帧合计通常数千至数万条，上限 20 万），直接转 list 消费即可。"""
        with self.lock:
            items = list(self.buffer)
        if offset >= len(items):
            return [], offset
        return items[offset:], len(items)

    def start(self) -> None:
        self.thread.start()

    def _run_safe(self) -> None:
        try:
            run_training(self)
        except Exception as exc:  # noqa: BLE001 - 任何训练错误都要回传给前端
            self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.finished_at = time.time()
            self.finished.set()


SESSIONS: dict[str, TrainSession] = {}
SESSION_TTL_SEC = 30 * 60  # 训练结束 30 分钟后清理会话对象，防止内存只增不减


def _gc_sessions() -> None:
    """惰性清理：已结束超过 SESSION_TTL_SEC 且没有 WebSocket 在回放/订阅的会话。"""
    now = time.time()
    for sid, s in list(SESSIONS.items()):
        if s.finished_at is not None and now - s.finished_at > SESSION_TTL_SEC and s.active_ws == 0:
            SESSIONS.pop(sid, None)


def _gc_loop() -> None:
    while True:
        time.sleep(300)  # 每 5 分钟扫描一次
        _gc_sessions()


threading.Thread(target=_gc_loop, daemon=True).start()

# ----------------------------------------------------------------------
# 训练主循环
# ----------------------------------------------------------------------


def _run_one_episode_qlearning(env: GridMazeEnv, agent: QLearningAgent, collect: bool):
    """跑一回合表格 Q-Learning；collect=True 时记录逐步轨迹供前端回放。"""
    obs, info = env.reset()
    s = info["state_index"]
    total, steps, success = 0.0, 0, False
    cells = [[env.start_pos[0], env.start_pos[1]]]
    decisions: list[dict] = []  # 与 cells[1:] 一一对应：第 i 条决策把智能体从 cells[i] 带到 cells[i+1]
    done = False
    while not done:
        if collect:
            # 采集回合：记录"决策解说"（探索/寻优、骰子、Q 值、概率），随机数序列与普通选动作一致
            a, why = agent.select_action_explained(s)
        else:
            a = agent.select_action(s)
        obs, r, terminated, truncated, info = env.step(a)
        ns = info["state_index"]
        if collect:
            q_old = float(agent.q_table[s, a])
            alpha = float(agent.lr)
            next_max = 0.0 if terminated else float(agent.q_table[ns].max())
        agent.update(s, a, r, ns, terminated, truncated)
        total += r
        steps += 1
        if collect:
            cells.append(list(info["pos"]))
            td_target = r + agent.gamma * next_max  # 终止时 next_max=0，TD 目标即 r
            decisions.append(_decision_record(a, why, r, q_old, float(agent.q_table[s, a]),
                                              alpha, next_max, td_target, terminated))
        s = ns
        done = terminated or truncated
        success = info["is_success"]
    agent.decay_epsilon()
    return total, steps, success, cells, decisions


def _decision_record(a: int, why: dict, r: float, q_old: float, q_new: float,
                     alpha: float, next_max: float, td_target: float, terminated: bool) -> dict:
    """压缩一条决策解说（短键名 + 保留 4 位小数），控制 WebSocket 数据量。"""
    def r4(x: float) -> float:
        return round(float(x), 4)
    return {
        "a": a,                                   # 实际动作 0上 1下 2左 3右
        "x": 1 if why["explore"] else 0,          # 1=探索 0=寻优
        "u": r4(why["u"]),                        # 掷出的随机数
        "e": r4(why["epsilon"]),                  # 当前 ε
        "q": [r4(v) for v in why["q"]],           # 决策前 Q(s,·)
        "b": why["best"],                         # Q 最大的动作（可能并列）
        "p": r4(why["prob"]),                     # 选中该动作的总概率
        "r": r4(r),                               # 即时奖励
        "q0": r4(q_old),                          # 更新前 Q(s,a)
        "q1": r4(q_new),                          # 更新后 Q(s,a)
        "al": r4(alpha),                          # 当前学习率 α
        "nm": r4(next_max),                       # max Q(s',·)（终止时为 0）
        "tt": r4(td_target),                      # TD 目标
        "t": 1 if terminated else 0,              # 是否终止
    }


def run_qlearning(session: TrainSession) -> dict:
    req = session.req
    hp = req.hyperparams
    episodes = int(hp.get("episodes", 500))
    env = GridMazeEnv(req.maze, RewardConfig.from_dict(req.reward_config))
    gamma = float(hp.get("gamma", 0.99))
    env.set_shaping_gamma(gamma)  # PBRS 塑形 γ 必须与智能体学习目标 γ 一致
    agent = QLearningAgent(
        n_states=env.n_states,
        learning_rate=float(hp.get("learning_rate", 0.1)),
        gamma=gamma,
        epsilon_start=float(hp.get("epsilon_start", 1.0)),
        epsilon_end=float(hp.get("epsilon_end", 0.05)),
        epsilon_decay_episodes=int(hp.get("epsilon_decay_episodes", episodes * 0.8)),
        seed=req.seed,
    )
    # 最多发 ~40 条代表性轨迹，防止 WS 洪泛；用向上取整保证小回合数时也不超量
    traj_every = max(1, -(-episodes // 40))
    for ep in range(1, episodes + 1):
        if session.stop_event.is_set():
            break
        collect = (ep % traj_every == 0) or ep == 1
        total, steps, success, cells, decisions = _run_one_episode_qlearning(env, agent, collect)
        session.metrics_rows.append((ep, round(total, 4), steps, int(success)))
        session.emit(
            {"type": "episode", "episode": ep, "reward": round(total, 4),
             "steps": steps, "success": success, "epsilon": round(agent.epsilon, 4)}
        )
        if collect:
            session.emit({"type": "trajectory", "cells": cells, "episode": ep,
                          "decisions": decisions, "gamma": agent.gamma})
            # 逐步关键帧（仅该回合）：前端动画用
            for pos in cells:
                session.emit({"type": "step", "pos": pos, "episode": ep})
    # 评估：统一口径（gridworld_edu.evaluation）——ε-评估（ε=0.05）+ 平局偏向终点
    # + 环路检测。纯贪心 argmax 在 Q 值近似平局时会绕圈（"训练 100% vs 评估 0%"
    # 的假矛盾），loop_rate 与 pure_greedy_success_rate 用于区分"没学会"与
    # "学会了但贪心未稳定"。
    m = evaluate_qlearning(env, agent, n=20, seed=req.seed)
    policy = agent.greedy_policy(env)
    summary = {"eval_success_rate": m["success_rate"],
               "eval_loop_rate": m["loop_rate"],
               "eval_stable_success_rate": m["stable_success_rate"],
               "eval_pure_greedy_success_rate": m["pure_greedy_success_rate"],
               "policy": policy,
               "episodes_run": len(session.metrics_rows), "stopped": session.stop_event.is_set()}
    return summary, agent


def run_sb3(session: TrainSession) -> dict:
    req = session.req
    hp = req.hyperparams
    total_timesteps = int(hp.get("total_timesteps", 10_000))
    env = GridMazeEnv(req.maze, RewardConfig.from_dict(req.reward_config))
    gamma = float(hp.get("gamma", 0.99))
    env.set_shaping_gamma(gamma)  # PBRS 塑形 γ 必须与智能体学习目标 γ 一致
    model = create_sb3_agent(
        req.algo, env, seed=req.seed,
        learning_rate=float(hp["learning_rate"]) if "learning_rate" in hp else None,
        gamma=gamma,
    )

    # 回调：把 SB3 的回合统计与逐步轨迹转成与 Q-Learning 一致的事件流
    from stable_baselines3.common.callbacks import BaseCallback

    class StreamCallback(BaseCallback):
        def __init__(self, start_pos):
            super().__init__()
            self.ep = 0              # 已完成的回合数
            self.traj_every = 25     # 每 25 回合采一条轨迹，防止 WS 洪泛
            self.start_pos = [int(start_pos[0]), int(start_pos[1])]
            # 非 None 表示当前回合正在采集轨迹（第 1 回合必采）
            self.cells: list | None = [self.start_pos]

        def _on_step(self) -> bool:
            # SB3 在 infos 里通过 Monitor 提供 episode 统计；info 同时保留 env 的 pos/is_success
            for info in self.locals.get("infos", []):
                if self.cells is not None and "pos" in info:
                    pos = [int(info["pos"][0]), int(info["pos"][1])]
                    self.cells.append(pos)
                    # 逐步关键帧（仅采集回合）：前端动画用
                    session.emit({"type": "step", "pos": pos, "episode": self.ep + 1})
                if "episode" in info:  # Monitor 在回合结束时附加 episode 统计
                    self.ep += 1
                    ep = self.ep
                    r = float(info["episode"]["r"])
                    l = int(info["episode"]["l"])
                    session.metrics_rows.append((ep, round(r, 4), l, int(info.get("is_success", False))))
                    session.emit({"type": "episode", "episode": ep, "reward": round(r, 4),
                                  "steps": l, "success": bool(info.get("is_success", False)),
                                  "epsilon": None})
                    if self.cells is not None:
                        session.emit({"type": "trajectory", "cells": self.cells, "episode": ep})
                    # 决定下一回合是否采集：第 1、26、51… 回合
                    self.cells = [self.start_pos] if ep % self.traj_every == 0 else None
            return not session.stop_event.is_set()

    from stable_baselines3.common.monitor import Monitor

    model.set_env(Monitor(env))
    model.learn(total_timesteps=total_timesteps,
                callback=StreamCallback(env.start_pos), progress_bar=False)

    # 评估：统一口径（ε-评估 + 环路检测），另跑一条贪心轨迹给前端回放
    m = evaluate_sb3(env, model, n=20, seed=req.seed)
    obs, info = env.reset()
    last_cells = [[int(env.start_pos[0]), int(env.start_pos[1])]]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        last_cells.append(list(info["pos"]))
        done = terminated or truncated
    session.emit({"type": "trajectory", "cells": last_cells, "episode": -1})
    policy = export_greedy_policy(model, env)
    summary = {"eval_success_rate": m["success_rate"],
               "eval_loop_rate": m["loop_rate"],
               "eval_stable_success_rate": m["stable_success_rate"],
               "policy": policy,
               "episodes_run": len(session.metrics_rows), "stopped": session.stop_event.is_set()}
    return summary, model


def run_training(session: TrainSession) -> None:
    req = session.req
    t0 = time.time()
    session.emit({"type": "started", "algo": req.algo, "seed": req.seed})
    if req.algo == "qlearning":
        summary, _ = run_qlearning(session)
    elif req.algo in ("ppo", "dqn"):
        summary, _ = run_sb3(session)
    else:
        raise ValueError(f"未知算法 {req.algo}")
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    session.emit({"type": "done", "summary": summary})
    _save_run(session, summary)


def _save_run(session: TrainSession, summary: dict) -> Path:
    """把一次 run 的 config / metrics / policy 落盘，作为复现基础。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{ts}_{session.req.algo}_{session.session_id[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "session_id": session.session_id,
        "maze": session.req.maze,
        "algo": session.req.algo,
        "hyperparams": session.req.hyperparams,
        "reward_config": session.req.reward_config,
        "seed": session.req.seed,
        "label": session.req.label,  # 实验标签（可空），前端图例用
        "created_at": ts,
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "steps", "success"])
        w.writerows(session.metrics_rows)
    (run_dir / "policy.json").write_text(
        json.dumps({"summary": summary}, ensure_ascii=False), encoding="utf-8")
    return run_dir


# ----------------------------------------------------------------------
# REST 端点
# ----------------------------------------------------------------------


@app.get("/api/presets")
def get_presets():
    return {
        "presets": [{"id": k, "name": v["name"], "grid": v["grid"]} for k, v in PRESETS.items()],
        "curriculum": curriculum(),
    }


@app.post("/api/train/start")
def train_start(req: TrainRequest):
    if req.algo not in ("qlearning", "ppo", "dqn"):
        raise HTTPException(400, f"未知算法 {req.algo}")
    # 校验迷宫合法性（尺寸、起终点）
    try:
        GridMazeEnv(req.maze)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _gc_sessions()  # 惰性清理已过期会话
    session_id = req.session_id or uuid.uuid4().hex[:12]
    session = TrainSession(session_id, req)
    SESSIONS[session_id] = session
    session.start()
    return {"session_id": session_id}


@app.post("/api/train/{session_id}/stop")
def train_stop(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    session.stop_event.set()
    return {"ok": True}


@app.get("/api/experiments")
def list_experiments():
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        cfg_file = d / "config.json"
        if not cfg_file.exists():
            continue
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        policy_file = d / "policy.json"
        summary = {}
        if policy_file.exists():
            summary = json.loads(policy_file.read_text(encoding="utf-8")).get("summary", {})
        runs.append({
            "id": d.name,
            "algo": cfg.get("algo"),
            "seed": cfg.get("seed"),
            "label": cfg.get("label"),
            "created_at": cfg.get("created_at"),
            "episodes_run": summary.get("episodes_run"),
            "eval_success_rate": summary.get("eval_success_rate"),
            "stopped": summary.get("stopped"),
        })
    return {"runs": runs}


# run_id 即 runs/ 下的目录名。必须严格校验格式，否则 %2e%2e（解码后为 ".."）之类的
# 编码输入会逃逸出 RUNS_DIR 或打到不存在的路径上抛 500。
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _check_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(400, f"非法的 run_id: {run_id!r}")
    return run_id


@app.get("/api/experiments/{run_id}")
def get_experiment(run_id: str):
    _check_run_id(run_id)
    d = RUNS_DIR / run_id
    if not d.exists() or not d.is_dir():
        raise HTTPException(404, "实验不存在")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    metrics = []
    with open(d / "metrics.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics.append({"episode": int(row["episode"]), "reward": float(row["reward"]),
                            "steps": int(row["steps"]), "success": int(row["success"])})
    policy = json.loads((d / "policy.json").read_text(encoding="utf-8"))
    return {"config": cfg, "metrics": metrics, "policy": policy}


@app.post("/api/compare")
def compare(run_ids: list[str]):
    """返回多条 run 的曲线，供前端叠加对比。"""
    for rid in run_ids:
        _check_run_id(rid)
    return {"runs": [get_experiment(rid) | {"id": rid} for rid in run_ids]}


@app.post("/api/train/curriculum")
def train_curriculum(req: CurriculumRequest):
    """课程学习便捷端点：逆向课程（起点由近及远）+ Q 表逐级迁移 + 阶段内探索率重启。

    同步返回（规模小，秒级完成）。前端用它画"课程学习 vs 直接上大迷宫"的对照。
    课程组：同一 Q 表依次在 4 个关卡上训练（每级 episodes_per_stage 回合，
    每级开始时探索率与学习率重启——否则到最后一级 ε 已衰减殆尽，
    无法学习从全程起点到已学会区域之间的新路段，这是课程迁移的经典陷阱）。
    对照组：直接在最终迷宫上训练同等总回合数。
    """
    if req.algo != "qlearning":
        raise HTTPException(400, "课程学习对照目前仅支持 qlearning（PPO/DQN 耗时过长）")
    hp = req.hyperparams
    rc = RewardConfig.from_dict(req.reward_config)
    gamma = float(hp.get("gamma", 0.99))
    stages = curriculum()
    per = req.episodes_per_stage
    total_eps = per * len(stages)
    final_env = GridMazeEnv(stages[-1]["grid"], rc)
    final_env.set_shaping_gamma(gamma)

    def new_agent():
        return QLearningAgent(
            n_states=final_env.n_states,
            learning_rate=float(hp.get("learning_rate", 0.1)),
            gamma=gamma,
            epsilon_start=float(hp.get("epsilon_start", 1.0)),
            epsilon_end=float(hp.get("epsilon_end", 0.05)),
            epsilon_decay_episodes=int(per * 0.8),
            seed=req.seed,
        )

    def train_on(grid, episodes, agent):
        env = GridMazeEnv(grid, rc)
        env.set_shaping_gamma(gamma)
        curve = []
        for ep in range(1, episodes + 1):
            total, steps, success, _, _ = _run_one_episode_qlearning(env, agent, False)
            curve.append({"episode": ep, "reward": round(total, 4), "success": int(success)})
        return curve

    def greedy_success(env, agent, n=20):
        # 统一评估口径（ε-评估 + 平局偏向终点 + 环路检测）；
        # 以 is_success（到达终点）为准：trap_terminal=True 时踩陷阱终止不算成功
        return evaluate_qlearning(env, agent, n=n, seed=req.seed)["success_rate"]

    # 课程组：Q 表逐级迁移，每级重启探索率与学习率
    cur_agent = new_agent()
    stages_out = []
    for st in stages:
        # 阶段内独立衰减：ε 与 α 重启
        cur_agent.epsilon = cur_agent.epsilon_start
        cur_agent.lr = cur_agent.lr_start
        curve = train_on(st["grid"], per, cur_agent)
        stages_out.append({"stage": st["stage"], "name": st["name"], "curve": curve})
    curriculum_eval = greedy_success(final_env, cur_agent)

    # 对照组：直接训练最终迷宫，同等总预算
    base_agent = new_agent()
    base_agent._eps_step = (base_agent.epsilon_start - base_agent.epsilon_end) / max(1, int(total_eps * 0.8))
    base_agent._lr_step = (base_agent.lr_start - base_agent.lr_end) / max(1, int(total_eps * 0.8))
    baseline = train_on(stages[-1]["grid"], total_eps, base_agent)
    baseline_eval = greedy_success(final_env, base_agent)

    return {
        "stages": stages_out,
        "baseline": {"name": f"直接训练：{stages[-1]['name']}", "curve": baseline},
        "eval": {"curriculum_success": curriculum_eval, "baseline_success": baseline_eval,
                 "note": "统一评估口径（ε=0.05 + 平局偏向终点 + 环路检测）在最终迷宫上评估 20 回合的成功率"},
    }


# ----------------------------------------------------------------------
# AI 奖励设计师：意图 → 奖励配置 → 对照实验 → 意图达成度验证
# ----------------------------------------------------------------------


@app.post("/api/reward-design")
def reward_design(req: RewardDesignRequest):
    """仅生成奖励方案（同步、快）。LLM 失败自动降级规则引擎，永不 500。"""
    try:
        GridMazeEnv(req.maze)  # 校验迷宫合法性
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not req.instruction.strip():
        raise HTTPException(400, "教学意图不能为空")
    return design_reward(req.instruction, req.maze, req.llm_config)


def evaluate_behavior(env: GridMazeEnv, act_fn, n: int = 20) -> dict[str, float]:
    """行为评估（成功率 / 平均步数 / 收集率 / 陷阱触发率 / 回报 / 环路率）。

    统一委托给 gridworld_edu.evaluation.evaluate_policy（含环路检测）；
    探索率与平局裁决由调用方在 act_fn 内部决定。
    """
    return evaluate_policy(env, act_fn, n=n)


def _train_sync(session: TrainSession) -> tuple[dict, Any]:
    """同步执行一次训练（不开线程），返回 (summary, 训练好的智能体)。"""
    if session.req.algo == "qlearning":
        return run_qlearning(session)
    return run_sb3(session)


def _convergence_episode(rows: list[tuple], window: int = 20, threshold: float = 0.8) -> int | None:
    """收敛回合：成功率滑动窗口（window 回合）首次达到 threshold 的回合数。

    这是奖励塑形最核心的教学指标——好的奖励设计不改变最优策略，
    但会显著加快收敛（PBRS 的经典结论），因此它比最终贪心行为更敏感。
    """
    if not rows:
        return None
    successes = [r[3] for r in rows]
    for i in range(len(successes)):
        lo = max(0, i - window + 1)
        if sum(successes[lo:i + 1]) / (i - lo + 1) >= threshold and i + 1 >= window:
            return i + 1
    return None  # 训练预算内未收敛，同样是重要信息


def _build_interpretation(
    instruction: str,
    ma: dict,
    mb: dict,
    maze: list[list[int]],
    mc: dict | None = None,
) -> str:
    """按意图规格（intent spec：目标指标 + 方向 + 阈值）自动生成中文解读。

    ma=AI 方案指标，mb=默认方案指标，mc=默认+PBRS 方案指标（可空）。
    判定逻辑：
    - 每个识别到的意图类别映射为一条规格（如避陷=陷阱触发率下降 ≥5%），逐条判定；
    - 成功率是相对默认方案的全局硬指标：提升 ≥30% 计为一条达成（避免
      "成功率 100% vs 0% 仍判部分达成"式的单一指标武断），显著下降则触发过度约束警告。
    """
    traps = sum(row.count(CellType.TRAP) for row in maze)
    bonuses = sum(row.count(CellType.BONUS) for row in maze)
    intent_names = [r["name"] for r, _ in detect_intents(instruction)]
    parts: list[str] = []
    verdicts: list[bool] = []

    def pct(x: float) -> str:
        return f"{x * 100:.0f}%"

    # 成功率与回报总览（回报采用统一标尺：各组策略都在默认奖励配置下评估，确保可比）
    mc_text = f"，默认+PBRS {pct(mc['success_rate'])}" if mc else ""
    parts.append(
        f"【总体表现】AI 方案成功率 {pct(ma['success_rate'])}（默认 {pct(mb['success_rate'])}{mc_text}），"
        f"统一标尺下平均回合回报 {ma['avg_reward_common']:+.3f}（默认 {mb['avg_reward_common']:+.3f}），"
        f"平均步数 {ma['avg_steps']:.0f}（默认 {mb['avg_steps']:.0f}）。"
        "评估口径：ε-评估（ε=0.05）+ Q 值平局偏向终点 + 环路检测。"
    )

    # 评估稳定性说明：ε-评估成功而纯贪心失败的差距 = "学会了但贪心未稳定"
    for label, mx in (("AI 方案", ma), ("默认方案", mb)):
        pg = mx.get("pure_greedy_success_rate")
        if pg is not None and mx["success_rate"] - pg >= 0.2:
            parts.append(
                f"【评估稳定性】{label} ε-评估成功率 {pct(mx['success_rate'])}，但纯贪心评估仅 "
                f"{pct(pg)}（绕圈回合占比 {pct(mx.get('pure_greedy_loop_rate', 0.0))}）——"
                "智能体已经学会（Q 值方向正确），只是若干动作的 Q 值几乎相等时纯贪心会在平局间绕圈。"
                "这是「学会了但贪心未稳定」，与「完全没学会」不同；加大训练预算或学习率衰减可消除。"
            )

    # 成功率硬指标：大幅提升必须计入判定；大幅下降给出过度约束警告
    d_success = ma["success_rate"] - mb["success_rate"]
    if d_success >= 0.3:
        parts.append(
            f"【成功率】AI 方案成功率较默认方案提升 {pct(d_success)}——"
            "奖励信号设计让智能体在训练预算内真正学会了到达终点，计入意图达成判定。"
        )
        verdicts.append(True)
    elif mb["success_rate"] - ma["success_rate"] > 0.3:
        parts.append(
            "⚠️ 【过度约束警告】AI 方案的成功率显著低于默认方案。"
            "可能原因：惩罚项过强压过了终点奖励信号（例如陷阱终止 + 必经之路有陷阱），"
            "或正奖励吸引导致智能体「流连忘返」。这也是奖励设计中的经典陷阱——"
            "惩罚必须让智能体「有路可走」。"
        )
        verdicts.append(False)

    # 收敛速度对比：奖励塑形的主要收益是学得更快，而非改变最优策略
    ca, cb = ma.get("episodes_to_converge"), mb.get("episodes_to_converge")
    if ca is not None and cb is not None:
        diff = cb - ca
        parts.append(
            f"【学习效率】AI 方案约第 {ca:.0f} 回合收敛（默认约第 {cb:.0f} 回合）"
            + (f"，提前 {diff:.0f} 回合——奖励信号设计显著加快了学习。" if diff > 0 else
               "，收敛更慢——奖励信号可能分散了智能体对终点的注意力。" if diff < 0 else
               "，两者相当。")
        )
        verdicts.append(diff > 0)
    elif ca is not None and cb is None:
        parts.append(f"【学习效率】AI 方案约第 {ca:.0f} 回合收敛，默认方案在训练预算内未收敛——奖励设计带来了实质提升。")
        verdicts.append(True)
    elif ca is None and cb is not None:
        parts.append(f"【学习效率】默认方案约第 {cb:.0f} 回合收敛，AI 方案在训练预算内未收敛——奖励信号可能过散。")
        verdicts.append(False)

    # 默认+PBRS 对照：AI 的价值体现在相对"只开 PBRS"的增量，而不是"PBRS 开 vs 关"
    if mc is not None:
        cc = mc.get("episodes_to_converge")
        if ca is not None and cc is not None:
            delta = cc - ca
            if delta > 0:
                parts.append(
                    f"【PBRS 增量】默认+PBRS 约第 {cc:.0f} 回合收敛、成功率 {pct(mc['success_rate'])}；"
                    f"AI 方案在此基础上又提前了 {delta:.0f} 回合"
                    "——这部分才是 AI 方案相对「单纯打开 PBRS」的增量价值。"
                )
            else:
                parts.append(
                    f"【PBRS 增量】默认+PBRS 约第 {cc:.0f} 回合收敛、成功率 {pct(mc['success_rate'])}，"
                    "与 AI 方案基本相当——本例中收敛加速主要来自 PBRS 本身，"
                    "AI 其余调参（步数代价/目标奖励/惩罚强度）的价值请对照行为指标判断。"
                )
        else:
            parts.append(
                f"【PBRS 增量】默认+PBRS 成功率 {pct(mc['success_rate'])}"
                f"（收敛回合：{('约第 %.0f 回合' % cc) if cc is not None else '预算内未收敛'}），"
                "请对照 AI 方案判断增量价值。"
            )

    # ---- 按意图规格逐条判定（目标指标 + 方向 + 阈值）----
    if "规避陷阱" in intent_names:
        if traps == 0:
            parts.append("当前迷宫没有陷阱格，避陷意图无从体现，建议先在编辑器中放置陷阱。")
        else:
            # 判定规格：AI 方案陷阱触发率 ≤5%（确实避开了），或较默认方案下降 >5%。
            # 注意"两组都是 0%"也是达成——默认惩罚本身已足以避开陷阱时，差异本来就不存在。
            d = mb["trap_trigger_rate"] - ma["trap_trigger_rate"]
            met = ma["trap_trigger_rate"] <= 0.05 or d > 0.05
            if mb["trap_trigger_rate"] <= 0.05 and ma["trap_trigger_rate"] <= 0.05:
                parts.append(
                    f"【避陷意图】两组陷阱触发率都约为 0%——默认惩罚（-1.0）已足以让智能体避开"
                    f"本迷宫的 {traps} 个陷阱，AI 方案同样保持了 0 踩陷。"
                    "若想演示「默认方案会踩陷阱」的场景，可缩短训练回合数或把陷阱放到更短的必经捷径上。"
                )
            else:
                parts.append(
                    f"【避陷意图】陷阱触发率从默认方案的 {pct(mb['trap_trigger_rate'])} "
                    f"变为 AI 方案的 {pct(ma['trap_trigger_rate'])}"
                    + (f"，下降了 {pct(d)}，智能体明显学会了绕行。" if d > 0.05 else
                       f"，几乎不变甚至上升——步数代价与陷阱惩罚的相对强度可能还需要调整。")
                )
            verdicts.append(met)
            parts.append(
                f"代价是平均步数从 {mb['avg_steps']:.0f} 变为 {ma['avg_steps']:.0f}"
                + ("，为安全付出了绕路成本，符合「宁愿绕远路」的意图。" if ma["avg_steps"] > mb["avg_steps"] else "。")
            )
    if "收集奖励" in intent_names:
        if bonuses == 0:
            parts.append("当前迷宫没有奖励格，收集意图无从体现，建议先在编辑器中放置奖励格。")
        else:
            d = ma["bonus_collect_rate"] - mb["bonus_collect_rate"]
            parts.append(
                f"【收集意图】奖励格收集率从默认方案的 {pct(mb['bonus_collect_rate'])} "
                f"提升至 AI 方案的 {pct(ma['bonus_collect_rate'])}"
                + (f"（+{pct(d)}），智能体更愿意为收集绕路了。" if d > 0.05 else
                   "，提升不明显——可适当再调高 bonus_reward 或降低 step_penalty。")
            )
            verdicts.append(d > 0.05)
    if "快速到达" in intent_names:
        d = mb["avg_steps"] - ma["avg_steps"]
        parts.append(
            f"【速度意图】平均步数从默认方案的 {mb['avg_steps']:.0f} 步降至 AI 方案的 {ma['avg_steps']:.0f} 步"
            + (f"（少 {d:.0f} 步），路径明显更直接。" if d > 0.5 else
               "，缩短不明显——可进一步加大 step_penalty 的绝对值。")
        )
        verdicts.append(d > 0.5)
    if "充分探索" in intent_names:
        d = ma["avg_steps"] - mb["avg_steps"]
        parts.append(
            f"【探索意图】平均步数从默认方案的 {mb['avg_steps']:.0f} 步变为 AI 方案的 {ma['avg_steps']:.0f} 步"
            + (f"（多 {d:.0f} 步），智能体更愿意走动试探。" if d > 1.0 else
               "，差异不明显——可进一步降低 step_penalty 或提高探索率。")
        )
        verdicts.append(d > 1.0)

    if verdicts:
        ok = sum(verdicts)
        if ok == len(verdicts):
            parts.append("结论：意图达成 ✅ ——生成的奖励函数成功地把行为推向了教学目标。")
        elif ok > 0:
            parts.append("结论：意图部分达成 ◐ ——部分指标朝目标方向改善，可迭代调整奖励强度。")
        else:
            parts.append("结论：意图未明显达成 ✗ ——两组行为差异不大，建议调整意图描述或训练回合数后重试。")
    else:
        parts.append("结论：未匹配到具体意图类别，以上为通用行为对比。")
    return "\n".join(parts)


def _aggregate_metrics(per_seed: list[dict]) -> dict:
    """把多个 seed 的行为指标聚合为 mean ± std。

    episodes_to_converge 只在已收敛的 seed 上取均值；全部未收敛时为 None，
    另附 converged_seeds 表示收敛的 seed 数，避免 None 被静默吞掉。
    """
    out: dict[str, Any] = {"n_seeds": len(per_seed)}
    keys = ["success_rate", "avg_steps", "avg_reward", "avg_reward_common",
            "bonus_collect_rate", "trap_trigger_rate", "avg_bonus_count", "avg_trap_count",
            "loop_rate", "stable_success_rate"]
    for k in keys:
        vals = [m[k] for m in per_seed]
        out[k] = round(float(np.mean(vals)), 3)
        out[f"{k}_std"] = round(float(np.std(vals)), 3)
    # 纯贪心对照（仅 Q-Learning 评估返回）：量化"学会了但贪心未稳定"的差距
    for k in ["pure_greedy_success_rate", "pure_greedy_loop_rate"]:
        vals = [m[k] for m in per_seed if k in m]
        if vals:
            out[k] = round(float(np.mean(vals)), 3)
            out[f"{k}_std"] = round(float(np.std(vals)), 3)
    convs = [m["episodes_to_converge"] for m in per_seed if m["episodes_to_converge"] is not None]
    out["converged_seeds"] = len(convs)
    out["episodes_to_converge"] = round(float(np.mean(convs)), 1) if convs else None
    out["episodes_to_converge_std"] = round(float(np.std(convs)), 1) if len(convs) > 1 else 0.0
    out["eval_episodes"] = per_seed[0]["eval_episodes"] if per_seed else 0
    return out


def _train_eval_group(
    req: IntentExperimentRequest,
    tag: str,
    label: str,
    rc: dict,
    seeds: list[int],
    episodes: int,
    env_common: GridMazeEnv,
) -> tuple[list[str], dict]:
    """训练一组配置（多 seed）并做行为评估，返回 (run_id 列表, 聚合指标)。

    从 intent_experiment 主体抽出，闭环迭代修正轮复用同一逻辑（只重训 AI 组）。
    """
    per_seed: list[dict] = []
    ids: list[str] = []
    for seed in seeds:
        if req.algo == "qlearning":
            hyper = {"episodes": episodes}
        else:
            hyper = {"total_timesteps": min(15_000, max(5_000, episodes * 50))}
        treq = TrainRequest(maze=req.maze, algo=req.algo, hyperparams=hyper,
                            reward_config=rc, seed=seed, label=label)
        # 随机串放在最前：_save_run 只取 session_id 前 6 位作目录名，若以 "int{tag}" 开头，
        # default 与 default_pbrs 会同名（"intdef"），同一秒内的多个种子互相覆盖
        session = TrainSession(f"{uuid.uuid4().hex[:6]}_int{tag}", treq)
        summary, agent = _train_sync(session)
        run_dir = _save_run(session, summary)
        ids.append(run_dir.name)

        env = GridMazeEnv(req.maze, RewardConfig.from_dict(rc))
        # 统一评估口径：ε-评估 + 平局偏向终点（Q-Learning）+ 环路检测
        if req.algo == "qlearning":
            m = evaluate_qlearning(env, agent, n=20, seed=seed)
        else:
            m = evaluate_sb3(env, agent, n=20, seed=seed)
        # 收敛回合：好的奖励设计主要提升学习效率，这是比最终行为更敏感的指标
        m["episodes_to_converge"] = _convergence_episode(session.metrics_rows)
        # 统一标尺回报：各组策略在同一默认奖励下评估，回报才可比（同一评估口径）
        if req.algo == "qlearning":
            m["avg_reward_common"] = evaluate_qlearning(env_common, agent, n=20, seed=seed)["avg_reward"]
        else:
            m["avg_reward_common"] = evaluate_sb3(env_common, agent, n=20, seed=seed)["avg_reward"]
        per_seed.append(m)
    return ids, _aggregate_metrics(per_seed)


def _metrics_digest(m_ai: dict, m_default: dict) -> str:
    """把 AI 组 vs 默认组的关键指标压缩成文本摘要，作为 LLM 闭环修正的反馈输入。"""
    return (
        f"- 成功率：AI 方案 {m_ai['success_rate']}（默认方案 {m_default['success_rate']}）\n"
        f"- 平均步数：AI 方案 {m_ai['avg_steps']}（默认方案 {m_default['avg_steps']}）\n"
        f"- 陷阱触发率：AI 方案 {m_ai['trap_trigger_rate']}（默认方案 {m_default['trap_trigger_rate']}）\n"
        f"- 奖励格收集率：AI 方案 {m_ai['bonus_collect_rate']}（默认方案 {m_default['bonus_collect_rate']}）\n"
        f"- 统一标尺回报：AI 方案 {m_ai['avg_reward_common']}（默认方案 {m_default['avg_reward_common']}）"
    )


# 闭环迭代修正的最大轮数（首轮之外）：每轮都要真实重训 AI 组，轮数必须受限以控制总耗时
MAX_REFINE_ROUNDS = 2


@app.post("/api/intent-experiment")
def intent_experiment(req: IntentExperimentRequest):
    """完整闭环：生成奖励 → 三组对照训练 → 行为指标对比 → 自动解读。

    三组设计（同迷宫、同算法、同种子集）：
    - 默认方案：裸奖励配置；
    - 默认+PBRS：仅打开势函数塑形的对照组——把"打开 PBRS"的效应从 AI 方案中剥离；
    - AI 方案：AI 生成的完整奖励配置。
    AI 的价值体现在第三组相对第二组的增量，而非"PBRS 开 vs 关"。

    每组跑 n_seeds 个种子（默认 5），指标汇报均值 ± 标准差。
    全部 run 落盘 runs/，可在「实验对比」页查看曲线。

    闭环迭代（仅 LLM 路径）：首轮三组对照完成后，把 AI 组指标摘要反馈给 LLM，
    由它判断意图是否达成；未达成则给出修正版配置并重训 AI 组（压缩预算：
    回合数减半、种子数减半），最多迭代 MAX_REFINE_ROUNDS 轮。规则引擎路径跳过迭代。
    """
    try:
        GridMazeEnv(req.maze)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not req.instruction.strip():
        raise HTTPException(400, "教学意图不能为空")
    if req.algo not in ("qlearning", "ppo", "dqn"):
        raise HTTPException(400, f"未知算法 {req.algo}")
    if not (1 <= req.n_seeds <= 8):
        raise HTTPException(400, "n_seeds 须在 1~8 之间")

    t0 = time.time()
    design = design_reward(req.instruction, req.maze, req.llm_config)
    seeds = [req.seed + i for i in range(req.n_seeds)]

    out: dict[str, Any] = {"design": design, "n_seeds": req.n_seeds, "seeds": seeds}
    # 统一标尺：各组策略都在默认奖励配置下评回报，组间才可比
    env_common = GridMazeEnv(req.maze, RewardConfig())
    groups = [
        ("default", "默认方案", RewardConfig().to_dict()),
        ("default_pbrs", "默认+PBRS", RewardConfig(potential_shaping=True).to_dict()),
        ("ai", "AI 方案", design["reward_config"]),
    ]
    run_ids: dict[str, list[str]] = {}
    group_metrics: dict[str, dict] = {}
    for tag, label, rc in groups:
        ids, group_metrics[tag] = _train_eval_group(
            req, tag, label, rc, seeds, req.episodes, env_common)
        run_ids[tag] = ids

    # 迭代过程记录：round 0 = 首轮设计；后续每轮 = LLM 根据指标反馈给出的修正版
    iterations: list[dict[str, Any]] = [{
        "round": 0,
        "reward_config": design["reward_config"],
        "metrics": group_metrics["ai"],
        "note": design["explanation"],
    }]

    # ---- 闭环迭代修正（指标反馈 → LLM 修正，仅 LLM 路径；规则引擎跳过）----
    llm_cfg = resolve_llm_config(req.llm_config)
    if design["source"] == "llm" and llm_cfg:
        cur_cfg = design["reward_config"]
        for rd in range(1, MAX_REFINE_ROUNDS + 1):
            try:
                rev = refine_reward(req.instruction, req.maze, llm_cfg, cur_cfg,
                                    _metrics_digest(group_metrics["ai"], group_metrics["default"]))
            except Exception:  # noqa: BLE001 - LLM 修正失败：保留当前方案，不影响主流程
                break
            if rev.get("achieved"):
                # LLM 判断意图已达成：记录结论并停止迭代（不重训，省时）
                iterations[-1]["judgement"] = rev.get("reason") or "LLM 判断意图已达成，无需修正。"
                break
            cur_cfg = rev["reward_config"]
            # 迭代轮压缩预算：回合数与种子数减半，每轮仍需真实训练以验证修正效果
            sub_seeds = seeds[: max(1, req.n_seeds // 2)]
            sub_eps = max(20, req.episodes // 2)
            ids, group_metrics["ai"] = _train_eval_group(
                req, "ai", f"AI 方案·第 {rd} 轮修正", cur_cfg, sub_seeds, sub_eps, env_common)
            run_ids["ai"].extend(ids)
            iterations.append({
                "round": rd,
                "reward_config": cur_cfg,
                "metrics": group_metrics["ai"],
                "note": rev.get("reason") or "",
            })
        design["reward_config"] = cur_cfg  # 最终生效配置以最后一轮为准

    out["groups"] = [
        {"key": tag, "label": label, "metrics": group_metrics[tag]}
        for tag, label, _ in groups
    ]
    # 兼容字段：metrics_a=AI 方案、metrics_b=默认方案
    out["metrics_a"] = group_metrics["ai"]
    out["metrics_b"] = group_metrics["default"]
    out["metrics_c"] = group_metrics["default_pbrs"]
    out["run_ids"] = run_ids
    out["iterations"] = iterations
    out["interpretation"] = _build_interpretation(
        req.instruction, group_metrics["ai"], group_metrics["default"],
        req.maze, mc=group_metrics["default_pbrs"])
    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


# ----------------------------------------------------------------------
# WebSocket：训练事件流
# ----------------------------------------------------------------------


@app.websocket("/ws/train/{session_id}")
async def ws_train(ws: WebSocket, session_id: str):
    await ws.accept()
    session = SESSIONS.get(session_id)
    if not session:
        await ws.send_json({"type": "error", "message": "会话不存在或已过期"})
        await ws.close()
        return
    import asyncio

    with session.lock:
        session.active_ws += 1  # 回放/订阅期间会话不可被 GC 清理
    offset = 0
    try:
        while True:
            events, offset = session.drain_from(offset)
            for ev in events:
                await ws.send_json(ev)
            if session.finished.is_set() and not events:
                # 训练结束且缓冲已取空：再取一次兜底后关闭
                events, offset = session.drain_from(offset)
                for ev in events:
                    await ws.send_json(ev)
                await ws.close()
                return
            await asyncio.sleep(0.05)  # 20Hz 轮询，兼顾实时性与开销
    except WebSocketDisconnect:
        return
    finally:
        with session.lock:
            session.active_ws -= 1


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
