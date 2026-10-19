"""llm_reward.py —— AI 奖励设计师：自然语言意图 → 结构化 RewardConfig

两条路径：
1. LLM 路径：调用 OpenAI 兼容 chat/completions 接口（仅标准库 urllib，30s 超时），
   从输出中鲁棒提取 JSON、逐字段校验并 clamp 到安全范围；
2. 规则回退：无 API Key 或 LLM 调用失败时，基于意图关键词的规则引擎生成配置
   与模板化专业解释。任何失败都不会把异常抛给前端——演示永不受网络影响。

安全约定：api_key 只存在于单次请求的内存中，绝不写入磁盘或日志。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Any

from gridworld_edu.envs.grid_maze import CellType, RewardConfig

# ----------------------------------------------------------------------
# 安全范围：LLM 输出的所有数值都会 clamp 到这里
# ----------------------------------------------------------------------
CLAMP_LO, CLAMP_HI = -2.0, 5.0
GOAL_MIN = 0.1  # goal_reward 必须为正，否则任务没有学习目标

REWARD_FIELDS_DOC = """\
- step_penalty: 每走一步的奖励（通常为小负数，如 -0.01 ~ -0.1；负得越多越催促智能体走捷径）
- wall_penalty: 撞墙惩罚（小负数，如 -0.05 ~ -0.3）
- goal_reward: 到达终点奖励（正数，0.5 ~ 5，是整个任务的学习目标，必须为正）
- bonus_reward: 踩到奖励格的一次性奖励（0 ~ 3；想鼓励收集就调高）
- trap_penalty: 踩到陷阱惩罚（负数，-0.5 ~ -2；想让智能体极力避开陷阱就调得更负）
- trap_terminal: 踩到陷阱是否立即结束回合（true/false；true 表示"踩陷阱=失败"，最严厉的避陷信号）
- potential_shaping: 是否启用势函数奖励塑形（true/false；加速收敛且不改变最优策略）
"""

SYSTEM_PROMPT = f"""你是一名强化学习奖励设计专家，为网格迷宫环境设计奖励函数（Reward Shaping）。
环境规则：智能体每回合从起点出发，动作上下左右；到达终点回合成功结束；踩到陷阱受惩罚；
奖励格给一次性正奖励；撞墙原地不动；回合有步数上限。

可调字段（JSON 格式）：
{REWARD_FIELDS_DOC}

奖励设计的教学原则（必须遵守）：
1. 任何单项惩罚的绝对值不宜显著超过 goal_reward，否则智能体可能"宁可不动"或学到奇怪行为；
2. 想鼓励的行为用正奖励引导，想抑制的行为用负惩罚约束，两者幅度要拉开层次；
3. 当用户希望智能体"绕远路也要避开陷阱"时，把 trap_penalty 调得明显更负，并可设 trap_terminal=true，
   同时把 step_penalty 的绝对值调小（容忍绕路多走的步数）；
4. 当用户希望"尽快到达"时，加大 step_penalty 绝对值，并可开启 potential_shaping 加速收敛；
5. 势函数塑形（potential_shaping）理论上不改变最优策略（Ng et al., 1999），只改变学习速度，
   因此它适合作为"加速项"而不是"目标项"；
6. 迷宫里没有奖励格时不要把 bonus_reward 调得很高（没有意义）；没有陷阱时同理。

只输出一个 JSON 对象，不要输出任何其他文字、不要用 markdown 代码块。
字段必须全部包含。数值字段输出数字，布尔字段输出 true/false。
"""

USER_TEMPLATE = """\
当前迷宫统计：
- 尺寸：{h} × {w}（共 {n} 格，其中墙 {walls} 格）
- 陷阱格数量：{traps}，奖励格数量：{bonuses}
- 起点到终点的曼哈顿距离：{dist} 步

教师的教学目标（自然语言）：
「{instruction}」

请为这个目标设计 RewardConfig，只输出 JSON。
"""


# ----------------------------------------------------------------------
# 迷宫统计
# ----------------------------------------------------------------------


def maze_stats(maze: list[list[int]]) -> dict[str, int]:
    h = len(maze)
    w = len(maze[0]) if h else 0
    walls = traps = bonuses = 0
    start = goal = (0, 0)
    for r, row in enumerate(maze):
        for c, v in enumerate(row):
            if v == CellType.WALL:
                walls += 1
            elif v == CellType.TRAP:
                traps += 1
            elif v == CellType.BONUS:
                bonuses += 1
            elif v == CellType.START:
                start = (r, c)
            elif v == CellType.GOAL:
                goal = (r, c)
    dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return {"h": h, "w": w, "n": h * w, "walls": walls, "traps": traps,
            "bonuses": bonuses, "dist": dist}


# ----------------------------------------------------------------------
# 输出校验与 clamp
# ----------------------------------------------------------------------


def sanitize_reward_config(raw: dict[str, Any]) -> dict[str, Any]:
    """把 LLM/规则引擎的输出清洗成合法 RewardConfig 字典。"""
    cfg = RewardConfig()  # 先取默认值，逐字段覆盖
    numeric_fields = ["step_penalty", "wall_penalty", "goal_reward", "bonus_reward", "trap_penalty"]
    for f in numeric_fields:
        if f in raw:
            try:
                v = float(raw[f])
            except (TypeError, ValueError):
                continue
            v = max(CLAMP_LO, min(CLAMP_HI, v))  # clamp 到安全范围
            setattr(cfg, f, round(v, 4))
    cfg.goal_reward = max(GOAL_MIN, cfg.goal_reward)  # 目标奖励必须为正

    def _to_bool(v: Any) -> bool:
        # LLM 可能把布尔输出成字符串："false"/"0" 用 bool() 会误判为 True
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    for f in ["trap_terminal", "potential_shaping"]:
        if f in raw:
            setattr(cfg, f, _to_bool(raw[f]))
    return cfg.to_dict()


def extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中鲁棒提取 JSON：容忍 markdown 代码块和前后杂文本。"""
    if not text:
        return None
    # 1) 尝试 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    # 2) 退化：取第一个 { 到最后一个 }
    if candidate is None:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            candidate = text[i: j + 1]
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ----------------------------------------------------------------------
# LLM 调用（仅标准库）
# ----------------------------------------------------------------------


def _call_llm(instruction: str, maze: list[list[int]], llm_config: dict[str, Any]) -> dict[str, Any]:
    """调用 OpenAI 兼容接口。失败抛异常，由上层降级到 fallback。"""
    base_url = (llm_config.get("base_url") or "https://api.moonshot.cn/v1").rstrip("/")
    api_key = llm_config.get("api_key") or ""
    model = llm_config.get("model") or "moonshot-v1-8k"
    if not api_key:
        raise ValueError("未提供 api_key")

    stats = maze_stats(maze)
    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(instruction=instruction, **stats)},
        ],
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # 30s 超时
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    raw = extract_json(content)
    if raw is None:
        raise ValueError(f"LLM 输出无法解析为 JSON: {content[:200]}")
    cfg = sanitize_reward_config(raw)
    explanation = (
        f"LLM（{model}）根据您的意图「{instruction}」生成了上述奖励方案。"
        f"迷宫共 {stats['traps']} 个陷阱格、{stats['bonuses']} 个奖励格，"
        f"起点距终点 {stats['dist']} 步。请结合下方对照实验验证意图是否达成。"
    )
    return {"reward_config": cfg, "explanation": explanation, "source": "llm"}


# ----------------------------------------------------------------------
# 规则回退引擎（LLM 不可用时的确定性兜底路径）
# ----------------------------------------------------------------------

# 意图词库：每类意图 = (关键词列表, 配置调整, 解释句式)
_INTENT_RULES = [
    {
        "name": "规避陷阱",
        "keywords": ["陷阱", "危险", "绕开", "避开", "躲开", "安全", "远离", "不要踩", "别踩", "宁愿绕",
                     "trap", "danger", "hazard", "safe", "avoid"],
        "strong_keywords": ["绝不", "一旦", "立即结束", "终止", "等于失败", "零容忍",
                            "must not", "never", "zero tolerance", "at all costs"],
        # 否定词只列"明确撤销避陷意图"的表达；"避免/不要踩/avoid"本身即避陷意图，不算否定
        "negations": ["不需要", "不必", "不用", "无需", "无须", "用不着", "no need"],
        "apply": lambda cfg, stats, strong=False: cfg.update(
            trap_penalty=-1.8,
            # 仅当意图含"绝不/一旦/零容忍"等强约束词时才开启"踩陷阱即终止"：
            # 若陷阱恰在必经之路上，终止制会让任务无解（惩罚过强的典型失效模式）
            trap_terminal=strong,
            step_penalty=-0.005,   # 容忍绕远路：把每步代价减半
            wall_penalty=-0.1,
            potential_shaping=False,  # 塑形鼓励直奔终点，与"绕路避陷"目标冲突
        ),
        "explain": lambda stats, strong=False: (
            "识别到【规避陷阱】意图：将陷阱惩罚加深至 -1.8（约为终点奖励的 1.8 倍），"
            "一次踩陷阱的代价远超绕路成本；同时把每步代价减半到 -0.005，"
            "明确告诉智能体「宁愿多走几步也不要冒险」。"
            + ("并开启「踩陷阱即终止回合」——最严厉的避陷信号。"
               "注意：若迷宫的必经之路被陷阱堵死，此设置会导致任务无解。" if strong else "")
            + "刻意关闭了势函数塑形——它会持续给「直奔终点方向」的动作发糖，与绕路避陷的目标冲突。"
            + ("注意：当前迷宫没有陷阱格，该配置主要在您添加陷阱后生效。" if stats["traps"] == 0 else "")
        ),
    },
    {
        "name": "收集奖励",
        "keywords": ["奖励格", "收集", "奖励", "多拿", "捡", "拾取", "金币", "宝石", "哪怕多花",
                     "collect", "bonus", "gem", "coin", "pick up", "gather"],
        "apply": lambda cfg, stats, strong=False: cfg.update(
            bonus_reward=2.0,
            step_penalty=-0.005,  # 降低时间压力，允许为收集绕路
            potential_shaping=False,  # 直奔终点的塑形会压制收集动机
        ),
        "explain": lambda stats, strong=False: (
            "识别到【收集奖励】意图：将奖励格收益提升至 2.0（终点奖励的 2 倍），"
            "并把每步代价降到 -0.005，让「绕路去收集」在期望回报上划算。"
            "关闭势函数塑形，避免「直奔终点」的梯度信号盖过收集动机。"
            + ("注意：当前迷宫没有奖励格，建议先在编辑器中放置若干奖励格再观察效果。" if stats["bonuses"] == 0 else "")
        ),
    },
    {
        "name": "快速到达",
        "keywords": ["快", "最短", "尽快", "迅速", "步数", "速度", "hurry", "效率", "直达",
                     "fast", "quick", "shortest", "speed", "fastest", "asap"],
        "apply": lambda cfg, stats, strong=False: cfg.update(
            step_penalty=-0.08,       # 显著的时间压力
            goal_reward=2.0,
            potential_shaping=True,   # PBRS 加速收敛且不改变最优策略
            trap_terminal=False,
        ),
        "explain": lambda stats, strong=False: (
            "识别到【快速到达】意图：把每步代价加深到 -0.08（约为默认的 8 倍），"
            "每犹豫一步都在「流血」，智能体会被推向最短路径；终点奖励提升到 2.0 保持目标吸引力；"
            "开启势函数塑形——F=γΦ(s′)−Φ(s) 理论上不改变最优策略（Ng et al., 1999），"
            "但能显著加速「朝终点方向」的学习收敛。"
        ),
    },
    {
        "name": "充分探索",
        "keywords": ["探索", "好奇", "多走", "尝试", "遍历", "看看",
                     "explore", "exploration", "wander", "curious"],
        "apply": lambda cfg, stats, strong=False: cfg.update(
            step_penalty=-0.002,   # 几乎不惩罚移动
            wall_penalty=-0.01,    # 降低撞墙恐惧，敢于试探边界
            bonus_reward=1.0,
            potential_shaping=False,
        ),
        "explain": lambda stats, strong=False: (
            "识别到【充分探索】意图：把每步代价降到 -0.002、撞墙惩罚降到 -0.01，"
            "几乎消除「乱动」的成本，让智能体敢于试探每个角落；不开启塑形，"
            "避免过强的终点梯度把行为过早锁定。适合配合较高的 ε 初始值使用。"
        ),
    },
]

# 通用否定词：出现在意图词之前（窗口内）时，视为撤销该意图。
# 中文匹配子串，英文按词边界匹配（统一小写后进行）。
_DEFAULT_NEGATIONS = [
    "不需要", "不必", "不用", "无需", "无须", "不要", "别", "勿", "莫", "避免", "防止",
    "don't", "do not", "dont", "never", "not", "no ", "avoid", "without",
]
_NEG_WINDOW = 8  # 否定词与意图词之间的最大字符间隔


def _is_ascii_word(kw: str) -> bool:
    return kw.isascii()


def _iter_keyword_hits(text: str, kw: str):
    """返回关键词在文本中的全部起始下标。英文词按词边界匹配，避免快/fast 误伤 faster 之外的词根。"""
    if _is_ascii_word(kw):
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(kw.lower()) + r"(?![a-z0-9])", text):
            yield m.start()
    else:
        start = 0
        while True:
            i = text.find(kw, start)
            if i == -1:
                return
            yield i
            start = i + 1


def _negated(text: str, kw_pos: int, negations: list[str]) -> bool:
    """关键词前 _NEG_WINDOW 个字符内出现否定词则视为否定（如「不需要收集」「avoid collecting」）。"""
    window = text[max(0, kw_pos - _NEG_WINDOW): kw_pos]
    return any(neg in window for neg in negations)


def detect_intents(instruction: str) -> list[tuple[dict, bool]]:
    """从自然语言意图中识别命中的规则，返回 [(规则, 是否强约束)]。

    - 支持中英文关键词；
    - 否定句抑制：意图词前的窗口内出现否定词（不/别/无需/避免/avoid/not 等）时，
      该次命中被丢弃；一个规则的全部命中都被否定才视为未命中
      （例如「不需要收集奖励格」不会触发收集意图，「avoid traps」仍触发避陷意图，
      因为避陷规则的否定词表不含 avoid——它本身就是避陷语义）。
    """
    text = instruction.lower()
    matched: list[tuple[dict, bool]] = []
    for rule in _INTENT_RULES:
        negations = rule.get("negations", _DEFAULT_NEGATIONS)
        hit = False
        for kw in rule["keywords"]:
            for pos in _iter_keyword_hits(text, kw):
                if not _negated(text, pos, negations):
                    hit = True
                    break
            if hit:
                break
        if hit:
            strong = any(
                any(True for _ in _iter_keyword_hits(text, sk))
                for sk in rule.get("strong_keywords", [])
            )
            matched.append((rule, strong))
    return matched

_DEFAULT_EXPLAIN = (
    "未识别到特定意图关键词，采用均衡型默认配置：小步代价 + 适中撞墙惩罚 + 开启势函数塑形加速收敛。"
    "该配置是多数迷宫的安全起点；如需针对性行为，请在意图中使用「陷阱/奖励格/快速/探索」等关键词。"
)


def _fallback_design(instruction: str, maze: list[list[int]], reason: str | None = None) -> dict[str, Any]:
    """规则回退生成器：意图识别（含否定句抑制与英文词）→ 配置组合 → 模板化专业解释。"""
    stats = maze_stats(maze)
    cfg = RewardConfig().to_dict()
    matched = detect_intents(instruction)
    for rule, strong in matched:
        # 强约束词（如"绝不""零容忍"）触发更严厉的配置分支
        rule["apply"](cfg, stats, strong)

    if matched:
        explanation = "\n\n".join(r["explain"](stats, s) for r, s in matched)
        if len(matched) > 1:
            explanation = f"本条意图同时命中 {len(matched)} 类目标，配置已做叠加融合：\n\n" + explanation
    else:
        explanation = _DEFAULT_EXPLAIN
        cfg["potential_shaping"] = True

    if reason:
        explanation = f"（LLM 不可用：{reason}，已自动切换至内置规则引擎）\n\n" + explanation
    return {"reward_config": sanitize_reward_config(cfg),
            "explanation": explanation, "source": "fallback"}


# ----------------------------------------------------------------------
# 对外入口
# ----------------------------------------------------------------------


def design_reward(
    instruction: str,
    maze: list[list[int]],
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """意图 → RewardConfig。永不抛异常：LLM 失败自动降级到规则引擎。

    llm_config 解析顺序：请求体传入 > 环境变量 GEWU_LLM_*  > 规则回退。
    api_key 仅存在于本次调用的内存中，不落盘、不写日志。
    """
    cfg = dict(llm_config or {})
    if not cfg.get("api_key"):
        env_key = os.environ.get("GEWU_LLM_API_KEY")
        if env_key:
            cfg = {
                "base_url": os.environ.get("GEWU_LLM_BASE_URL", "https://api.moonshot.cn/v1"),
                "api_key": env_key,
                "model": os.environ.get("GEWU_LLM_MODEL", "moonshot-v1-8k"),
            }
    if cfg.get("api_key"):
        try:
            return _call_llm(instruction, maze, cfg)
        except Exception as exc:  # noqa: BLE001 - 任何 LLM 故障都降级
            return _fallback_design(instruction, maze, reason=f"{type(exc).__name__}")
    return _fallback_design(instruction, maze)
