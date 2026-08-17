# novel_agent_eval/metrics.py
CORE_WEIGHTS = {
    "consistency": 0.25, "writing": 0.08, "ai_flavor": 0.07,
    "dialogue": 0.10, "plot": 0.10, "instruction": 0.15,
    "creativity": 0.10, "controllability": 0.10, "efficiency": 0.05,
}

STAGE_WEIGHTS = {
    "opening": {"consistency": 0.10, "writing": 0.18, "ai_flavor": 0.17,
                "dialogue": 0.12, "plot": 0.12, "instruction": 0.15,
                "creativity": 0.08, "controllability": 0.05, "efficiency": 0.03},
    "middle":  {"consistency": 0.25, "writing": 0.12, "ai_flavor": 0.10,
                "dialogue": 0.10, "plot": 0.15, "instruction": 0.15,
                "creativity": 0.05, "controllability": 0.05, "efficiency": 0.03},
    "long":    {"consistency": 0.35, "writing": 0.08, "ai_flavor": 0.08,
                "dialogue": 0.08, "plot": 0.12, "instruction": 0.15,
                "creativity": 0.05, "controllability": 0.05, "efficiency": 0.04},
}

def weighted_score(scores: dict, stage: str) -> float:
    w = STAGE_WEIGHTS[stage]
    return round(sum(scores[k] * v for k, v in w.items()), 2)


def efficiency_score(elapsed_seconds: float, tokens: int, rounds: int) -> int:
    """归一化到 0-100（当前只使用耗时和轮次，tokens 保留作接口字段）。"""
    time_score = max(0.0, min(1.0, (300 - elapsed_seconds) / 270)) * 100
    round_score = max(0, 100 - rounds * 10)
    return round(time_score * 0.7 + round_score * 0.3)
