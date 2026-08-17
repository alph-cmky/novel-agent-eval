# tests/test_metrics.py
from novel_agent_eval.metrics import STAGE_WEIGHTS, weighted_score


def test_each_stage_weights_sum_to_one():
    for stage, w in STAGE_WEIGHTS.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, stage

def test_weighted_score():
    scores = {"consistency": 85, "writing": 78, "ai_flavor": 72, "dialogue": 80,
              "plot": 75, "instruction": 90, "creativity": 70, "controllability": 65,
              "efficiency": 50}
    # opening 阶段：consistency 权重 0.10
    result = weighted_score(scores, "opening")
    expected = 85*0.10 + 78*0.18 + 72*0.17 + 80*0.12 + 75*0.12 + 90*0.15 + 70*0.08 + 65*0.05 + 50*0.03
    assert abs(result - expected) < 1e-6

def test_efficiency_score():
    from novel_agent_eval.metrics import efficiency_score
    assert efficiency_score(30, 5000, 0) == 100   # 快、零轮次 → 满分
    assert efficiency_score(300, 5000, 0) == 30   # 超时 → 时间项归零
    assert efficiency_score(30, 5000, 10) == 70   # 多轮 → 轮次项归零
