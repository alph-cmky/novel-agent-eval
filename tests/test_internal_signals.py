# tests/test_internal_signals.py
"""internal_signals.py 测试（全 mock，喂构造的 graph state dict，不跑 LLM / graph）。

- 主用例：evolution_enabled=True 且存在 evolution_best_*_report →
  用最终选定版本报告提取，composite_score 与
  novel_agent.graph.evolution.composite_score(extract_scores(...)) 一致。
- 回退用例：无 best 报告（evolution_enabled=False 仅当前轮报告）→ 用当前轮报告计算。
- continuity_by_category：从 inconsistencies 列表按 category 计数。
"""
from novel_agent.graph.evolution import (
    EDITOR_DIMENSIONS,
    composite_score,
    extract_scores,
)

from novel_agent_eval.internal_signals import InternalSignalCollector


def _best_state():
    """evolution_enabled=True 且 best 报告已填充的完整 graph state。"""
    best_editor = {
        "overall_score": 85,
        "dimensions": {"rhythm": 80, "ai_flavor": 70, "dialogue": 90, "logic": 75, "writing": 88},
        "issues": [],
        "highlights": [],
        "verdict": "pass",
    }
    best_continuity = {
        "overall_score": 72,
        "inconsistencies": [
            {"category": "character", "severity": "major", "description": "角色动机矛盾"},
            {"category": "character", "severity": "minor", "description": "姓名拼写不一致"},
            {"category": "timeline", "severity": "critical", "description": "时间线冲突"},
        ],
        "verdict": "minor_fix",
    }
    return {
        "evolution_enabled": True,
        "evolution_round": 3,
        "evolution_termination": "converged",
        "evolution_best_editor_report": best_editor,
        "evolution_best_continuity_report": best_continuity,
        # 当前轮报告故意不同且低分：若 collector 误用当前轮，断言会失败
        "editor_report": {"overall_score": 10, "dimensions": {d: 10 for d in EDITOR_DIMENSIONS}},
        "continuity_report": {"overall_score": 10, "inconsistencies": []},
    }


def test_collect_uses_best_reports_and_matches_evolution_composite():
    """最佳版本报告被采用；composite_score 与主仓库函数从同一报告算出的值一致。"""
    state = _best_state()
    best_editor = state["evolution_best_editor_report"]
    best_continuity = state["evolution_best_continuity_report"]

    signals = InternalSignalCollector().collect(state)

    expected_scores = extract_scores(
        {"editor_report": best_editor, "continuity_report": best_continuity}
    )
    expected_composite = composite_score(expected_scores)
    assert signals.composite_score == expected_composite

    # 采用的是 best 报告而非当前轮报告（当前轮 composite 不同）
    current_composite = composite_score(extract_scores(state))
    assert expected_composite != current_composite

    assert signals.editor_overall == 85
    assert signals.continuity_overall == 72
    assert signals.editor_dimensions == best_editor["dimensions"]
    assert signals.evolution_round == 3
    assert signals.termination_reason == "converged"


def test_collect_falls_back_to_current_round_reports():
    """evolution_enabled=False、无 best 报告 → 用当前轮 editor/continuity 报告计算。"""
    state = {
        "evolution_enabled": False,
        "editor_report": {
            "overall_score": 60,
            "dimensions": {"rhythm": 50, "ai_flavor": 60, "dialogue": 70, "logic": 55, "writing": 65},
        },
        "continuity_report": {
            "overall_score": 40,
            "inconsistencies": [{"category": "worldbuilding", "severity": "minor"}],
        },
    }

    signals = InternalSignalCollector().collect(state)

    assert signals.composite_score == composite_score(extract_scores(state))
    assert signals.editor_overall == 60
    assert signals.continuity_overall == 40


def test_continuity_by_category_counts():
    """continuity_by_category 按 category 统计 inconsistencies 条数。"""
    state = _best_state()

    signals = InternalSignalCollector().collect(state)

    assert signals.continuity_by_category == {"character": 2, "timeline": 1, "worldbuilding": 0}


def test_collect_survives_partial_state():
    """空/缺字段的 state 不崩溃，数字字段退化到默认值（0 / 空 dict）。"""
    signals = InternalSignalCollector().collect({})

    assert signals.composite_score == 0.0
    assert signals.evolution_round == 0
    assert signals.termination_reason == ""
    assert signals.editor_overall == 0
    assert signals.editor_dimensions == {d: 0 for d in EDITOR_DIMENSIONS}
    assert signals.continuity_overall == 0
    assert signals.continuity_by_category == {"character": 0, "timeline": 0, "worldbuilding": 0}
