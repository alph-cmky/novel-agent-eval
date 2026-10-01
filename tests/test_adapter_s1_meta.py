"""S1 adapter meta：从 agent_loop 最终 state 提取 gate skip 信号。"""

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter


def test_extract_meta_includes_s1_gate_skip_fields():
    values = {
        "evolution_history": [],
        "quality_gate_passed": True,
        "quality_gate_report": {"passed": True, "violations": []},
        "editor_skipped": True,
        "continuity_skipped": True,
        "editor_report": {"overall_score": None, "status": "skipped"},
        "continuity_report": {"overall_score": None, "status": "skipped"},
        "worldbuilding_warnings": ["0 entities extracted"],
        "revision_feedback_consumed": False,
        "human_approved": None,
        "writer_input_tokens": 10,
        "writer_output_tokens": 5,
    }
    meta = NovelAgentAdapter._extract_meta(values, elapsed=1.25)
    assert meta["orchestration"] == "discourse"
    assert meta["contract_report"] == {}
    assert meta["surface_applied"] is False
    assert meta["quality_gate_passed"] is True
    assert meta["editor_skipped"] is True
    assert meta["continuity_skipped"] is True
    assert meta["worldbuilding_warnings"] == ["0 entities extracted"]
    assert meta["revision_feedback_consumed"] is False


def test_adapter_orchestration_constant():
    assert NovelAgentAdapter().orchestration == "discourse"
