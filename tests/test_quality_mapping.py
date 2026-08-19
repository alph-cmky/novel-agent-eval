import pytest

from novel_agent_eval.quality_mapping import (
    evidence_to_repair_instruction,
    mapping_for,
)


def test_constory_timeline_is_hard_constraint():
    mapping = mapping_for("constory", "timeline_plot")
    assert mapping.hard_constraint is True
    assert "timeline_consistency" in mapping.internal_dimensions


def test_eqbench_dialogue_becomes_repair_instruction():
    result = evidence_to_repair_instruction(
        "eqbench", "Weak Dialogue", evidence="角色台词缺少潜台词"
    )
    assert result["internal_dimensions"] == ["dialogue"]
    assert "角色台词缺少潜台词" in result["repair_instruction"]


def test_unknown_external_dimension_fails():
    with pytest.raises(KeyError):
        mapping_for("constory", "unknown")
