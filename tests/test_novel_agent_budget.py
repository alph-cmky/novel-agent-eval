from novel_agent_eval.agents.novel_agent import NovelAgentAdapter


def test_max_rounds_is_forwarded_to_initial_state():
    adapter = NovelAgentAdapter(max_rounds=1, label="novel_agent_r1")
    state = adapter._map_initial_state(_case(), "/tmp/eval")
    assert state["evolution_max_rounds"] == 1
    assert adapter.name == "novel_agent_r1"


def test_longform_case_uses_explicit_chapter_number():
    case = _case()
    case.name = "eqbench_p1_ch07"
    assert NovelAgentAdapter._chapter_number(case) == 7


def test_story_session_fields_are_forwarded():
    case = _case()
    case.project_id = "story_1"
    case.persist_dir = "/tmp/story_1"
    state = NovelAgentAdapter()._map_initial_state(case, "/tmp/fallback")
    assert state["project_id"] == "story_1"
    assert state["persist_dir"] == "/tmp/story_1"


def _case():
    from novel_agent_eval.dataset.schema import EvalCase

    return EvalCase(
        name="budget_case",
        stage="opening",
        story_outline="outline",
        previous_context="",
        target_chapter_outline="chapter",
    )
