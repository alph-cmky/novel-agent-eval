# tests/test_production_parity.py
"""B-2 ProductionParity 回归测试：eval adapter 默认 == production 入口默认。

权威来源：novel-agent/api/routes.py:636-648（generate 章节入口的 initial_state）
+ novel-agent/graph/state.py 的 state.get 默认值。

Production 改默认值时本测试自动失败，强制显式审视 eval 侧是否跟随。
"""
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.dataset.schema import EvalCase


def _case() -> EvalCase:
    return EvalCase(
        name="parity_ch01",
        stage="opening",
        story_outline="大纲",
        previous_context="",
        target_chapter_outline="本章大纲",
        word_target=3000,
    )


def test_entry_flags_match_routes_initial_state():
    """routes.py: scene_first=False；deterministic_gate_first 未设（falsy）。"""
    state = NovelAgentAdapter()._map_initial_state(_case(), persist_dir="/tmp")
    assert state["scene_first"] is False
    assert state["deterministic_gate_first"] is True  # Phase4 消融：默认开启


def test_review_settings_match_state_defaults():
    """state.py: review_interval 默认 1；skip_* 默认关闭。"""
    state = NovelAgentAdapter()._map_initial_state(_case(), persist_dir="/tmp")
    assert state["review_interval"] == 1
    assert "skip_orchestrator" not in state
    assert "skip_reviews" not in state
    assert "skip_worldbuilding" not in state
    assert "skip_evolution_enrichment" not in state


def test_evolution_budget_matches_state_default():
    """state.py: evolution_max_rounds 默认 5（未设时 state.get 兜底）。

    adapter 默认 max_rounds=None → 不写入 state，交给生产默认；
    只有消融显式传 max_rounds 才覆盖。
    """
    state = NovelAgentAdapter()._map_initial_state(_case(), persist_dir="/tmp")
    assert "evolution_max_rounds" not in state

    ablation = NovelAgentAdapter(max_rounds=0)._map_initial_state(_case(), persist_dir="/tmp")
    assert ablation["evolution_max_rounds"] == 0


def test_narrative_mode_passthrough_matches_production():
    """routes.py: narrative_mode 来自 project（默认 None）→ adapter 透传 case 输入。"""
    state = NovelAgentAdapter()._map_initial_state(_case(), persist_dir="/tmp")
    assert state["narrative_mode"] is None

    case = _case()
    case.narrative_mode = "unit_arc"
    state2 = NovelAgentAdapter()._map_initial_state(case, persist_dir="/tmp")
    assert state2["narrative_mode"] == "unit_arc"


def test_synthetic_context_off_by_default():
    """B-3：默认不做 eval 侧 memory 模拟；synthetic 必须显式开启。"""
    adapter = NovelAgentAdapter()
    assert adapter.synthetic_context is False
    assert NovelAgentAdapter(synthetic_context=True).synthetic_context is True
