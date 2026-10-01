# tests/test_adapter_contract.py
"""Phase 0 Adapter 契约测试（纯逻辑，不依赖 LLM / graph）。

覆盖执行方案 §0.1 / §0.2 验收点：
- previous_context 折叠进 V2 单一载体 context_packet.recent_summary
- _merge_context_packet：B-3 production 优先（DB memory > eval previous_context），
  synthetic_context 显式反转；DB 结构化 Canon 合并
- 真实 token trace：per-role + total 聚合，不再为 None
- context_packet_hash 兜底计算
- legacy 顶层字段已删除；S1 不再向 state 注入 scene_first /
  deterministic_gate_first（gate 条件化在主仓库 loop 内）
- generate() post-gen 写库在 rmtree 之前（清理顺序回归保护）
"""
import asyncio

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.dataset.schema import EvalCase


def _case(previous_context: str = "前文内容", stage: str = "opening") -> EvalCase:
    return EvalCase(
        name="contract_ch03",
        stage=stage,
        story_outline="全书大纲",
        previous_context=previous_context,
        target_chapter_outline="本章大纲",
        word_target=3000,
    )


# ── §0.1 previous_context → context_packet 契约 ──


def test_previous_context_folded_into_context_packet():
    case = _case(previous_context="第二章结尾：主角离开村庄。")
    state = NovelAgentAdapter()._map_initial_state(case, persist_dir="/tmp")

    # V2 单一载体：previous_context 只出现在 context_packet.recent_summary
    assert state["context_packet"]["recent_summary"] == "第二章结尾：主角离开村庄。"
    assert "recent_summary" not in state  # 顶层 legacy 字段不存在


def test_long_previous_context_truncation_is_synthetic_only():
    long_ctx = "前文。" * 2000  # 远超 1500 字上限
    case = _case(previous_context=long_ctx)

    # 默认（production parity）：benchmark 输入原样透传，不做 eval 侧 memory 模拟
    state = NovelAgentAdapter()._map_initial_state(case, persist_dir="/tmp")
    assert state["context_packet"]["recent_summary"] == long_ctx

    # synthetic_context=True（显式标记的 harness-context benchmark）才截断
    synth = NovelAgentAdapter(synthetic_context=True)._map_initial_state(case, persist_dir="/tmp")
    summary = synth["context_packet"]["recent_summary"]
    assert len(summary) < len(long_ctx)
    assert "[...中间章节前文已由世界观记忆库接管...]" in summary


def test_no_legacy_top_level_fields_in_state():
    state = NovelAgentAdapter()._map_initial_state(_case(), persist_dir="/tmp")
    legacy = {
        "character_context",
        "world_context",
        "recent_summary",
        "existing_world_entities",
        "retry_count",
        "writer_prompt_profile",
    }
    assert not (legacy & set(state.keys()))


def test_s1_adapter_rejects_scene_first_and_omits_legacy_gate_flags():
    # S1：条件化 Hard Gate 在主仓库 agent_loop 内；state 不再注入 scene/gate flags
    default_state = NovelAgentAdapter()._map_initial_state(_case(), persist_dir="/tmp")
    assert "scene_first" not in default_state
    assert "deterministic_gate_first" not in default_state

    adapter = NovelAgentAdapter(deterministic_gate_first=True)
    assert adapter.orchestration == "discourse"
    assert adapter.deterministic_gate_first is True
    state = adapter._map_initial_state(_case(), persist_dir="/tmp")
    assert "scene_first" not in state
    assert "deterministic_gate_first" not in state

    try:
        NovelAgentAdapter(scene_first=True)
        raise AssertionError("expected ValueError for scene_first=True")
    except ValueError as exc:
        assert "scene_first" in str(exc)


# ── §0.1 _merge_context_packet ──


def test_merge_keeps_eval_previous_context_when_db_summary_empty():
    eval_packet = NovelAgentAdapter()._eval_context_packet(_case("eval 前文"), 3)
    db_packet = {
        "project_id": "p1",
        "chapter_number": 3,
        "character_context": "- 林远: 剑修",
        "world_context": "",
        "recent_summary": "",  # DB 快照无最近章摘要
        "unresolved_foreshadowings": ["[第1章] 神秘力量"],
        "timeline_events": [],
        "timeline_findings": [],
    }
    merged = NovelAgentAdapter._merge_context_packet(eval_packet, db_packet)

    # eval 忠实前文保留；DB 结构化 Canon 合并进来
    assert merged["recent_summary"] == "eval 前文"
    assert merged["character_context"] == "- 林远: 剑修"
    assert merged["unresolved_foreshadowings"] == ["[第1章] 神秘力量"]
    assert merged["project_id"] == "p1"


def test_merge_falls_back_to_db_summary_when_eval_empty():
    eval_packet = NovelAgentAdapter()._eval_context_packet(_case(""), 1)
    db_packet = {"recent_summary": "DB 最近章摘要"}
    merged = NovelAgentAdapter._merge_context_packet(eval_packet, db_packet)
    assert merged["recent_summary"] == "DB 最近章摘要"


def test_merge_empty_db_returns_eval_packet():
    eval_packet = {"recent_summary": "eval", "chapter_number": 2}
    assert NovelAgentAdapter._merge_context_packet(eval_packet, {}) == eval_packet


def test_merge_production_prefers_db_summary_when_both_present():
    """B-3 parity：DB（Production memory）非空时优先于 eval previous_context。"""
    eval_packet = NovelAgentAdapter()._eval_context_packet(_case("eval 前文"), 3)
    db_packet = {"recent_summary": "DB 最近章摘要"}
    merged = NovelAgentAdapter._merge_context_packet(eval_packet, db_packet)
    assert merged["recent_summary"] == "DB 最近章摘要"

    # synthetic_context benchmark 显式反转优先级
    merged_synth = NovelAgentAdapter._merge_context_packet(
        eval_packet, db_packet, prefer="synthetic"
    )
    assert merged_synth["recent_summary"] == "eval 前文"


# ── §0.2 真实 token trace ──


def test_extract_token_usage_sums_all_roles():
    values = {
        "orchestrator_input_tokens": 100,
        "orchestrator_output_tokens": 50,
        "writer_input_tokens": 500,
        "writer_output_tokens": 2000,
        "writer_cached_tokens": 100,
        "writer_reasoning_tokens": 300,
        "editor_input_tokens": 400,
        "editor_output_tokens": 150,
        "editor_cached_tokens": 20,
    }
    usage = NovelAgentAdapter._extract_token_usage(values)
    # total = input + output only (cached⊂input, reasoning⊂output — no double count)
    assert usage["total_tokens"] == 100 + 50 + 500 + 2000 + 400 + 150
    assert usage["total_input_tokens"] == 100 + 500 + 400
    assert usage["total_output_tokens"] == 50 + 2000 + 150
    assert usage["cached_tokens"] == 100 + 20       # telemetry, not in total
    assert usage["reasoning_tokens"] == 300          # telemetry, not in total


def test_extract_token_usage_empty_state_yields_zeros():
    assert NovelAgentAdapter._extract_token_usage({})["total_tokens"] == 0


def test_extract_meta_tokens_no_longer_none():
    meta = NovelAgentAdapter._extract_meta({}, elapsed=0.5)
    assert meta["tokens"] == 0  # 不再是 None
    assert meta["token_usage"]["total_tokens"] == 0


# ── §0.1 context_packet_hash ──


def test_packet_hash_deterministic_and_none_for_empty():
    packet = {"recent_summary": "前文", "chapter_number": 2}
    h1 = NovelAgentAdapter._packet_hash(packet)
    h2 = NovelAgentAdapter._packet_hash(dict(packet))
    assert h1 == h2  # 同内容同 hash
    assert len(h1) == 64
    assert NovelAgentAdapter._packet_hash({}) is None
    assert NovelAgentAdapter._packet_hash(None) is None  # type: ignore[arg-type]


def test_extract_meta_records_packet_hash_when_state_lacks_it():
    # 主仓库未在 state 落 context_packet_hash → adapter 由最终 packet 兜底计算
    meta = NovelAgentAdapter._extract_meta(
        {"context_packet": {"recent_summary": "前文", "chapter_number": 2}},
        elapsed=1.0,
    )
    assert meta["context_packet_hash"]
    assert meta["context_packet_hash"] == NovelAgentAdapter._packet_hash(
        {"recent_summary": "前文", "chapter_number": 2}
    )


# ── generate() post-gen 写库 vs 清理顺序（回归保护）──


def test_generate_writes_before_cleanup(monkeypatch):
    """回归：rmtree(persist_dir) 必须在 attach_candidate/commit 之后。

    Phase 0 smoke 暴露的 bug：finally 里 cleanup()（rmtree 临时目录）在
    post-gen attach_candidate/commit（读写 novel.db）之前执行，导致
    get_writing_run 返回 None → "Run not found"。此 bug 此前因 ContextCompiler
    导入错误、ProjectManager 恒为 None 而从未触发。

    用 fake graph 跑通 generate() 的 post-gen 路径，不耗 API：断言 writing_run_id
    落库成功（证明 rmtree 推迟到写库之后）。
    """
    import novel_agent_eval.agents.novel_agent as na_mod

    draft = "本章正文内容。" * 60
    values = {
        "draft_content": draft,
        "worldbuilding_report": {},  # 空 → 跳过 canon 提案，聚焦 attach/commit 顺序
        "scene_plan": [],
        "scene_drafts": [],
        "context_packet": {"recent_summary": "前文", "chapter_number": 2},
        "orchestrator_input_tokens": 100,
        "writer_input_tokens": 500,
        "writer_output_tokens": 2000,
        "quality_gate_passed": True,
        "quality_gate_report": {"passed": True, "violations": []},
        "editor_skipped": True,
        "continuity_skipped": True,
        "worldbuilding_warnings": [],
        "evolution_history": [],
        "chapter_number": 2,
    }

    class _FakeState:
        def __init__(self):
            self.values = values
            self.next = []

    class _FakeGraph:
        async def astream_events(self, *a, **kw):
            if False:
                yield  # async generator that yields nothing

        async def aget_state(self, config):
            return _FakeState()

    async def _fake_build(initial_state=None, **kw):
        return values

    monkeypatch.setattr(na_mod, "run_agent_loop", _fake_build)

    case = EvalCase(
        name="regression_ch02",
        stage="opening",
        story_outline="大纲",
        previous_context="前文",
        target_chapter_outline="本章大纲",
        word_target=300,
    )
    adapter = NovelAgentAdapter(max_rounds=0)  # persist_dir=None → mkdtemp + cleanup

    gen = asyncio.run(adapter.generate(case))

    # post-gen 写库成功（未被过早清理）→ writing_run_id 可读
    assert gen.content == draft
    assert gen.meta["writing_run_id"]
    assert gen.meta["context_packet_hash"]
    assert gen.meta["token_usage"]["total_tokens"] == 100 + 500 + 2000
