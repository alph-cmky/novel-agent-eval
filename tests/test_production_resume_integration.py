# tests/test_production_resume_integration.py
"""B-4/B-8 Production durable resume 集成测试（fake graph + 真实 V2 SQLite）。

验证跨实例 resume 恢复的是 Production durable state：
- ch1 生成 → worldbuilding canon 提案 accepted → commit → 实体/章节落 SQLite
- 新 adapter 实例（resume=True）生成 ch2：
  - 不重建 ch1（approved 章节跳过）
  - ch2 的 initial_state.context_packet 由真实 ContextCompiler 从 DB 编译：
    character_context 含 ch1 世界实体、recent_summary 含 ch1 正文
- harness checkpoint（token/hash）作为补充层随 run_baseline 语义存在，
  不充当跳过依据（durable.py 既有契约，此处聚焦 V2 DB 侧）
"""
import asyncio

from novel_agent.storage.manager import ProjectManager

import novel_agent_eval.agents.novel_agent as na_mod
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.dataset.schema import EvalCase


def _case(ch: int) -> EvalCase:
    return EvalCase(
        name=f"resume_int_ch{ch:02d}",
        stage="opening",
        story_outline="《断剑重铸》全书大纲",
        previous_context="",
        target_chapter_outline=f"第{ch}章大纲",
        word_target=300,
        project_id="resume_int",
    )


class _CapturedGraph:
    """捕获 astream_events 收到的 initial_state，供断言 DB 编译上下文。"""

    last_initial_state: dict | None = None

    def __init__(self, draft: str, wb: dict | None = None):
        self._draft = draft
        self._wb = wb or {}

    async def astream_events(self, initial_state, config, version="v2"):
        _CapturedGraph.last_initial_state = dict(initial_state)
        if False:
            yield

    async def aget_state(self, config):
        from types import SimpleNamespace

        return SimpleNamespace(
            values={
                "draft_content": self._draft,
                "worldbuilding_report": self._wb,
                "scene_plan": [],
                "scene_drafts": [],
                "context_packet": {},
            },
            next=[],
        )


_WB_REPORT = {
    "new_entities": [
        {"entity_type": "character", "name": "林远", "properties": {"身份": "外门弟子"}},
        {"entity_type": "item", "name": "断剑", "properties": {"状态": "封印"}},
    ],
    "foreshadowings": [],
    "resolved_foreshadowings": [],
    "chapter_events": ["林远捡到断剑"],
}


def _patch_build(monkeypatch, draft: str, wb: dict | None = None) -> None:
    async def _fb(initial_state=None, **kw):
        _CapturedGraph.last_initial_state = initial_state or kw.get("initial_state")
        return {
            "draft_content": draft,
            "quality_gate_passed": True,
            "quality_gate_report": {"passed": True, "violations": []},
            "editor_skipped": True,
            "continuity_skipped": True,
            "editor_report": {},
            "continuity_report": {},
            "worldbuilding_report": wb or {},
            "worldbuilding_warnings": [],
            "evolution_history": [],
            "chapter_number": (initial_state or {}).get("chapter_number", 1),
            "context_packet": (initial_state or {}).get("context_packet") or {},
            "scene_plan": [],
            "scene_drafts": [],
            "writer_input_tokens": 100,
            "writer_output_tokens": 200,
        }

    monkeypatch.setattr(na_mod, "run_agent_loop", _fb)


def test_production_resume_restores_canon_and_chapters(monkeypatch, tmp_path):
    persist = str(tmp_path)

    # ── 实例 A：生成 ch1，worldbuilding 实体经 commit 落 V2 DB ──
    _patch_build(monkeypatch, "第一章正文：林远在剑冢捡到断剑。", wb=_WB_REPORT)
    adapter_a = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na_a")
    gen1 = asyncio.run(adapter_a.generate(_case(1)))
    assert gen1.content
    assert gen1.meta["writing_run_id"]

    mgr = ProjectManager(persist)
    pid = mgr.list_projects()[0]["id"]
    ch1 = mgr.get_chapter(pid, 1)
    assert ch1 and ch1["status"] == "approved"
    assert mgr.get_entities_by_names(pid, ["林远"], entity_type="character"), "实体应已落库"

    # ── 实例 B：resume=True，新进程语义，生成 ch2 ──
    _patch_build(monkeypatch, "第二章正文。")
    adapter_b = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_b")
    gen2 = asyncio.run(adapter_b.generate(_case(2)))
    assert gen2.content == "第二章正文。"
    assert not gen2.meta.get("resumed")  # ch2 未 approved → 正常生成

    init = _CapturedGraph.last_initial_state
    assert init is not None
    packet = init["context_packet"]
    # 真实 ContextCompiler 从 V2 DB 编译出 ch1 的世界状态与前文
    assert "林远" in packet.get("character_context", "")
    assert "第1章" in packet.get("recent_summary", "")

    # ── 再 resume ch1：approved → 缓存直返，不重建 ──
    _CapturedGraph.last_initial_state = None

    async def _must_not_build(**kw):
        raise AssertionError("resume 命中 approved 章节不应构造 graph")

    monkeypatch.setattr(na_mod, "run_agent_loop", _must_not_build)
    adapter_c = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_c")
    gen1_again = asyncio.run(adapter_c.generate(_case(1)))
    assert gen1_again.meta["resumed"] is True
    assert gen1_again.content == ch1["draft_content"]
    assert gen1_again.meta["writing_run_id"] == gen1.meta["writing_run_id"]
