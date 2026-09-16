# tests/test_baseline_run.py
"""Phase 2 run_baseline driver 测试（fake graph + fake judge，不耗 API）。

验证：连续 3 章 durable + 逐章 Judge/ConStory 评分 → BaselineRunResult；
resume 3 章（从 V2 DB 读 committed content 评分）→ ch resumed 且有 overall。
"""
import asyncio

import novel_agent_eval.agents.novel_agent as na_mod
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.baseline import (
    CHAPTER_COMPLETED,
    CHAPTER_RESUMED,
    run_baseline,
)
from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.judge import QUALITY_DIMS, JudgeScore


def _case(ch: int) -> EvalCase:
    return EvalCase(
        name=f"eq_p1_ch{ch:02d}",
        stage="opening",
        story_outline="大纲",
        previous_context="",
        target_chapter_outline=f"第{ch}章",
        word_target=300,
        project_id="story1",
    )


class _FakeState:
    def __init__(self, values):
        self.values = values
        self.next = []


class _FakeGraph:
    def __init__(self, draft):
        self._values = {
            "draft_content": draft,
            "worldbuilding_report": {},
            "scene_plan": [],
            "scene_drafts": [],
            "context_packet": {"recent_summary": "前文", "chapter_number": 1},
            "orchestrator_input_tokens": 100,
            "writer_input_tokens": 500,
            "writer_output_tokens": 2000,
        }

    async def astream_events(self, *a, **kw):
        if False:
            yield

    async def aget_state(self, config):
        return _FakeState(self._values)


def _patch_build(monkeypatch, draft):
    async def _fb(initial_state=None, **kw):
        return {
            "draft_content": draft,
            "quality_gate_passed": True,
            "quality_gate_report": {"passed": True, "violations": []},
            "editor_skipped": True,
            "continuity_skipped": True,
            "editor_report": {},
            "continuity_report": {},
            "worldbuilding_report": {},
            "worldbuilding_warnings": [],
            "evolution_history": [],
            "chapter_number": (initial_state or {}).get("chapter_number", 1),
            "context_packet": (initial_state or {}).get("context_packet") or {},
            "writer_input_tokens": 100,
            "writer_output_tokens": 200,
            "orchestrator_input_tokens": 50,
            "orchestrator_output_tokens": 20,
        }

    monkeypatch.setattr(na_mod, "run_agent_loop", _fb)


class _FakeJudge:
    async def score(self, draft, case):
        return JudgeScore(dimensions={d: 80 for d in QUALITY_DIMS}, overall=80)


class _FakeConReport:
    def __init__(self, total, failed):
        self.total = total
        self.failed_categories = failed
        self.character = []
        self.timeline = []
        self.worldbuilding = []
        self.status = "success" if not failed else "partial"
        self.coverage = 1.0 if not failed else 0.6


class _FakeConsistency:
    def __init__(self, total=2, failed=()):
        self._total = total
        self._failed = failed

    async def check_consistency(self, content, reference=None):
        return _FakeConReport(self._total, self._failed)


def test_run_baseline_scores_chapters(monkeypatch, tmp_path):
    _patch_build(monkeypatch, "本章正文内容。")
    persist = str(tmp_path)
    ckpt = tmp_path / "run.json"
    adapter = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na")

    result = asyncio.run(run_baseline(
        adapter=adapter, judge=_FakeJudge(),
        cases=[_case(i) for i in range(1, 4)],
        persist_dir=persist, checkpoint_path=ckpt,
        consistency_checker=_FakeConsistency(total=2),
    ))

    assert result.expected_chapters == 3
    assert result.completed_chapters == 3
    assert len(result.chapters) == 3
    for cs in result.chapters:
        assert cs.status == CHAPTER_COMPLETED
        assert cs.overall is not None  # weighted_score 已算
        assert "efficiency" in cs.dimensions
        assert cs.consistency_errors == 2
        assert cs.consistency_score is not None
        assert cs.token_usage.get("total_tokens", 0) > 0
    assert result.quality_mean is not None
    assert result.ced == round(2.0, 3)  # 2 errors/chapter


def test_run_baseline_resume_scores_from_db(monkeypatch, tmp_path):
    """resume：ch1-3 从 V2 DB 读 committed content 评分（不重生）。"""
    persist = str(tmp_path)
    ckpt = tmp_path / "run.json"

    # 先连续跑 ch1-3
    _patch_build(monkeypatch, "原始正文。")
    adapter_a = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na_a")
    asyncio.run(run_baseline(
        adapter=adapter_a, judge=_FakeJudge(),
        cases=[_case(i) for i in range(1, 4)],
        persist_dir=persist, checkpoint_path=ckpt,
        consistency_checker=_FakeConsistency(total=1),
    ))

    # resume：新实例，ch1-3 应 resumed 并从 DB 评分
    build_calls = [0]

    async def _fb(**kw):
        build_calls[0] += 1
        return _FakeGraph("不应被调用")

    monkeypatch.setattr(na_mod, "run_agent_loop", _fb)
    adapter_b = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_b")
    result = asyncio.run(run_baseline(
        adapter=adapter_b, judge=_FakeJudge(),
        cases=[_case(i) for i in range(1, 4)],
        persist_dir=persist, checkpoint_path=ckpt, resume=True,
        consistency_checker=_FakeConsistency(total=1),
    ))

    assert build_calls[0] == 0, "resume 不应构造 graph"
    assert result.completed_chapters == 3
    for cs in result.chapters:
        assert cs.status == CHAPTER_RESUMED
        assert cs.overall is not None  # 从 DB content 评分
        assert cs.consistency_errors == 1
