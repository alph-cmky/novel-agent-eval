# tests/test_durable_run.py
"""Phase 1.1 连续运行 vs 进程恢复 状态等价（fake graph，不耗 API）。

验证 V2 durable state resume（§1.1 验收）：
- resume 后继续使用同一 Project（project_id 跨实例复用）
- 已完成章节不重新生成（ch1-3 resumed，build_calls 仅 ch4-5）
- Canon 快照随提交累积（ch4 的 input_snapshot 含 ch1-3）
- 连续运行与恢复运行状态语义等价（最终 completed_chapters / outcome 一致）

V2 durable state 是真相源；eval 侧 checkpoint 只补进程内消亡的 token/hash。
"""
import asyncio
import pathlib

from novel_agent.storage.manager import ProjectManager

import novel_agent_eval.agents.novel_agent as na_mod
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.durable import (
    CHAPTER_COMPLETED,
    CHAPTER_RESUMED,
    OUTCOME_COMPLETED,
    run_durable,
)


def _case(ch: int) -> EvalCase:
    return EvalCase(
        name=f"eq_p1_ch{ch:02d}",
        stage="opening",
        story_outline="大纲",
        previous_context=f"第{ch - 1}章前文" if ch > 1 else "",
        target_chapter_outline=f"第{ch}章大纲",
        word_target=300,
        project_id="story1",
    )


class _FakeState:
    def __init__(self, values):
        self.values = values
        self.next = []


class _FakeGraph:
    def __init__(self, draft):
        self._draft = draft

    async def astream_events(self, *a, **kw):
        if False:
            yield

    async def aget_state(self, config):
        return _FakeState({
            "draft_content": self._draft,
            "worldbuilding_report": {},
            "scene_plan": [],
            "scene_drafts": [],
            "context_packet": {"recent_summary": "前文", "chapter_number": 1},
            "orchestrator_input_tokens": 100,
            "writer_input_tokens": 500,
            "writer_output_tokens": 2000,
        })


def _patch_build(monkeypatch, draft):
    calls = [0]

    async def _fb(initial_state=None, **kw):
        calls[0] += 1
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
        }

    monkeypatch.setattr(na_mod, "run_agent_loop", _fb)
    return calls


def test_continuous_vs_resumed_state_equivalence(monkeypatch, tmp_path):
    persist = str(tmp_path)
    ckpt = tmp_path / "run.json"

    # Run A（进程 A，continuous）：ch1-3，resume=False
    calls_a = _patch_build(monkeypatch, "A 章正文。")
    adapter_a = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na_a")
    status_a = asyncio.run(run_durable(
        adapter=adapter_a, cases=[_case(i) for i in range(1, 4)],
        persist_dir=persist, checkpoint_path=ckpt, resume=False,
    ))
    pid_a = status_a.checkpoints[0].run_id and _project_id_of(persist, "story1")
    assert status_a.completed_chapters == 3
    assert status_a.outcome == OUTCOME_COMPLETED
    assert calls_a[0] == 3  # 3 章都生成

    # Run B（进程 B，resumed）：ch1-5，resume=True
    calls_b = _patch_build(monkeypatch, "B 章正文。")
    adapter_b = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_b")
    status_b = asyncio.run(run_durable(
        adapter=adapter_b, cases=[_case(i) for i in range(1, 6)],
        persist_dir=persist, checkpoint_path=ckpt, resume=True,
    ))

    # resume 后继续使用同一 Project
    pid_b = _project_id_of(persist, "story1")
    assert pid_b == pid_a, "resume 应复用同一 Project"

    # ch1-3 resumed（不重生），ch4-5 completed → build 仅调 2 次
    nums = {c.chapter_number: c.status for c in status_b.checkpoints}
    assert nums[1] == CHAPTER_RESUMED
    assert nums[2] == CHAPTER_RESUMED
    assert nums[3] == CHAPTER_RESUMED
    assert nums[4] == CHAPTER_COMPLETED
    assert nums[5] == CHAPTER_COMPLETED
    assert calls_b[0] == 2, "ch1-3 应跳过，仅 ch4-5 生成"

    # 连续运行与恢复运行状态语义等价
    assert status_b.completed_chapters == 5
    assert status_b.outcome == OUTCOME_COMPLETED

    # Canon 快照随提交累积：ch4 的 input_snapshot 含 ch1-3
    mgr = ProjectManager(pathlib.Path(persist))
    runs_ch4 = [r for r in mgr.list_writing_runs(pid_b, 4) if r["status"] == "succeeded"]
    assert runs_ch4, "ch4 应有 succeeded 的 run"
    snapshot = mgr.get_canon_snapshot(runs_ch4[0]["input_snapshot_id"])
    chapters_in_snapshot = snapshot["payload"]["chapters"]
    assert len(chapters_in_snapshot) == 3, "ch4 快照应含前 3 章已提交内容"


def _project_id_of(persist_dir: str, name: str) -> str | None:
    mgr = ProjectManager(pathlib.Path(persist_dir))
    return next((p["id"] for p in mgr.list_projects() if p["name"] == name), None)
