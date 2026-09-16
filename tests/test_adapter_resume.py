# tests/test_adapter_resume.py
"""Phase 1.1 Adapter 跨进程 resume 契约测试（fake graph，不耗 API）。

验证 V2 durable state resume：
- resume=True：跨「进程」（两 adapter 实例同 persist_dir + case.project_id）复用同一 Project，
  不再每次 init_project 生成随机 id。
- resume=True + 该章已 approved：返回缓存 content，不构造 graph、不重生。
- resume=False：不复用（init_project 新建，id 不同）。

V2 durable state（Project/ChapterVersion/Canon）即真相源；eval 侧 checkpoint 只补
进程内消亡的 token/hash，不充当恢复依据（方案禁止项 6）。
"""
import asyncio

from novel_agent.storage.manager import ProjectManager

import novel_agent_eval.agents.novel_agent as na_mod
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.dataset.schema import EvalCase


def _case(ch: int, project_id: str = "story1") -> EvalCase:
    return EvalCase(
        name=f"resume_p1_ch{ch:02d}",
        stage="opening",
        story_outline="大纲",
        previous_context="前文",
        target_chapter_outline=f"第{ch}章大纲",
        word_target=300,
        project_id=project_id,
    )


class _FakeState:
    def __init__(self, values):
        self.values = values
        self.next = []


class _FakeGraph:
    """固定产出 draft；build_calls 记录 run_agent_loop 调用次数。"""

    def __init__(self, draft):
        self.values = {
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
            yield  # async generator yielding nothing

    async def aget_state(self, config):
        return _FakeState(self.values)


def _patch_build(monkeypatch, draft):
    """让 run_agent_loop 返回固定 final state；返回调用计数列表。"""
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


def test_resume_reuses_project_across_instances(monkeypatch, tmp_path):
    """两 adapter 实例（模拟两进程）同 persist_dir + case.project_id：
    resume=True 复用同一 Project id；resume=False 新建不同 id。"""
    _patch_build(monkeypatch, "第一章正文。")
    persist = str(tmp_path)

    # 进程 A：生成 ch1（resume=False，continuous run），commit 进 DB
    adapter_a = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na_a")
    gen_a = asyncio.run(adapter_a.generate(_case(1)))
    pid_a = gen_a.meta["project_id"]

    # 进程 B：resume=True，生成 ch2 → 应复用 pid_a（不是新建）
    _patch_build(monkeypatch, "第二章正文。")
    adapter_b = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_b")
    gen_b = asyncio.run(adapter_b.generate(_case(2)))
    assert gen_b.meta["project_id"] == pid_a, "resume 应复用同一 Project"

    # 进程 C：resume=False，同 project_id → init_project 新建，id 不同
    _patch_build(monkeypatch, "第一章正文。")
    adapter_c = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na_c")
    gen_c = asyncio.run(adapter_c.generate(_case(1)))
    assert gen_c.meta["project_id"] != pid_a, "非 resume 应新建 Project"


def test_resume_skips_approved_chapter_without_graph(monkeypatch, tmp_path):
    """resume=True 且该章已 approved：返缓存 content，不构造 graph（build_calls=0）。"""
    persist = str(tmp_path)
    # 预置：project "story1" + ch2 已 approved（commit_chapter_version 直接置 approved）
    manager = ProjectManager(tmp_path)
    pid = manager.init_project(name="story1", title="story1", story_length="short", target_chapter_words=300)
    run = manager.create_writing_run(pid, 2, run_type="evaluation", workflow_version="v2")
    version = manager.create_chapter_version(
        pid, 2, "缓存的第二章正文。", run_id=run["id"], origin="evaluation"
    )
    manager.commit_chapter_version(version["id"])

    build_calls = _patch_build(monkeypatch, "不应被调用")
    adapter = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_res")
    gen = asyncio.run(adapter.generate(_case(2)))

    assert gen.meta["resumed"] is True
    assert gen.content == "缓存的第二章正文。"
    assert gen.meta["project_id"] == pid
    assert gen.meta["writing_run_id"]  # 取自 writing_runs
    assert build_calls[0] == 0, "已 approved 章节不应构造 graph"


def test_resume_does_not_skip_unapproved_chapter(monkeypatch, tmp_path):
    """resume=True 但该章未 approved（仅存在 draft）→ 不跳过，正常生成。"""
    persist = str(tmp_path)
    # 预置 project 但 ch3 未提交
    manager = ProjectManager(tmp_path)
    manager.init_project(name="story1", title="story1", story_length="short", target_chapter_words=300)

    build_calls = _patch_build(monkeypatch, "新生成的第三章正文。")
    adapter = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_res2")
    gen = asyncio.run(adapter.generate(_case(3)))

    assert gen.meta.get("resumed") is not True
    assert gen.content == "新生成的第三章正文。"
    assert build_calls[0] == 1, "未 approved 章节应正常生成"
