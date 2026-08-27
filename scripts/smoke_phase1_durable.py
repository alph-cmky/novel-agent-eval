# scripts/smoke_phase1_durable.py
"""Phase 1 真实多章 durable run：连续 vs 进程恢复 状态等价（耗真实 API）。

Run A（进程 A，continuous）：ch1-3，resume=False，每章 commit 进 V2 DB。
Run B（进程 B，resumed）：ch1-5，resume=True——ch1-3 经 V2 DB 跳过（不重生），
ch4-5 生成。验证：
1. project_id 跨实例复用（同一 Project）
2. ch1-3 resumed（build 不调用），ch4-5 completed
3. ch4 的 Canon 快照含前 3 章
4. DB recent_summary（ContextCompiler 编译）含 ch1-3 前文
5. 无 Run not found（清理顺序修复在真实 run 下成立）
"""
import asyncio
import os
import pathlib
import sys
import tempfile

_env_path = pathlib.Path(__file__).resolve().parents[1].parent / "novel-agent" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))
os.environ.setdefault("QUALITY_IS_REASONING", "true")
os.environ.setdefault("BUDGET_IS_REASONING", "true")

from novel_agent.services.context import ContextCompiler
from novel_agent.storage.manager import ProjectManager

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
        story_outline="主角林远穿越玄幻大陆，立志成为剑仙。",
        previous_context="",  # 留空 → 验证 DB Canon 经 ContextCompiler 提供前文
        target_chapter_outline=f"第{ch}章：林远的第{ch}次修炼突破。",
        word_target=600,
        project_id="story1",
    )


def _project_id(persist_dir):
    mgr = ProjectManager(pathlib.Path(persist_dir))
    return next((p["id"] for p in mgr.list_projects() if p["name"] == "story1"), None)


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 未加载", file=sys.stderr)
        sys.exit(1)
    persist = tempfile.mkdtemp(prefix="novel_phase1_")
    ckpt = pathlib.Path(persist) / "run.json"
    print(f"[smoke] persist_dir={persist} model={os.environ.get('QUALITY_MODEL')}", flush=True)

    # ── Run A：ch1-3，continuous ──
    print("\n=== Run A: ch1-3 (continuous, resume=False) ===", flush=True)
    adapter_a = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na_a")
    status_a = await run_durable(
        adapter=adapter_a, cases=[_case(i) for i in range(1, 4)],
        persist_dir=persist, checkpoint_path=ckpt, resume=False,
    )
    pid_a = _project_id(persist)
    print(f"[A] completed={status_a.completed_chapters}/{status_a.expected_chapters} "
          f"outcome={status_a.outcome} project={pid_a}", flush=True)
    for c in status_a.checkpoints:
        print(f"[A] ch{c.chapter_number} status={c.status} tokens={c.token_usage.get('total_tokens')}", flush=True)

    # ── Run B：ch1-5，resumed（新实例模拟新进程）──
    print("\n=== Run B: ch1-5 (resumed, resume=True) ===", flush=True)
    adapter_b = NovelAgentAdapter(max_rounds=0, persist_dir=persist, resume=True, label="na_b")
    status_b = await run_durable(
        adapter=adapter_b, cases=[_case(i) for i in range(1, 6)],
        persist_dir=persist, checkpoint_path=ckpt, resume=True,
    )
    pid_b = _project_id(persist)
    print(f"[B] completed={status_b.completed_chapters}/{status_b.expected_chapters} "
          f"outcome={status_b.outcome} project={pid_b}", flush=True)
    nums = {c.chapter_number: c.status for c in status_b.checkpoints}
    for c in status_b.checkpoints:
        print(f"[B] ch{c.chapter_number} status={c.status} tokens={c.token_usage.get('total_tokens')}", flush=True)

    # ── 验证 ──
    print("\n=== 验证 ===", flush=True)

    assert pid_b == pid_a, f"project 未复用: {pid_a} != {pid_b}"
    print(f"✓ project 跨实例复用: {pid_a}", flush=True)

    assert nums[1] == CHAPTER_RESUMED and nums[2] == CHAPTER_RESUMED and nums[3] == CHAPTER_RESUMED, \
        f"ch1-3 应 resumed: {nums}"
    print("✓ ch1-3 resumed（V2 DB 跳过，未重生）", flush=True)

    assert nums[4] == CHAPTER_COMPLETED and nums[5] == CHAPTER_COMPLETED, \
        f"ch4-5 应 completed: {nums}"
    print("✓ ch4-5 completed（新生成）", flush=True)

    assert status_b.completed_chapters == 5 and status_b.outcome == OUTCOME_COMPLETED
    print("✓ 连续 vs resume 状态等价: completed=5 outcome=completed", flush=True)

    # ch4 的 Canon 快照含前 3 章
    mgr = ProjectManager(pathlib.Path(persist))
    runs_ch4 = [r for r in mgr.list_writing_runs(pid_b, 4) if r["status"] == "succeeded"]
    assert runs_ch4, "ch4 应有 succeeded run"
    snap = mgr.get_canon_snapshot(runs_ch4[0]["input_snapshot_id"])
    snap_chapters = snap["payload"]["chapters"]
    assert len(snap_chapters) == 3, f"ch4 快照应含 3 章，实际 {len(snap_chapters)}"
    print(f"✓ ch4 Canon 快照含前 3 章（{len(snap_chapters)} chapters）", flush=True)

    # DB recent_summary（ContextCompiler 编译）含 ch1-3 前文
    packet = ContextCompiler(mgr).compile(pid_b, 4)
    summary = packet.recent_summary
    assert "第1章" in summary and "第2章" in summary, f"recent_summary 应含前 3 章: {summary[:120]}"
    print(f"✓ ch4 recent_summary 含前文（来自 DB Canon，非 Chroma）: {summary[:80]!r}", flush=True)

    print("\n✅ Phase 1 durable smoke 通过：连续 vs resume 状态等价 + Canon 累积真实生效")
    await _aclose()


async def _aclose():
    from novel_agent.graph.chapter import aclose_checkpointers

    await aclose_checkpointers()


if __name__ == "__main__":
    asyncio.run(main())
