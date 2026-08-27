# scripts/smoke_phase0_contract.py
"""Phase 0 契约 smoke run：真实 LLM 单章，确认 context_packet 链路 + token trace 不再断裂。

耗真实 API；不进 pytest/CI。验证三件事：
1. WritingRun/Snapshot/ContextPacket 链路接通（writing_run_id 非 None）——此前因
   `from novel_agent.context import ContextCompiler` 错误路径被静默禁用。
2. 真实 token 流转（token_usage.total_tokens > 0，不再 None）。
3. previous_context 抵达最终 context_packet.recent_summary（经 monkeypatch 直接观测）。
"""
import asyncio
import os
import sys
from pathlib import Path

# 1. 加载 novel-agent/.env（仅本进程）
_env_path = Path(__file__).resolve().parents[1].parent / "novel-agent" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

# step-3.7-flash 是 reasoning 模型：必须声明，否则正文被推理挤空 / stream 超时
os.environ.setdefault("QUALITY_IS_REASONING", "true")
os.environ.setdefault("BUDGET_IS_REASONING", "true")

# 2. monkeypatch _extract_meta 暴露最终 context_packet.recent_summary（仅 smoke 可观测，不改产品码）
import novel_agent_eval.agents.novel_agent as _na

_orig_extract = _na.NovelAgentAdapter._extract_meta


@staticmethod
def _patched_extract(values, elapsed):
    meta = _orig_extract(values, elapsed)
    packet = values.get("context_packet") or {}
    summary = packet.get("recent_summary", "") or ""
    meta["_debug_recent_summary_len"] = len(summary)
    meta["_debug_recent_summary_head"] = summary[:80]
    return meta


_na.NovelAgentAdapter._extract_meta = _patched_extract

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.dataset.schema import EvalCase


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 未加载（检查 novel-agent/.env）", file=sys.stderr)
        sys.exit(1)

    case = EvalCase(
        name="smoke_ch02",
        stage="opening",
        story_outline="主角林远穿越到玄幻大陆，立志成为剑仙。",
        previous_context=(
            "第一章：林远在异世界醒来，发现体内有神秘力量，"
            "被青云剑派掌门收为外门弟子，得到一柄木剑。"
        ),
        target_chapter_outline="第二章：林远开始入门修炼，第一次运转灵气时体内神秘力量躁动。",
        word_target=1500,
    )
    adapter = NovelAgentAdapter(max_rounds=0, label="novel_agent_smoke")
    print(
        f"[smoke] model={os.environ.get('QUALITY_MODEL')} "
        f"base={os.environ.get('OPENAI_BASE_URL')}",
        flush=True,
    )
    print(f"[smoke] previous_context head: {case.previous_context[:60]}", flush=True)

    gen = asyncio.run(adapter.generate(case))
    # 短生命周期进程收口 aiosqlite 非 daemon 线程，避免退出挂死
    asyncio.run(_aclose())

    meta = gen.meta
    content = gen.content.strip()
    u = meta.get("token_usage") or {}

    print("\n=== smoke 结果 ===")
    print(f"content_len={len(content)} (非空={bool(content)})")
    print(
        f"writing_run_id={meta.get('writing_run_id')} "
        f"(非None={meta.get('writing_run_id') is not None})"
    )
    print(
        f"context_packet_hash={meta.get('context_packet_hash')} "
        f"(非None={meta.get('context_packet_hash') is not None})"
    )
    print(
        f"tokens: total={u.get('total_tokens')} "
        f"orch_in={u.get('orchestrator_input_tokens')} "
        f"writer_in={u.get('writer_input_tokens')} "
        f"writer_out={u.get('writer_output_tokens')} "
        f"editor_in={u.get('editor_input_tokens')}"
    )
    print(
        f"_debug recent_summary_len={meta.get('_debug_recent_summary_len')} "
        f"head={meta.get('_debug_recent_summary_head')!r}"
    )

    assert content, "正文不应为空"
    assert meta.get("writing_run_id"), "writing_run_id 应非 None（ContextCompiler/ProjectManager 链路已接通）"
    assert meta.get("context_packet_hash"), "context_packet_hash 应非 None"
    assert (u.get("total_tokens") or 0) > 0, "token 总数应 > 0"
    assert (u.get("orchestrator_input_tokens") or 0) > 0 or (
        u.get("writer_input_tokens") or 0
    ) > 0, "orchestrator/writer input 应 > 0"
    assert (meta.get("_debug_recent_summary_len") or 0) > 0, (
        "previous_context 应抵达最终 context_packet.recent_summary"
    )
    print("\n✅ Phase 0 契约 smoke 通过：链路接通 + token 流转 + previous_context 抵达 Writer packet")


async def _aclose():
    from novel_agent.graph.chapter import aclose_checkpointers

    await aclose_checkpointers()


if __name__ == "__main__":
    main()
