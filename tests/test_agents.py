# tests/test_agents.py
"""被测 Agent 适配器测试。

- VanillaLLMAdapter 用 mock client 测（不耗真实 API）。
- NovelAgentAdapter 的纯逻辑（_map_initial_state / _compose_chapter_outline /
  _extract_meta）单测覆盖，不依赖 LLM。
- NovelAgentAdapter.generate 需要真实跑主仓库 graph（会调 LLM），标记 slow 并在
  无 OPENAI_API_KEY 时 skip（对齐主仓库 tests/eval 的 skipif 模式）。
"""
import asyncio
import os

import pytest

from novel_agent_eval.agents.base import GeneratedChapter
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.agents.vanilla_llm import (
    DEFAULT_BASELINE_MODEL,
    VanillaLLMAdapter,
    build_vanilla_prompt,
)
from novel_agent_eval.dataset.schema import EvalCase


def _make_case(stage="opening") -> EvalCase:
    return EvalCase(
        name="test_opening_01",
        stage=stage,
        story_outline="主角穿越到玄幻大陆，立志成为剑仙。",
        previous_context="第一章：主角在异世界醒来，发现体内有神秘力量。",
        target_chapter_outline="第二章：主角拜入青云剑派，随师父学剑。",
        word_target=3000,
    )


# ── Vanilla 基线（mock client） ─────────────────────────


class _FakeCompletions:
    def __init__(self, content, usage=None):
        self._content = content
        self._usage = usage or type(
            "Usage", (), {"prompt_tokens": 10, "completion_tokens": 20}
        )()
        self.calls = 0
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        message = type("Msg", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice], "usage": self._usage})()


class _FakeClient:
    def __init__(self, content, usage=None):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content, usage)})()


def test_vanilla_generate_returns_content_and_meta():
    client = _FakeClient("第一章正文……")
    gen = asyncio.run(VanillaLLMAdapter(client).generate(_make_case()))

    assert isinstance(gen, GeneratedChapter)
    assert gen.content == "第一章正文……"
    assert gen.meta["adapter"] == "vanilla_llm"
    assert gen.meta["model"] == DEFAULT_BASELINE_MODEL
    assert gen.meta["elapsed_seconds"] >= 0
    assert gen.meta["tokens"] == {"input": 10, "output": 20}


def test_vanilla_prompt_includes_all_input_sections():
    case = _make_case()
    prompt = build_vanilla_prompt(case)

    assert case.story_outline in prompt
    assert case.previous_context in prompt
    assert case.target_chapter_outline in prompt
    assert "3000" in prompt  # word_target


def test_vanilla_passes_model_and_prompt_to_client():
    client = _FakeClient("正文")
    adapter = VanillaLLMAdapter(client, model="my-model")
    asyncio.run(adapter.generate(_make_case()))

    kwargs = client.chat.completions.last_kwargs
    assert kwargs["model"] == "my-model"
    assert kwargs["messages"][0]["role"] == "user"
    assert "本章大纲" in kwargs["messages"][0]["content"]
    assert kwargs["reasoning_effort"] == "low"


def test_vanilla_empty_content_is_tolerated():
    gen = asyncio.run(VanillaLLMAdapter(_FakeClient("  ")).generate(_make_case()))
    assert gen.content == ""
    assert gen.meta["elapsed_seconds"] >= 0


# ── NovelAgentAdapter 纯逻辑（不依赖 LLM） ───────────────


def test_map_initial_state_field_mapping():
    adapter = NovelAgentAdapter()
    state = adapter._map_initial_state(_make_case(), persist_dir="/tmp/eval")

    assert state["project_id"] == ""  # 空 project_id 跳过 ProjectManager/Chroma 检索
    assert state["retry_count"] == 0
    assert state["target_chapter_words"] == 3000
    assert state["recent_summary"] == _make_case().previous_context
    assert state["character_context"] == ""
    assert state["world_context"] == ""
    assert state["existing_world_entities"] == []
    assert state["narrative_mode"] is None
    assert state["persist_dir"] == "/tmp/eval"
    assert isinstance(state["chapter_number"], int) and state["chapter_number"] >= 1


def test_map_initial_state_stage_to_story_length():
    adapter = NovelAgentAdapter()
    expected = {"opening": "short", "middle": "medium", "long": "long"}
    for stage, want in expected.items():
        state = adapter._map_initial_state(_make_case(stage=stage), persist_dir="")
        assert state["story_length"] == want, f"stage={stage}"


def test_map_initial_state_story_outline_prepended_to_chapter_outline():
    """主仓库 Orchestrator 只从 chapter_outline 读大纲 → story_outline 前置到本章大纲。"""
    case = _make_case()
    adapter = NovelAgentAdapter()
    state = adapter._map_initial_state(case, persist_dir="")

    assert case.story_outline in state["chapter_outline"]
    assert case.target_chapter_outline in state["chapter_outline"]
    assert "## 全书大纲" in state["chapter_outline"]
    assert "## 本章大纲" in state["chapter_outline"]


def test_chapter_number_is_deterministic():
    adapter = NovelAgentAdapter()
    a = adapter._chapter_number(_make_case())
    b = adapter._chapter_number(_make_case())
    assert a == b


def test_extract_meta_from_final_state():
    values = {
        "evolution_history": [
            {"v": 0, "composite": 70.0},
            {"v": 1, "composite": 82.5},
            {"v": 2, "composite": 78.0},
        ],
        "evolution_termination": "converged",
        "evolution_best_version": 1,
        "editor_report": {"overall_score": 80},
        "continuity_report": {"overall_score": 85},
        "human_approved": True,
    }
    meta = NovelAgentAdapter._extract_meta(values, elapsed=1.234, evolution_enabled=True)

    assert meta["composite_score"] == 82.5  # 历史里最高 composite
    assert meta["evolution_rounds"] == 3
    assert meta["evolution_termination"] == "converged"
    assert meta["evolution_best_version"] == 1
    assert meta["editor_overall"] == 80
    assert meta["continuity_overall"] == 85
    assert meta["human_approved"] is True
    assert meta["evolution_enabled"] is True
    assert meta["elapsed_seconds"] == 1.234
    assert meta["tokens"] is None  # 主仓库 state 无 token 字段


def test_extract_meta_empty_history():
    meta = NovelAgentAdapter._extract_meta({}, elapsed=0.5, evolution_enabled=False)
    assert meta["composite_score"] is None
    assert meta["evolution_rounds"] == 0
    assert meta["evolution_termination"] == ""


# ── NovelAgentAdapter.generate 集成（真实 graph，需 LLM） ─


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_novel_agent_adapter_generate_integration():
    """真实跑主仓库进化流水线，产出章节正文 + meta 信号。"""
    adapter = NovelAgentAdapter(evolution_enabled=True)
    gen = asyncio.run(adapter.generate(_make_case()))

    assert gen.content.strip(), "draft_content 不应为空"
    assert gen.meta["adapter"] == "novel_agent"
    assert gen.meta["evolution_enabled"] is True
    assert "elapsed_seconds" in gen.meta
    # 至少跑完一轮进化，history 有记录
    assert gen.meta["evolution_rounds"] >= 1
