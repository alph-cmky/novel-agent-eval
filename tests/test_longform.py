# tests/test_longform.py
"""run_longform / degradation mock 测试：不消耗真实 API。"""
import asyncio

import pytest

from novel_agent_eval.agents.base import GeneratedChapter
from novel_agent_eval.eqbench_bridge import LongformPlan
from novel_agent_eval.longform import (
    degradation_score,
    render_longform_table,
    run_longform,
)


class _FakeAgent:
    name = "fake_agent"

    def __init__(self):
        self.contexts = []  # 记录每次 generate 收到的 previous_context

    async def generate(self, case):
        self.contexts.append(case.previous_context)
        return GeneratedChapter(content=f"chapter text {len(self.contexts)}", meta={"tokens": None})


class _FakeJudge:
    def __init__(self, scores):
        self._scores = scores
        self.chapter_numbers = []

    async def score_chapter(self, **kwargs):
        self.chapter_numbers.append(kwargs["chapter_number"])
        return dict(self._scores)


def _plan(n_chapters=8) -> LongformPlan:
    return LongformPlan(
        prompt_id="1",
        title="T",
        category="F",
        writing_prompt="Write a story.",
        n_chapters=n_chapters,
        final_plan="# Intention\nPlan text",
        character_profiles="# Hero\nProfile",
        step_outputs={},
    )


# 全部 14 维打满 20（负向维 0 → 反转后 20），单章分恒为 ~20
def _full_scores() -> dict[str, float]:
    return {
        "Nuanced Characters": 20, "Emotionally Engaging": 20, "Compelling Plot": 20,
        "Coherent": 20, "Weak Dialogue": 0, "Tell-Don't-Show": 0,
        "Unsurprising or Uncreative": 0, "Amateurish": 0, "Purple Prose": 0,
        "Forced Poetry or Metaphor": 0, "Well-earned Lightness or Darkness": 20,
        "Characters Consistent with Profile": 20, "Followed Chapter Plan": 20,
        "Faithful to Writing Prompt": 20,
    }


def test_degradation_score():
    assert degradation_score([10, 10, 10, 5, 5, 5], window=2) == pytest.approx(-5.0)
    assert degradation_score([10, 10, 10, 0, 0, 0], window=2) == pytest.approx(-10.0)
    assert degradation_score([1, 2, 3], window=2) is None  # 长度不足 2*window


def test_run_longform_generates_8_chapters_with_context():
    agent = _FakeAgent()
    judge = _FakeJudge(_full_scores())
    result = asyncio.run(run_longform(agent=agent, judge=judge, plan=_plan()))
    assert len(result.chapters) == 8
    assert judge.chapter_numbers == list(range(1, 9))
    # 第 1 章无前文；后续章累积前文
    assert agent.contexts[0] == ""
    assert "[Chapter 1]" in agent.contexts[1]
    assert "[Chapter 7]" in agent.contexts[7]
    assert result.mean_score > 0
    assert result.eqbench_0_100 == pytest.approx(result.mean_score * 5)
    assert result.degradation == 0.0  # 每章分恒定，无衰减


def test_run_longform_degradation_negative_when_tail_drops():
    agent = _FakeAgent()

    class _DecliningJudge:
        def __init__(self):
            self._i = 0

        async def score_chapter(self, **kwargs):
            # 前 4 章 20 分，后 4 章弱化（负向维抬高 → 反转后降分）
            self._i += 1
            s = _full_scores()
            if self._i > 4:
                s["Amateurish"] = 10  # 反转后 10，拉低单章分
            return s

    result = asyncio.run(run_longform(agent=agent, judge=_DecliningJudge(), plan=_plan()))
    assert result.degradation < 0


def test_render_longform_table():
    agent = _FakeAgent()
    judge = _FakeJudge(_full_scores())
    result = asyncio.run(run_longform(agent=agent, judge=judge, plan=_plan()))
    table = render_longform_table([result])
    assert "fake_agent" in table
    assert "eqbench(0-100)" in table
    assert "degradation" in table
    # per-chapter 列含 8 个分值
    assert table.count("/") >= 7
