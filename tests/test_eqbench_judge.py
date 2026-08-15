# tests/test_eqbench_judge.py
"""EQBenchJudge mock 测试：不消耗真实 DeepSeek API。"""
import asyncio

import pytest

from novel_agent_eval.eqbench_judge import (
    CRITERIA,
    NEGATIVE_CRITERIA,
    EQBenchJudge,
    eqbench_chapter_score,
    invert_if_negative,
    parse_eqbench_scores,
)


# ── mock client（对齐 test_judge 的 FakeClient 风格） ──────────────


class _FakeCompletions:
    """模拟 client.chat.completions.create，返回固定 content，并记录调用次数。"""

    def __init__(self, content):
        self._content = content
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        message = type("Msg", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


# ── 14 维样本响应 ────────────────────────────────────────────────


def _chapter_response(**overrides) -> str:
    """生成 14 维 "Metric Name: [Score]" 响应，overrides 覆盖指定维。"""
    base = {
        "Nuanced Characters": 16,
        "Emotionally Engaging": 15,
        "Compelling Plot": 14,
        "Coherent": 18,
        "Weak Dialogue": 3,
        "Tell-Don't-Show": 4,
        "Unsurprising or Uncreative": 5,
        "Amateurish": 2,
        "Purple Prose": 1,
        "Forced Poetry or Metaphor": 0,
        "Well-earned Lightness or Darkness": 13,
        "Characters Consistent with Profile": 17,
        "Followed Chapter Plan": 19,
        "Faithful to Writing Prompt": 18,
    }
    base.update(overrides)
    lines = ["[Analysis]", "A sample analysis.", "[Scores]", ""]
    lines += [f"{name}: [{score}]" for name, score in base.items()]
    return "\n".join(lines)


def _kwargs() -> dict:
    return dict(
        writing_prompt="Write a fantasy chapter.",
        final_plan="The hero enters the city.",
        character_profiles="Hero: brave.",
        chapter_number=1,
        chapter_text="The hero walked into the city.",
    )


# ── parse_eqbench_scores ─────────────────────────────────────────


def test_parse_eqbench_scores_basic():
    scores = parse_eqbench_scores(_chapter_response())
    assert set(scores) == set(CRITERIA)
    assert scores["Nuanced Characters"] == 16
    assert scores["Weak Dialogue"] == 3


def test_parse_eqbench_scores_filters_non_metric_and_clamps():
    text = (
        "[Analysis]\nsome text\n[Scores]\n"
        "Nuanced Characters: [25]\n"
        "Weak Dialogue: [-2]\n"
        "Overall Assessment: [15]\n"
    )
    scores = parse_eqbench_scores(text)
    assert scores["Nuanced Characters"] == 20  # clamp 上限
    assert scores["Weak Dialogue"] == 0         # clamp 下限
    assert "Overall Assessment" not in scores   # 非指标行过滤


def test_parse_eqbench_scores_empty():
    assert parse_eqbench_scores("") == {}
    assert parse_eqbench_scores("no scores here at all") == {}


# ── invert_if_negative / eqbench_chapter_score ────────────────────


def test_invert_if_negative():
    assert invert_if_negative("Weak Dialogue", 3) == 17
    assert invert_if_negative("weak dialogue", 0) == 20   # 大小写不敏感
    assert invert_if_negative("Nuanced Characters", 3) == 3  # 正向原样


def test_eqbench_chapter_score_forced_poetry_scaling():
    # 仅 forced poetry 一维：反转后 10 → (10/20)^1.7*20 ≈ 6.156，权重 5 约掉
    expected = (10 / 20) ** 1.7 * 20
    assert eqbench_chapter_score({"Forced Poetry or Metaphor": 10}) == pytest.approx(expected)


def test_eqbench_chapter_score_weighted_mean():
    # 两正向 20 + 两负向 20（反转后 0）：加权 40/8 = 5.0
    scores = {
        "Nuanced Characters": 20,
        "Emotionally Engaging": 20,
        "Forced Poetry or Metaphor": 20,  # weight 5, 反转后 0
        "Purple Prose": 20,               # weight 1, 反转后 0
    }
    assert eqbench_chapter_score(scores) == pytest.approx(5.0)


def test_eqbench_chapter_score_empty_returns_none():
    assert eqbench_chapter_score({}) is None


# ── EQBenchJudge ─────────────────────────────────────────────────


def test_build_prompt_replaces_all_placeholders():
    judge = EQBenchJudge(client=_FakeClient(_chapter_response()))
    prompt = judge._build_prompt(**_kwargs())
    assert "{writing_prompt}" not in prompt
    assert "{final_plan}" not in prompt
    assert "{character_profiles}" not in prompt
    assert "{chapter_number}" not in prompt
    assert "{chapter_text}" not in prompt
    assert "{creative_writing_criteria}" not in prompt
    assert "{lower_is_better_criteria}" not in prompt
    assert "Write a fantasy chapter." in prompt
    assert "The hero walked into the city." in prompt


def test_score_chapter_returns_raw_14_dim_dict():
    client = _FakeClient(_chapter_response())
    judge = EQBenchJudge(client=client)
    scores = asyncio.run(judge.score_chapter(**_kwargs()))
    assert set(scores) == set(CRITERIA)
    assert scores["Weak Dialogue"] == 3  # 原始分，未反转（反转在 eqbench_chapter_score）


def test_score_chapter_retries_on_empty_parse():
    client = _FakeClient("garbage with no metric: lines")
    judge = EQBenchJudge(client=client, max_attempts=3)
    scores = asyncio.run(judge.score_chapter(**_kwargs()))
    assert scores == {}
    assert client.chat.completions.calls == 3


def test_score_chapter_median_across_samples():
    # n_samples=3 时逐维取中位数，抑制极端分
    class _SeqCompletions:
        def __init__(self, contents):
            self._contents = list(contents)
            self._idx = 0

        async def create(self, **kwargs):
            content = self._contents[self._idx % len(self._contents)]
            self._idx += 1
            message = type("Msg", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Resp", (), {"choices": [choice]})()

    client = type(
        "Client",
        (),
        {
            "chat": type(
                "Chat",
                (),
                {
                    "completions": _SeqCompletions(
                        [
                            _chapter_response(**{"Nuanced Characters": 10}),
                            _chapter_response(**{"Nuanced Characters": 20}),
                            _chapter_response(**{"Nuanced Characters": 15}),
                        ]
                    )
                },
            )(),
        },
    )()
    judge = EQBenchJudge(client=client, n_samples=3)
    scores = asyncio.run(judge.score_chapter(**_kwargs()))
    assert scores["Nuanced Characters"] == 15  # [10, 20, 15] 中位数
    assert scores["Weak Dialogue"] == 3        # 三维一致取原值
