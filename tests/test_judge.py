# tests/test_judge.py
"""Judge mock 测试：不消耗真实 StepFun API。"""
import asyncio
import json

from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.judge import Judge, JudgeScore, QUALITY_DIMS


def _fixed() -> dict:
    """每次返回全新固定 JSON，避免测试间共享 dict 被误改。"""
    return {
        "dimensions": {"consistency": 85, "writing": 78, "ai_flavor": 72, "dialogue": 80,
                       "plot": 75, "instruction": 90, "creativity": 70, "controllability": 65},
        "overall": 77,
    }


def _make_case() -> EvalCase:
    return EvalCase(
        name="test",
        stage="opening",
        story_outline="主角穿越到玄幻大陆，立志成为剑仙。",
        previous_context="第一章：主角在异世界醒来，发现体内有神秘力量。",
        target_chapter_outline="第二章：主角拜入青云剑派，随师父学剑。",
    )


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


def _run(client, draft="第二章正文……"):
    return asyncio.run(Judge(client).score(draft, _make_case()))


def test_score_returns_judge_score():
    score = _run(_FakeClient(json.dumps(_fixed(), ensure_ascii=False)))
    assert isinstance(score, JudgeScore)
    assert set(score.dimensions) == set(QUALITY_DIMS)
    for v in score.dimensions.values():
        assert isinstance(v, int) and 0 <= v <= 100
    assert score.overall == 77


def test_score_parses_markdown_code_block():
    wrapped = "```json\n" + json.dumps(_fixed(), ensure_ascii=False) + "\n```"
    score = _run(_FakeClient(wrapped))
    assert set(score.dimensions) == set(QUALITY_DIMS)
    assert score.dimensions["consistency"] == 85
    assert score.overall == 77


def test_score_fills_missing_dim_with_zero_after_retries():
    partial = _fixed()
    del partial["dimensions"]["dialogue"]
    client = _FakeClient(json.dumps(partial, ensure_ascii=False))
    score = _run(client)
    assert client.chat.completions.calls == 3          # 初次 + 2 次重试
    assert set(score.dimensions) == set(QUALITY_DIMS)  # 缺失维度 0 兜底
    assert score.dimensions["dialogue"] == 0
    assert score.dimensions["consistency"] == 85


def test_score_overall_falls_back_to_mean():
    data = _fixed()
    del data["overall"]
    score = _run(_FakeClient(json.dumps(data, ensure_ascii=False)))
    vals = list(data["dimensions"].values())
    assert score.overall == round(sum(vals) / len(vals))


def test_score_garbage_retries_then_all_zero():
    client = _FakeClient("完全不是 JSON 的输出")
    score = _run(client)
    assert client.chat.completions.calls == 3
    assert set(score.dimensions) == set(QUALITY_DIMS)
    assert all(v == 0 for v in score.dimensions.values())
    assert score.overall == 0


def test_score_accepts_flat_dimensions():
    # 模型平铺输出 8 维（漏 dimensions 外壳）+ overall → 首次即成功，不重试
    flat = {"consistency": 85, "writing": 78, "ai_flavor": 72, "dialogue": 80,
            "plot": 75, "instruction": 90, "creativity": 70, "controllability": 65,
            "overall": 77}
    client = _FakeClient(json.dumps(flat, ensure_ascii=False))
    score = _run(client)
    assert client.chat.completions.calls == 1
    assert score.dimensions["consistency"] == 85
    assert score.dimensions["controllability"] == 65
    assert score.overall == 77


def test_score_accepts_flat_dimensions_missing_dim():
    # 平铺但缺一个维度 → 重试后 0 兜底（与嵌套缺失维度行为一致）
    flat = {"consistency": 85, "writing": 78, "ai_flavor": 72, "dialogue": 80,
            "plot": 75, "instruction": 90, "creativity": 70, "overall": 77}
    client = _FakeClient(json.dumps(flat, ensure_ascii=False))
    score = _run(client)
    assert client.chat.completions.calls == 3
    assert score.dimensions["controllability"] == 0
    assert score.dimensions["consistency"] == 85


def test_score_accepts_flat_without_overall():
    # 平铺 8 维全 + 无 overall → overall 退化为 8 维平均
    flat = {"consistency": 80, "writing": 80, "ai_flavor": 80, "dialogue": 80,
            "plot": 80, "instruction": 80, "creativity": 80, "controllability": 80}
    client = _FakeClient(json.dumps(flat, ensure_ascii=False))
    score = _run(client)
    assert client.chat.completions.calls == 1
    assert score.overall == 80
