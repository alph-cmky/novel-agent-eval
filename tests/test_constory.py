# tests/test_constory.py
"""ConStory-Checker 适配器 mock 测试：不消耗真实 StepFun API、不联网。

注入 mock AsyncOpenAI client（chat.completions.create 按 user message 里的 Query
识别类别并返回固定 content），覆盖 3 类聚合 / 全空 / 失败降级，以及
_split_chatml / consistency_score 两个纯函数。
"""
import asyncio
import json

from novel_agent_eval.constory import (
    ConsistencyReport,
    ConStoryCheckerAdapter,
    _split_chatml,
    consistency_score,
)
from novel_agent_eval.vendor.constory.judge import EVALUATION_CRITERIA

# 固定错误：character / timeline / narrative_style 各 1 条，其余子类型空
_FIXED_ERRORS = {
    "characterization_memory_contradictions": {
        "exact_quote": "他体内有神秘力量",
        "location": "第2段",
        "contradiction_pair": "前文说力量被封印 / 此处说力量可用",
        "contradiction_location": "第1段",
        "context": "角色力量状态矛盾",
    },
    "timeline_plot_absolute_time_contradictions": {
        "exact_quote": "三日后",
        "location": "第5段",
        "contradiction_pair": "前文说七日后 / 此处说三日后",
        "contradiction_location": "第3段",
        "context": "时间线矛盾",
    },
    "narrative_style_tone_inconsistencies": {
        "exact_quote": "话锋一转",
        "location": "第8段",
        "context": "语气不一致",
    },
}


def _make_content(criteria_name: str, errors: dict) -> str:
    """构造某类别原始 LLM 响应 content：子类型名（bare）→ 错误数组。"""
    content = {
        sc: [errors[f"{criteria_name}_{sc}"]]
        if f"{criteria_name}_{sc}" in errors
        else []
        for sc in EVALUATION_CRITERIA[criteria_name]["sub_criteria"]
    }
    return json.dumps(content, ensure_ascii=False)


class _FakeCompletions:
    """mock client.chat.completions.create：按 user message 里的 Query 识别类别。

    识别到类别后返回该类的固定 content；fail_categories 里的类别抛异常（模拟 API 失败）。
    """

    def __init__(self, responses: dict[str, str], fail_categories=()):
        self._responses = responses
        self._fail = set(fail_categories)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        user = kwargs["messages"][1]["content"]
        content = "{}"
        for cat, cfg in EVALUATION_CRITERIA.items():
            if f"{cfg['name']} Analysis" in user:
                if cat in self._fail:
                    raise RuntimeError("HTTP 429")
                content = self._responses[cat]
                break
        message = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, responses: dict[str, str], fail_categories=()):
        self.chat = type(
            "Chat", (), {"completions": _FakeCompletions(responses, fail_categories)}
        )()


def _responses(errors: dict) -> dict[str, str]:
    return {cat: _make_content(cat, errors) for cat in EVALUATION_CRITERIA}


def _run(
    errors: dict,
    fail_categories=(),
    narrative: str = "测试正文……",
) -> ConsistencyReport:
    client = _FakeClient(_responses(errors), fail_categories)
    return asyncio.run(ConStoryCheckerAdapter(client=client).check_consistency(narrative))


def test_check_consistency_aggregates_3_classes():
    report = _run(_FIXED_ERRORS)

    assert len(report.character) == 1
    assert report.character[0].subtype == "characterization_memory_contradictions"
    assert report.character[0].exact_quote == "他体内有神秘力量"
    assert report.character[0].contradiction_pair is not None

    assert len(report.timeline) == 1
    assert report.timeline[0].subtype == "timeline_plot_absolute_time_contradictions"

    assert report.worldbuilding == []
    assert report.total == 2

    # narrative_style 只进 raw，不计入 3 类
    assert len(report.raw["narrative_style_tone_inconsistencies"]) == 1
    assert len(report.raw["characterization_memory_contradictions"]) == 1
    assert report.raw["characterization_knowledge_contradictions"] == []


def test_check_consistency_all_empty_total_zero():
    report = _run({})

    assert report.total == 0
    assert report.character == []
    assert report.timeline == []
    assert report.worldbuilding == []
    assert all(items == [] for items in report.raw.values())


def test_check_consistency_empty_narrative_total_zero():
    report = _run({}, narrative="")

    assert report.total == 0


def test_check_consistency_failed_category_degrades():
    # characterization 类失败 → 该 4 子类型空，不抛异常，其余类正常聚合
    report = _run(_FIXED_ERRORS, fail_categories=("characterization",))

    assert report.character == []
    assert report.raw["characterization_memory_contradictions"] == []
    assert len(report.timeline) == 1
    assert report.timeline[0].subtype == "timeline_plot_absolute_time_contradictions"
    assert report.total == 1
    assert report.failed_categories == ["characterization"]


def test_split_chatml_extracts_system_and_user():
    template = (
        "<|im_start|>system\n你是检查器。\n<|im_end|>\n\n"
        "<|im_start|>user\n正文：{{ Content }}\n<|im_end|>"
    )
    system, user = _split_chatml(template)
    assert system == "你是检查器。"
    assert "正文：" in user
    assert "<|im_start|>" not in system
    assert "<|im_start|>" not in user


def test_consistency_score_penalty():
    assert consistency_score(0) == 100
    assert consistency_score(2) == 60
    assert consistency_score(5) == 0
    assert consistency_score(6) == 0  # 下限 0，不出现负分
