# tests/test_constory.py
"""ConStory-Checker 适配器 mock 测试：不消耗真实 StepFun API、不联网。"""
import asyncio
import json

import pytest

from novel_agent_eval.constory import ConsistencyReport, ConStoryCheckerAdapter
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


async def _fake_evaluate_criteria(
    self, session, prompt_template, story_content, criteria_name
):
    return {"success": True, "content": _make_content(criteria_name, _FIXED_ERRORS)}


async def _fake_empty_evaluate_criteria(
    self, session, prompt_template, story_content, criteria_name
):
    return {"success": True, "content": _make_content(criteria_name, {})}


async def _fake_failing_evaluate_criteria(
    self, session, prompt_template, story_content, criteria_name
):
    # characterization 类失败 → 该类子类型记空，其余正常
    if criteria_name == "characterization":
        return {"success": False, "error": "HTTP 429"}
    return {"success": True, "content": _make_content(criteria_name, _FIXED_ERRORS)}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STEPFUN_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")


def _run(monkeypatch, fake, narrative="测试正文……") -> ConsistencyReport:
    monkeypatch.setattr(
        "novel_agent_eval.vendor.constory.judge.JudgeLLMClient.evaluate_criteria",
        fake,
    )
    return asyncio.run(ConStoryCheckerAdapter().check_consistency(narrative))


def test_check_consistency_aggregates_3_classes(monkeypatch, env):
    report = _run(monkeypatch, _fake_evaluate_criteria)

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


def test_check_consistency_all_empty_total_zero(monkeypatch, env):
    report = _run(monkeypatch, _fake_empty_evaluate_criteria)

    assert report.total == 0
    assert report.character == []
    assert report.timeline == []
    assert report.worldbuilding == []
    assert all(items == [] for items in report.raw.values())


def test_check_consistency_empty_narrative_total_zero(monkeypatch, env):
    report = _run(monkeypatch, _fake_empty_evaluate_criteria, narrative="")

    assert report.total == 0


def test_check_consistency_failed_category_degrades(monkeypatch, env):
    # characterization 类失败 → 该 4 子类型空，不抛异常，其余类正常聚合
    report = _run(monkeypatch, _fake_failing_evaluate_criteria)

    assert report.character == []
    assert report.raw["characterization_memory_contradictions"] == []
    assert len(report.timeline) == 1
    assert report.timeline[0].subtype == "timeline_plot_absolute_time_contradictions"
    assert report.total == 1
