# tests/test_e2e_smoke.py
"""端到端冒烟测试：跑通 generate → judge → metrics → report 全链路（全 mock，无 LLM / graph / 网络）。

- Test 1：1 个自制 case + mock agent + mock judge → BenchmarkRunner.run_suite
  → render_scorecard / render_json，断言跑分卡关键节头与 JSON 结构。
- Test 2：跨仓库（novel-agent 路径依赖）真实 import `novel_agent.style.ai_flavor.detect_ai_flavor`，
  对 Test 1 同款固定中文章节跑一遍 AI 味检测，断言返回 dict 结构。

异步约定：本仓库未装 pytest-asyncio，沿用 tests/test_runner.py 的 `asyncio.run()` 助手模式。
"""
import asyncio
import json
from pathlib import Path

from novel_agent_eval.agents.base import GeneratedChapter
from novel_agent_eval.dataset.loader import load_cases
from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.judge import QUALITY_DIMS, JudgeScore
from novel_agent_eval.report import render_json, render_scorecard
from novel_agent_eval.runner import BenchmarkReport, BenchmarkRunner

_SELF_BUILT_DIR = Path(__file__).resolve().parents[1] / "novel_agent_eval" / "dataset" / "self_built"

# 固定中文章节选段：多段落 + 对话，足够长使 detect_ai_flavor 的结构检查不触发「太少」。
FIXED_CHINESE_TEXT = (
    "清晨的山风裹着松针的气息，从青云剑派的山门一路灌进演武场。林远站在青石台阶上，"
    "掌心贴着腰间的剑柄，指尖微微发凉。他昨夜几乎没睡，反复揣摩师父教的那一式「破云」，"
    "此刻手腕还残留着酸痛。\n\n"
    "「今日起，你每日卯时来此，挥剑三百次。」师父负手而立，声音不高，却清晰地传进他耳中。\n\n"
    "「三百次？」林远愣了一下，抬头看向师父，「那……弟子什么时候才能学新剑招？」\n\n"
    "「根基未稳，学再多也是花架子。」师父瞥了他一眼，目光沉静得像一口古井，「剑招可以速成，"
    "剑心不能。你若真想走得远，就得先学会耐住这份枯燥。」\n\n"
    "林远咬了咬嘴唇，没有反驳。他想起穿越到这个世界的第一天，自己连剑都握不稳，如今好歹能把"
    "一套入门剑法完整走下来。山门外的云海翻涌起伏，像他此刻说不清道不明的心绪。\n\n"
    "他深吸一口气，拔出长剑，剑尖在晨光里划出一道雪亮的弧线。第一剑，力道过猛，手腕震得发麻；"
    "第二剑，他刻意放慢，反而找着了那股顺着剑身流淌的劲。第三剑下去，风声掠过剑刃，发出极轻的嗡鸣。\n\n"
    "直到第一百二十剑，他才停下来喘气，汗水顺着额角淌进领口。师父不知何时已经走了，石阶上只留下"
    "几片被风卷落的落叶。林远望着空荡荡的演武场，忽然明白师父那句话的深意。他握紧剑柄，又挥出了"
    "第一百二十一剑。"
)


class MockAgent:
    """固定产出：同一段中文正文 + 完整效率 meta（elapsed / tokens / evolution_rounds / composite_score）。"""

    name = "mock_agent"

    @staticmethod
    async def generate(case: EvalCase) -> GeneratedChapter:
        return GeneratedChapter(FIXED_CHINESE_TEXT, meta={
            "elapsed_seconds": 42.5,
            "tokens": {"input": 800, "output": 2200},
            "evolution_rounds": 3,
            "evolution_termination": "converged",
            "composite_score": 78.0,
        })


class MockJudge:
    """固定打分：8 个质量维全 80 分，overall 80（不触碰真实 Judge / LLM）。"""

    @staticmethod
    async def score(draft: str, case: EvalCase) -> JudgeScore:
        return JudgeScore(dimensions={d: 80 for d in QUALITY_DIMS}, overall=80)


def _run(coro):
    return asyncio.run(coro)


# ── Test 1：完整链路 generate → judge → metrics → report ──


def test_full_chain_smoke():
    """run_suite → render_scorecard / render_json：跑分卡含关键节头，JSON 结构完整。"""
    case = load_cases(str(_SELF_BUILT_DIR))[0]  # 恰好 1 个自制 case

    runner = BenchmarkRunner(judge=MockJudge(), repeat=2)
    report = _run(runner.run_suite([MockAgent()], [case]))

    assert isinstance(report, BenchmarkReport)
    assert report.repeat == 2
    assert report.agents == [MockAgent.name]
    assert report.cases == [case.name]

    # ── scorecard ──
    md = render_scorecard(report)
    assert isinstance(md, str) and md
    assert "### 总分" in md
    assert "内部信号 vs 外部评测一致性" in md
    assert MockAgent.name in md
    assert "### 效率" in md
    # 内部/外部一致性表该 stage 有 composite_score → 渲染均值而非「—」占位
    assert "| 长程案例均值 | 78.0 |" in md

    # ── json ──
    data = json.loads(render_json(report))
    assert "results" in data
    assert data["results"][0]["agent"] == MockAgent.name
    assert "runs" in data["results"][0]
    assert len(data["results"][0]["runs"]) == 2


# ── Test 2：跨仓库真实 detect_ai_flavor ──


def test_detect_ai_flavor_cross_repo():
    """novel_agent.style.ai_flavor 可 import 并对固定章节返回结构化的 AI 味报告。"""
    from novel_agent.style.ai_flavor import detect_ai_flavor

    result = detect_ai_flavor(FIXED_CHINESE_TEXT)

    assert isinstance(result, dict)
    assert isinstance(result["overall_score"], int)
    assert 0 <= result["overall_score"] <= 100
    assert isinstance(result["issues"], list)
