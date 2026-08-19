# tests/test_runner.py
"""BenchmarkRunner 测试（全 mock，不耗 LLM）。

- FakeAgent 返回固定 GeneratedChapter（meta 含 elapsed_seconds / tokens / evolution_rounds）。
- FakeJudge 返回固定 JudgeScore（8 质量维）。
- 覆盖：run_case 聚合均值±标准差、efficiency 并入 9 维加权、run_suite agent×case 全遍历、
  run_ablation 对 max_rounds 配置正确切换 adapter、未实现消融名 raise。
"""
import asyncio
import statistics
from typing import ClassVar

import pytest

from novel_agent_eval.agents.base import GeneratedChapter
from novel_agent_eval.constory import ConsistencyError, ConsistencyReport
from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.judge import QUALITY_DIMS, JudgeScore
from novel_agent_eval.metrics import STAGE_WEIGHTS
from novel_agent_eval.runner import BenchmarkRunner

NINE_DIMS = QUALITY_DIMS + ["efficiency"]


# ── mocks ──────────────────────────────────────────────


class FakeAgent:
    """固定产出：content + meta 可指定；记录被调用次数与收到的 case。"""

    def __init__(self, name: str, content: str = "第 N 章正文……", meta: dict | None = None):
        self.name = name
        self._content = content
        self._meta = meta or {}
        self.calls = 0
        self.seen_cases = []

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        self.calls += 1
        self.seen_cases.append(case)
        return GeneratedChapter(self._content, dict(self._meta))


class FakeJudge:
    """responses 为固定 JudgeScore，或 callable(draft, case) -> JudgeScore（按调用次数变化）。"""

    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    async def score(self, draft: str, case: EvalCase) -> JudgeScore:
        self.calls += 1
        if callable(self._responses):
            return self._responses(draft, case)
        return self._responses


def _fixed_score(**overrides) -> JudgeScore:
    dims = {d: 80 for d in QUALITY_DIMS}
    dims.update(overrides)
    return JudgeScore(dimensions=dims, overall=80)


def _all_dims(value: int) -> JudgeScore:
    return _fixed_score(**{d: value for d in QUALITY_DIMS})


def _case(stage: str = "opening", name: str = "opening_01") -> EvalCase:
    return EvalCase(
        name=name,
        stage=stage,
        story_outline="主角穿越到玄幻大陆，立志成为剑仙。",
        previous_context="第一章：主角在异世界醒来，发现体内有神秘力量。",
        target_chapter_outline="第二章：主角拜入青云剑派，随师父学剑。",
    )


def _meta(elapsed: float = 30.0, rounds: int = 0, tokens=100) -> dict:
    return {"elapsed_seconds": elapsed, "evolution_rounds": rounds, "tokens": tokens}


def _run(coro):
    return asyncio.run(coro)


# ── run_case：聚合均值±标准差 ───────────────────────────


def test_run_case_averages_repeats():
    """repeat=3，一致性维按调用次数 70/90/80 → mean 80 / std 10；其余维固定 80 → std 0。"""
    agent = FakeAgent("a", meta=_meta(elapsed=30, rounds=0))
    judge = FakeJudge(
        lambda draft, case: _fixed_score(consistency=[70, 90, 80][judge.calls - 1])
    )

    result = _run(BenchmarkRunner(judge).run_case(agent, _case(), repeat=3))

    assert result.repeat == 3
    assert result.agent == "a"
    assert result.case == "opening_01"
    assert set(result.dims_mean) == set(NINE_DIMS)          # 9 维 = 8 质量维 + efficiency
    assert result.dims_mean["consistency"] == 80
    assert result.dims_std["consistency"] == 10
    assert result.dims_std["writing"] == 0.0               # 固定维 → std 0
    assert len(result.runs) == 3
    assert all(r.meta["elapsed_seconds"] == 30 for r in result.runs)


def test_run_case_std_zero_when_single_repeat():
    """repeat=1：无样本方差，std 记 0.0（避免 stdev 抛 ValueError）。"""
    agent = FakeAgent("a", meta=_meta())
    judge = FakeJudge(_fixed_score())

    result = _run(BenchmarkRunner(judge).run_case(agent, _case(), repeat=1))

    assert result.overall_std == 0.0
    assert result.dims_std["consistency"] == 0.0


# ── efficiency 并入 9 维加权 ────────────────────────────


def test_efficiency_dimension_merged_and_weighted():
    """elapsed=300 → efficiency=30（efficiency_score(300, _, 0)=30）；overall 按 stage 加权。"""
    agent = FakeAgent("a", meta=_meta(elapsed=300, rounds=0))
    judge = FakeJudge(_fixed_score())  # 8 维全 80

    result = _run(BenchmarkRunner(judge).run_case(agent, _case(stage="opening"), repeat=1))

    run = result.runs[0]
    assert run.dimensions["efficiency"] == 30
    assert run.dimensions["consistency"] == 80
    assert len(run.dimensions) == 9

    w = STAGE_WEIGHTS["opening"]
    expected = round(sum(80 * w[d] for d in QUALITY_DIMS) + 30 * w["efficiency"], 2)
    assert run.overall == expected
    assert result.overall_mean == expected


# ── consistency_checker：ConStory 覆盖 consistency 维 ────


class FakeConsistencyChecker:
    """返回固定 ConsistencyReport（character/timeline/worldbuilding 各若干错误）。"""

    def __init__(self, character=1, timeline=1, worldbuilding=0):
        self._report = ConsistencyReport(
            character=[ConsistencyError(subtype="characterization_memory_contradictions", exact_quote="x")] * character,
            timeline=[ConsistencyError(subtype="timeline_plot_absolute_time_contradictions", exact_quote="y")] * timeline,
            worldbuilding=[ConsistencyError(subtype="world_building_core_rules_violations", exact_quote="z")] * worldbuilding,
            raw={},
            total=character + timeline + worldbuilding,
        )
        self.calls = 0

    async def check_consistency(self, narrative: str) -> ConsistencyReport:
        self.calls += 1
        return self._report


def test_consistency_checker_overrides_consistency_dim():
    """注入 checker 后：consistency 维 = consistency_score(total)，Judge 原始分与 ConStory 详情进 meta。"""
    agent = FakeAgent("a", meta=_meta())  # elapsed=30 → efficiency=100
    judge = FakeJudge(_fixed_score(consistency=85))  # Judge 打 85
    checker = FakeConsistencyChecker(character=1, timeline=1)  # 2 矛盾 → 60

    result = _run(BenchmarkRunner(judge, consistency_checker=checker).run_case(agent, _case(), repeat=1))
    run = result.runs[0]

    assert checker.calls == 1
    assert run.dimensions["consistency"] == 60          # consistency_score(2)=60
    assert run.meta["consistency_judge"] == 85          # Judge 原始分保留
    assert run.meta["consistency_constory"] == 60
    assert run.meta["consistency_errors"] == 2
    assert run.meta["consistency_breakdown"] == {"character": 1, "timeline": 1, "worldbuilding": 0}

    # overall 按覆盖后的 consistency 加权，其余质量维仍 80，efficiency=100
    w = STAGE_WEIGHTS["opening"]
    expected = round(
        sum(80 * w[d] for d in QUALITY_DIMS if d != "consistency")
        + 60 * w["consistency"]
        + 100 * w["efficiency"],
        2,
    )
    assert run.overall == expected


def test_no_consistency_checker_leaves_consistency_and_meta_untouched():
    """不注入 checker：consistency 维保持 Judge 原值，meta 不含 consistency_* 键。"""
    agent = FakeAgent("a", meta=_meta())
    judge = FakeJudge(_fixed_score(consistency=85))

    run = _run(BenchmarkRunner(judge).run_case(agent, _case(), repeat=1)).runs[0]

    assert run.dimensions["consistency"] == 85
    assert "consistency_judge" not in run.meta
    assert "consistency_constory" not in run.meta


# ── run_suite：agent × case 全遍历 ──────────────────────


def test_run_suite_averages_repeats():
    """2 agent × 2 case，repeat=3 → 4 个 BenchmarkResult，每 case 聚合均值±标准差。"""
    agents = [
        FakeAgent("novel_agent", meta=_meta(elapsed=30)),
        FakeAgent("vanilla_llm", meta=_meta(elapsed=60)),
    ]
    cases = [_case(stage="opening", name="opening_01"), _case(stage="long", name="long_01")]
    judge = FakeJudge(_fixed_score())

    report = _run(BenchmarkRunner(judge).run_suite(agents, cases, repeat=3))

    assert len(report.results) == 4
    assert report.repeat == 3
    assert report.agents == ["novel_agent", "vanilla_llm"]
    assert report.cases == ["opening_01", "long_01"]

    m = report.matrix()
    assert set(m) == {"novel_agent", "vanilla_llm"}
    assert set(m["novel_agent"]) == {"opening_01", "long_01"}
    assert set(m["vanilla_llm"]) == {"opening_01", "long_01"}

    r = m["novel_agent"]["opening_01"]
    assert r.repeat == 3
    assert r.overall_mean == r.runs[0].overall  # 固定分 → 均值 = 单次分
    assert r.overall_std == 0.0


def test_run_suite_each_agent_case_run_once():
    """每个 (agent, case) 组合恰好跑 repeat 次 generate + score。"""
    agent = FakeAgent("a", meta=_meta())
    judge = FakeJudge(_fixed_score())

    _run(BenchmarkRunner(judge).run_suite([agent], [_case()], repeat=2))

    assert agent.calls == 2
    assert judge.calls == 2


# ── run_ablation：消融配置切换 / raise ──────────────────


class _RecordingAdapter:
    """记录构造参数，generate 返回固定产出（替代真 NovelAgentAdapter，避免跑主仓库 graph）。"""

    name = "recording_adapter"
    constructed: ClassVar[list] = []
    generate_calls = 0

    def __init__(self, **kwargs):
        _RecordingAdapter.constructed.append(dict(kwargs))

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        _RecordingAdapter.generate_calls += 1
        return GeneratedChapter("draft", _meta(elapsed=30, rounds=0))


def test_run_ablation_switches_adapter_for_round_budget(monkeypatch):
    """max_rounds 消融使用不同的进化预算。"""
    import novel_agent_eval.runner as runner_mod

    _RecordingAdapter.constructed = []
    monkeypatch.setattr(runner_mod, "NovelAgentAdapter", _RecordingAdapter)
    judge = FakeJudge(_fixed_score())

    results = _run(BenchmarkRunner(judge).run_ablation(_case(), modules=["max_rounds_0"]))

    assert set(results) == {"baseline", "max_rounds_0"}
    assert _RecordingAdapter.constructed == [
        {"max_rounds": 2, "label": "novel_agent"},
        {"max_rounds": 0, "label": "novel_agent_r0"},
    ]
    assert results["max_rounds_0"].dims_mean["consistency"] == 80


def test_run_ablation_raises_for_unimplemented_modules(monkeypatch):
    """无 Continuity / Worldbuilding / EvoOrchestrator LLM 增强等消融，主仓库不支持 → raise。"""
    import novel_agent_eval.runner as runner_mod

    _RecordingAdapter.constructed = []
    monkeypatch.setattr(runner_mod, "NovelAgentAdapter", _RecordingAdapter)
    judge = FakeJudge(_fixed_score())

    for bad in ["no_continuity", "no_worldbuilding", "no_orchestrator_llm", "unknown_x"]:
        with pytest.raises(NotImplementedError):
            _run(BenchmarkRunner(judge).run_ablation(_case(), modules=[bad]))


def test_run_ablation_raises_before_generating(monkeypatch):
    """无效消融名在 baseline run_case 前即 raise：generate 从未被调用（不浪费一次付费生成）。"""
    import novel_agent_eval.runner as runner_mod

    _RecordingAdapter.constructed = []
    _RecordingAdapter.generate_calls = 0
    monkeypatch.setattr(runner_mod, "NovelAgentAdapter", _RecordingAdapter)
    judge = FakeJudge(_fixed_score())

    with pytest.raises(NotImplementedError):
        _run(BenchmarkRunner(judge).run_ablation(_case(), modules=["no_continuity"]))

    assert _RecordingAdapter.generate_calls == 0


# ── compare：跑分卡 ─────────────────────────────────────


def _expected_overall(value: int, stage: str) -> float:
    """8 质量维全为 value、efficiency=100（elapsed=30 → efficiency_score=100）时的 overall。"""
    w = STAGE_WEIGHTS[stage]
    return round(sum(value * w[d] for d in QUALITY_DIMS) + 100 * w["efficiency"], 2)


def test_compare_builds_scorecard_between_agents():
    """compare 汇总各 agent 总分 + 各维均值，供跑分卡排序。"""
    cases = [_case(stage="opening", name="c1"), _case(stage="long", name="c2")]

    high_runner = BenchmarkRunner(FakeJudge(_all_dims(90)))
    low_runner = BenchmarkRunner(FakeJudge(_all_dims(60)))
    high = FakeAgent("high", meta=_meta(elapsed=30))
    low = FakeAgent("low", meta=_meta(elapsed=30))

    high_results = [_run(high_runner.run_case(high, c, repeat=1)) for c in cases]
    low_results = [_run(low_runner.run_case(low, c, repeat=1)) for c in cases]

    comp = high_runner.compare(high_results + low_results)

    assert comp.agents == ["high", "low"]
    assert set(comp.dims) == set(NINE_DIMS)
    assert comp.overall_mean["high"] > comp.overall_mean["low"]
    assert comp.dims_mean["high"]["consistency"] == 90
    assert comp.dims_mean["low"]["consistency"] == 60
    # efficiency=100 使各 stage 的 overall 略高于质量维分，跨 case 有波动 → 有 std
    exp_high = [_expected_overall(90, c.stage) for c in cases]
    exp_low = [_expected_overall(60, c.stage) for c in cases]
    assert comp.overall_mean["high"] == round(statistics.fmean(exp_high), 3)
    assert comp.overall_std["high"] == round(statistics.stdev(exp_high), 3)
    assert comp.overall_mean["low"] == round(statistics.fmean(exp_low), 3)
    assert comp.ranking() == ["high", "low"]
