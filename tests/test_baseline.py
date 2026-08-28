# tests/test_baseline.py
"""Phase 2 baseline 指标 + Go/No-Go 门槛 测试（纯逻辑，不依赖 LLM / graph）。"""
from statistics import fmean

from novel_agent_eval.baseline import (
    CHAPTER_COMPLETED,
    CHAPTER_FAILED,
    CHAPTER_INVALID,
    CHAPTER_RESUMED,
    CHAPTER_TIMEOUT,
    BaselineRunResult,
    ChapterScore,
    aggregate_reliability,
    go_no_go_gates,
)


def _ch(n, status=CHAPTER_COMPLETED, *, overall=80.0, errors=0, tokens=1000, latency=10.0, **kw) -> ChapterScore:
    return ChapterScore(
        chapter_number=n, status=status, overall=overall,
        consistency_errors=errors, token_usage={"total_tokens": tokens},
        latency_seconds=latency, **kw
    )


def _run(chapters, expected=20, agent="na", prompt_id="p1", sample=0) -> BaselineRunResult:
    return BaselineRunResult(
        agent=agent, prompt_id=prompt_id, sample_index=sample,
        expected_chapters=expected, chapters=chapters,
    )


# ── Reliability ──


def test_full_run_success_binary():
    full = _run([_ch(i) for i in range(1, 21)])
    partial = _run([_ch(i) for i in range(1, 19)] + [_ch(19, CHAPTER_FAILED, failure_stage="generation")])
    assert full.full_run_success == 1.0
    assert partial.full_run_success == 0.0
    assert partial.chapter_completion_rate == 0.9


def test_timeout_invalid_censorship_rates():
    chapters = [
        _ch(1, CHAPTER_COMPLETED),
        _ch(2, CHAPTER_TIMEOUT, failure_stage="timeout", overall=None),
        _ch(3, CHAPTER_INVALID, failure_stage="generation", failure_reason="empty content"),
        _ch(4, CHAPTER_INVALID, failure_stage="generation", failure_reason="content policy refusal"),
    ]
    r = _run(chapters, expected=4)
    assert r.timeout_rate == 0.25
    assert r.invalid_rate == 0.5
    assert r.censorship_rate == 0.25  # 仅 policy refusal 算 censorship
    assert chapters[2].is_censorship is False  # empty → invalid 但非 censorship
    assert chapters[3].is_censorship is True  # policy refusal → censorship


def test_resume_success_rate():
    r = _run([_ch(1, CHAPTER_RESUMED), _ch(2, CHAPTER_RESUMED), _ch(3)])
    assert r.resume_success_rate == 1.0


# ── Quality ──


def test_quality_windows_degradation_trend():
    chapters = [_ch(i, overall=s) for i, s in enumerate([80, 82, 78, 76, 70, 68], start=1)]
    r = _run(chapters, expected=6)
    assert r.quality_mean == round((80 + 82 + 78 + 76 + 70 + 68) / 6, 3)
    assert r.first_window_score == 81.0  # (80+82)/2
    assert r.last_window_score == 69.0  # (70+68)/2
    assert r.degradation == round(69.0 - 81.0, 3)  # 负值=衰减
    assert r.trend_slope < 0  # 下降趋势


def test_quality_none_when_no_scores():
    chapters = [_ch(1, CHAPTER_FAILED, overall=None), _ch(2, CHAPTER_TIMEOUT, overall=None)]
    r = _run(chapters, expected=2)
    assert r.quality_mean is None
    assert r.degradation is None
    assert r.trend_slope is None


# ── Consistency ──


def test_ced_and_error_growth():
    chapters = [_ch(i, errors=e) for i, e in enumerate([0, 1, 2, 3, 4, 5], start=1)]
    r = _run(chapters, expected=6)
    assert r.ced == round((0 + 1 + 2 + 3 + 4 + 5) / 6, 3)
    assert r.first_error_density == 0.5  # (0+1)/2
    assert r.last_error_density == 4.5  # (4+5)/2
    assert r.error_growth_slope > 0  # 错误随连载增长


# ── Cost ──


def test_cost_aggregation():
    chapters = [_ch(i, tokens=1000 * i, latency=10.0 + i) for i in range(1, 4)]
    r = _run(chapters, expected=3)
    assert r.total_tokens == 1000 + 2000 + 3000
    assert r.tokens_per_chapter == 2000.0
    assert r.latency_per_chapter == round((11 + 12 + 13) / 3, 3)


def test_cost_excludes_failed_from_per_chapter():
    chapters = [_ch(1, tokens=1000), _ch(2, CHAPTER_FAILED, tokens=0, latency=0.0)]
    r = _run(chapters, expected=2)
    assert r.total_tokens == 1000  # 全部累加
    assert r.tokens_per_chapter == 1000.0  # 仅成功章
    assert r.latency_per_chapter == 10.0


# ── 跨 run 聚合 + Go/No-Go ──


def test_aggregate_reliability_across_runs():
    runs = [
        _run([_ch(i) for i in range(1, 21)]),  # full success
        _run([_ch(i) for i in range(1, 19)] + [
            _ch(19, CHAPTER_TIMEOUT, failure_stage="timeout", overall=None),
            _ch(20, CHAPTER_COMPLETED),
        ]),  # 19/20 completed, 1 timeout
    ]
    rel = aggregate_reliability(runs)
    assert rel["full_run_success_rate"] == 0.5  # 1 full / 2 runs
    assert rel["chapter_completion_rate"] == round(39 / 40, 3)
    assert rel["timeout_rate"] == round((0.0 + 0.05) / 2, 3)  # run2: 1/20 timeout


def test_go_no_go_pass():
    runs = [_run([_ch(i) for i in range(1, 21)]) for _ in range(3)]
    gates = go_no_go_gates(runs)
    assert gates["full_run_success_rate"] == 1.0
    assert gates["chapter_completion_rate"] == 1.0
    assert gates["failure_stage_classifiable"] is True
    assert gates["checkpoint_recoverable"] is True
    assert gates["passed"] is True


def test_go_no_go_fail_on_success_rate():
    runs = [
        _run([_ch(i) for i in range(1, 21)]),
        _run([_ch(i) for i in range(1, 10)] + [_ch(j, CHAPTER_FAILED, failure_stage="generation") for j in range(10, 21)]),
    ]
    gates = go_no_go_gates(runs)
    assert gates["full_run_success_rate"] == 0.5  # < 0.95
    assert gates["passed"] is False


def test_go_no_go_fail_on_unclassifiable_failure_stage():
    runs = [_run([_ch(i) for i in range(1, 21)] + [
        _ch(21, CHAPTER_FAILED, failure_stage=None),  # 不可分类
    ] * 0)]  # 实际构造：1 个 failed 无 stage
    runs = [_run([_ch(i) for i in range(1, 21)] + [
        ChapterScore(chapter_number=21, status=CHAPTER_FAILED, failure_stage=None),
    ], expected=21)]
    gates = go_no_go_gates(runs)
    assert gates["failure_stage_classifiable"] is False
    assert gates["passed"] is False


# ── C-5 分段 / C-6 三类 degradation / C-8 quality-per-cost ──


def test_segment_stats_windows():
    chapters = [_ch(i, overall=60.0 + i) for i in range(1, 21)]
    r = _run(chapters, expected=20)
    segs = r.segment_stats(window=5)
    assert len(segs) == 4
    assert segs[0]["segment"] == "1-5"
    assert segs[0]["mean"] == round(fmean([61, 62, 63, 64, 65]), 3)
    assert segs[3]["segment"] == "16-20"


def test_cost_and_context_growth_slopes():
    chapters = []
    for i in range(1, 6):
        c = _ch(i, tokens=1000 * i)
        c.context_sizes = {"writer_view": {"character_context_chars": 100 * i, "recent_summary_chars": 0, "world_context_chars": 0}}
        chapters.append(c)
    r = _run(chapters, expected=5)
    assert r.cost_growth_slope == 1000.0
    assert r.context_growth_slope == 100.0


def test_quality_per_cost():
    chapters = [_ch(i, overall=80.0, tokens=2_000_000) for i in range(1, 3)]
    r = _run(chapters, expected=2)
    qpc = r.quality_per_cost
    assert qpc["quality_mean"] == 80.0
    assert qpc["tokens_per_chapter"] == 2_000_000.0
    assert qpc["quality_per_million_tokens"] == 40.0


def test_evolution_gain_per_revision():
    path = [
        {"revision": 0, "composite": 70.0, "writer_tokens": {"input": 500, "output": 1000}},
        {"revision": 1, "composite": 75.0, "focus": ["dialogue"], "writer_tokens": {"input": 1500, "output": 2000}},
    ]
    c = _ch(1)
    c.evolution_path = path
    r = _run([c], expected=1)
    gains = r.evolution_gain_per_revision()
    assert len(gains) == 1
    g = gains[0]
    assert g["chapter"] == 1
    assert g["quality_gain"] == 5.0
    assert g["writer_input_delta"] == 1000
    assert g["writer_output_delta"] == 1000
    assert g["focus"] == ["dialogue"]
