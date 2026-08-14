# tests/test_report.py
"""report.py 跑分卡渲染测试（全离线，手造 report，不跑 LLM / graph）。

- render_scorecard：总分表 8 质量维中文表头 + 总分 + mean±std 单元格；
  内部 vs 外部一致性表相关系数（composite_score=None 跳过不崩）；效率表。
- render_ablation：Δ vs 完整 列符号正确。
- render_json：json.loads 可解析且含 results 键。
"""
import json
import statistics

from novel_agent_eval.judge import QUALITY_DIMS
from novel_agent_eval.report import DIM_LABELS, render_ablation, render_json, render_scorecard
from novel_agent_eval.runner import BenchmarkReport, BenchmarkResult, CaseRun


def _dims(**overrides) -> dict[str, int]:
    d = {k: 80 for k in QUALITY_DIMS}
    d.update(overrides)
    return d


def _run(index: int, meta: dict, overall: float = 80.0, dims: dict | None = None) -> CaseRun:
    return CaseRun(run_index=index, dimensions=dims or _dims(), overall=float(overall), meta=meta)


def _result(agent, case, stage, runs, dims_mean=None, dims_std=None,
            overall_mean=None, overall_std=None) -> BenchmarkResult:
    dims = list(runs[0].dimensions)
    if dims_mean is None:
        dims_mean = {d: statistics.fmean(r.dimensions[d] for r in runs) for d in dims}
    if dims_std is None:
        dims_std = {
            d: (statistics.stdev([r.dimensions[d] for r in runs]) if len(runs) > 1 else 0.0)
            for d in dims
        }
    if overall_mean is None:
        overall_mean = statistics.fmean(r.overall for r in runs)
    if overall_std is None:
        overall_std = statistics.stdev([r.overall for r in runs]) if len(runs) > 1 else 0.0
    return BenchmarkResult(
        agent=agent, case=case, stage=stage, repeat=len(runs),
        dims_mean=dims_mean, dims_std=dims_std,
        overall_mean=overall_mean, overall_std=overall_std, runs=runs,
    )


# ── render_scorecard ───────────────────────────────────


def test_scorecard_contains_headers_and_sections():
    """输出含 8 质量维中文表头 + 总分标记 + 内部/外部一致性节头 + 顶部元信息。"""
    run = _run(0, {"elapsed_seconds": 30.0, "tokens": 100, "evolution_rounds": 0,
                   "evolution_termination": "converged", "composite_score": 70.0})
    res = _result("novel_agent", "opening_01", "opening", [run])
    report = BenchmarkReport(results=[res], repeat=3, agents=["novel_agent"], cases=["opening_01"])

    md = render_scorecard(report)

    assert "## novel-agent 跑分卡" in md
    assert "**重复次数**: 3" in md
    assert "**Agent 数**: 1" in md
    assert "**案例数**: 1" in md
    for label in ["连贯性", "文笔", "AI味", "对话", "情节", "指令遵循", "创意", "可操控"]:
        assert label in md
    assert "**总分**" in md
    assert "内部信号 vs 外部评测一致性" in md
    assert "### 效率" in md


def test_scorecard_renders_mean_plus_std_cell():
    """compare 重新聚合后 consistency 85.4±3.2 → 单元格渲染为 85±3。

    两个结果的 dims_mean 取 85.4 ± 3.2/√2，两样本样本标准差 = |diff|/√2 = 3.2。
    """
    v1 = 85.4 - 3.2 / 2 ** 0.5
    v2 = 85.4 + 3.2 / 2 ** 0.5
    run = _run(0, {"elapsed_seconds": 30.0})
    r1 = _result("novel_agent", "c1", "opening", [run],
                 dims_mean={**{d: 80.0 for d in QUALITY_DIMS}, "consistency": v1},
                 dims_std={d: 0.0 for d in QUALITY_DIMS},
                 overall_mean=80.0, overall_std=0.0)
    r2 = _result("novel_agent", "c2", "opening", [run],
                 dims_mean={**{d: 80.0 for d in QUALITY_DIMS}, "consistency": v2},
                 dims_std={d: 0.0 for d in QUALITY_DIMS},
                 overall_mean=80.0, overall_std=0.0)
    report = BenchmarkReport(results=[r1, r2], repeat=1, agents=["novel_agent"], cases=["c1", "c2"])

    md = render_scorecard(report)
    row = [ln for ln in md.splitlines() if ln.startswith("| novel_agent |")][0]

    assert "85±3" in row
    assert "80±0" in row


def test_internal_external_table_correlation_and_skip_none():
    """内部/外部一致性表：composite_score 已知的 run 算相关；None 的 run 跳过不崩。

    有效对 (80,82)/(90,88)/(70,75) → 内部 80.0 / 外部 81.7 / 相关系数 0.999。
    无 run 的 stage 渲染 —。
    """
    runs = [
        _run(0, {"composite_score": 80.0}, overall=82.0),
        _run(1, {"composite_score": 90.0}, overall=88.0),
        _run(2, {"composite_score": None}, overall=60.0),  # 无效，应跳过
        _run(3, {"composite_score": 70.0}, overall=75.0),
    ]
    res = _result("novel_agent", "opening_01", "opening", runs,
                  dims_mean={d: 80.0 for d in QUALITY_DIMS},
                  dims_std={d: 0.0 for d in QUALITY_DIMS},
                  overall_mean=81.25, overall_std=5.0)
    report = BenchmarkReport(results=[res], repeat=4, agents=["novel_agent"], cases=["opening_01"])

    md = render_scorecard(report)

    assert "| 开局案例均值 | 80.0 | 81.7 | 0.999 |" in md
    assert "| 中段案例均值 | — | — | — |" in md
    assert "| 长程案例均值 | — | — | — |" in md


def test_efficiency_table_handles_missing_meta():
    """效率表：有 elapsed/tokens 的渲染均值；无进化 meta 的列渲染 —。"""
    run = _run(0, {"elapsed_seconds": 12.0, "tokens": {"input": 1000, "output": 2000}})
    res = _result("vanilla_llm", "opening_01", "opening", [run])
    report = BenchmarkReport(results=[res], repeat=1, agents=["vanilla_llm"], cases=["opening_01"])

    md = render_scorecard(report)

    assert "| vanilla_llm | 12.0 | 3000 | — | — |" in md


# ── render_ablation ────────────────────────────────────


def test_ablation_delta_sign():
    """消融表：比 baseline 差的配置 Δ 为负；baseline 自身 Δ 为 0.0。"""
    run = _run(0, {"elapsed_seconds": 30.0})
    base = _result("novel_agent", "c", "opening", [run],
                   dims_mean={d: 80.0 for d in QUALITY_DIMS},
                   dims_std={d: 0.0 for d in QUALITY_DIMS},
                   overall_mean=78.5, overall_std=0.0)
    ablated = _result("novel_agent", "c", "opening", [run],
                      dims_mean={d: 70.0 for d in QUALITY_DIMS},
                      dims_std={d: 0.0 for d in QUALITY_DIMS},
                      overall_mean=70.0, overall_std=0.0)
    ablation = {"baseline": base, "evolution_enabled": ablated}

    md = render_ablation(ablation)

    assert "Δ vs 完整" in md
    assert "| baseline |" in md
    assert "| evolution_enabled |" in md
    assert "-8.5" in md
    assert "0.0" in md  # baseline 自身 Δ = 0.0


# ── render_json ────────────────────────────────────────


def test_render_json_parses():
    """render_json 输出可 json.loads，含 results 键与嵌套 CaseRun/meta。"""
    run = _run(0, {"composite_score": 80.0, "elapsed_seconds": 30.0,
                   "tokens": {"input": 100, "output": 200},
                   "evolution_rounds": 2, "evolution_termination": "converged"})
    res = _result("novel_agent", "c1", "opening", [run])
    report = BenchmarkReport(results=[res], repeat=1, agents=["novel_agent"], cases=["c1"])

    text = render_json(report)
    data = json.loads(text)

    assert "results" in data
    assert data["repeat"] == 1
    assert data["results"][0]["agent"] == "novel_agent"
    assert data["results"][0]["runs"][0]["meta"]["composite_score"] == 80.0
    assert data["results"][0]["runs"][0]["meta"]["tokens"]["output"] == 200


def test_dim_labels_mapping():
    """DIM_LABELS 覆盖 9 维中文映射（供所有表头复用）。"""
    assert DIM_LABELS["consistency"] == "连贯性"
    assert DIM_LABELS["efficiency"] == "效率"
    assert len(DIM_LABELS) == 9
