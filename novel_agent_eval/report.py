# novel_agent_eval/report.py
"""跑分卡生成 — 把 BenchmarkReport / 消融结果渲染成 Markdown 跑分卡与 JSON。

方案 §6.2 模板逐节渲染（纯函数，无 IO，可离线用构造的 report 测试）：
  - 总分表：按 agent 聚合，调 BenchmarkRunner.compare（不重复实现）取各维
    mean±std 与加权 overall。
  - 内部信号 vs 外部一致性：按 stage 收集各 run 的 composite_score 与外部
    overall，算均值与 Pearson 相关。
  - 效率表：elapsed / tokens / evolution_rounds / 终止原因分布。
  - 可操控性专项分析 / Judge 校准记录：占位节（需人工数据，当前 runner 未采集）。
"""
import dataclasses
import json
from collections import Counter
from statistics import fmean

from novel_agent_eval.judge import QUALITY_DIMS
from novel_agent_eval.runner import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    _normalize_tokens,
)

# 英文维度键 → 中文标签（9 维全表；总分表只用 8 质量维，效率列不进表）
DIM_LABELS = {
    "consistency": "连贯性",
    "writing": "文笔",
    "ai_flavor": "AI味",
    "dialogue": "对话",
    "plot": "情节",
    "instruction": "指令遵循",
    "creativity": "创意",
    "controllability": "可操控",
    "efficiency": "效率",
}

# 8 质量维的中文表头（与方案 §6.2 跑分卡列序一致）
QUALITY_DIM_LABELS = [DIM_LABELS[d] for d in QUALITY_DIMS]

# stage → 中文标签（方案：开局 1-10 章 / 中段 30-50 / 长程 80-100）
STAGE_LABELS = {"opening": "开局", "middle": "中段", "long": "长程"}


def _pearson(xs, ys):
    """Pearson 相关系数（标准公式，无第三方依赖）。样本 <2 或方差为 0 时返回 None。"""
    n = len(xs)
    if n < 2:
        return None
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """把表头 + 行渲染成 Markdown 表格（含分隔行）。"""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def render_scorecard(report: BenchmarkReport) -> str:
    """渲染完整跑分卡 Markdown。总分聚合复用 BenchmarkRunner.compare。"""
    comp = BenchmarkRunner.compare(report.results)
    sections: list[str] = []

    # ── 头部元信息 ────────────────────────────────
    sections.append("## novel-agent 跑分卡")
    sections.append("")
    sections.append("**评测日期**: —")
    sections.append(f"**重复次数**: {report.repeat}")
    sections.append(f"**Agent 数**: {len(report.agents)}")
    sections.append(f"**案例数**: {len(report.cases)}")

    # ── 总分表（8 质量维 + 总分，不含 efficiency） ──
    sections.append("")
    sections.append("### 总分")
    sections.append("")
    headers = ["Agent", *QUALITY_DIM_LABELS, "**总分**"]
    rows = []
    for agent in comp.agents:
        cells = [
            f"{comp.dims_mean[agent][d]:.0f}±{comp.dims_std[agent][d]:.0f}"
            for d in QUALITY_DIMS
        ]
        cells.append(f"{comp.overall_mean[agent]:.1f}")
        rows.append([agent, *cells])
    sections.append(_md_table(headers, rows))

    # ── 内部信号 vs 外部一致性（按 stage） ──────────
    sections.append("")
    sections.append("### 内部信号 vs 外部评测一致性")
    sections.append("")
    sections.append("| 指标 | 内部 composite_score | 外部 Judge 加权总分 | 相关系数 |")
    sections.append("|------|---------------------|--------------------|---------|")
    by_stage = {s: [] for s in STAGE_LABELS}
    for r in report.results:
        for run in r.runs:
            if r.stage in by_stage:
                by_stage[r.stage].append(run)
    for stage in STAGE_LABELS:
        label = f"{STAGE_LABELS[stage]}案例均值"
        # 只取 composite_score 非 None 的 run；内部/外部均值与相关系数同用该子集，
        # 保证三列同口径（与"跳过 None"一致，且相关样本与均值样本相同）。
        valid = [run for run in by_stage[stage] if run.meta.get("composite_score") is not None]
        if not valid:
            sections.append(f"| {label} | — | — | — |")
            continue
        internal = fmean(run.meta["composite_score"] for run in valid)
        external = fmean(run.overall for run in valid)
        rho = _pearson(
            [run.meta["composite_score"] for run in valid],
            [run.overall for run in valid],
        )
        corr = f"{rho:.3f}" if rho is not None else "—"
        sections.append(f"| {label} | {internal:.1f} | {external:.1f} | {corr} |")

    # ── 效率表（per agent；无该信号的列渲染 —） ──────
    sections.append("")
    sections.append("### 效率")
    sections.append("")
    headers = ["Agent", "单章耗时(s)", "token", "进化轮次", "终止原因分布"]
    rows = []
    for agent in comp.agents:
        runs = [run for r in report.results if r.agent == agent for run in r.runs]
        elapsed = [
            run.meta["elapsed_seconds"]
            for run in runs
            if run.meta.get("elapsed_seconds") is not None
        ]
        tokens = [
            _normalize_tokens(run.meta["tokens"])
            for run in runs
            if run.meta.get("tokens") is not None
        ]
        rounds = [
            run.meta["evolution_rounds"]
            for run in runs
            if run.meta.get("evolution_rounds") is not None
        ]
        terms = [
            run.meta["evolution_termination"]
            for run in runs
            if run.meta.get("evolution_termination")
        ]
        dist = (
            ", ".join(f"{k}×{v}" for k, v in Counter(terms).most_common()) if terms else "—"
        )
        rows.append([
            agent,
            f"{fmean(elapsed):.1f}" if elapsed else "—",
            f"{fmean(tokens):.0f}" if tokens else "—",
            f"{fmean(rounds):.1f}" if rounds else "—",
            dist,
        ])
    sections.append(_md_table(headers, rows))

    # ── 占位节：需人工/HITL 数据，当前 runner 未采集 ──
    sections.append("")
    sections.append("### 可操控性专项分析")
    sections.append("")
    sections.append("待采集（需 HITL 拒绝数据，当前 runner 未采集）")
    sections.append("")
    sections.append("### Judge 校准记录")
    sections.append("")
    sections.append("待采集（需人工盲测 ground truth）")

    sections.append("")
    return "\n".join(sections)


def render_ablation(ablation: dict[str, BenchmarkResult]) -> str:
    """渲染消融分析表：每配置一行，Δ vs 完整（baseline）为带符号差值。"""
    baseline = ablation.get("baseline")
    base_overall = baseline.overall_mean if baseline is not None else None

    headers = ["配置", *QUALITY_DIM_LABELS, "**总分**", "Δ vs 完整"]
    rows = []
    for name, res in ablation.items():
        cells = [f"{res.dims_mean[d]:.0f}" for d in QUALITY_DIMS]
        cells.append(f"{res.overall_mean:.1f}")
        if base_overall is None:
            cells.append("—")
        else:
            delta = res.overall_mean - base_overall
            # baseline 自身 Δ=0.0 不带符号；其余带 +/−
            cells.append("0.0" if abs(delta) < 0.05 else f"{delta:+.1f}")
        rows.append([name, *cells])

    return "\n".join([
        "## 消融分析",
        "",
        _md_table(headers, rows),
        "",
    ])


def render_json(report: BenchmarkReport) -> str:
    """把 report 序列化为 JSON 字符串（asdict 递归展开嵌套 CaseRun/meta）。"""
    return json.dumps(
        dataclasses.asdict(report),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
