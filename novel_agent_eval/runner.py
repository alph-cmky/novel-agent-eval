# novel_agent_eval/runner.py
"""横评 / 消融 / 多轮重复编排器 — 把 AgentAdapter（Task 7）+ Judge（Task 6）+
weighted_score / efficiency_score（Task 5）串成完整评测流程。

评分合并：Judge 只打 8 个质量维；第 9 维「效率」由 efficiency_score 从
GeneratedChapter.meta 算（elapsed_seconds / tokens / evolution_rounds）。每次生成：
  1. js = judge.score(draft, case)                     → 8 质量维
  2. eff = efficiency_score(elapsed_seconds, tokens, evolution_rounds)
  3. scores = {**js.dimensions, "efficiency": eff}     → 合成 9 维
  4. overall = weighted_score(scores, case.stage)       → 按连载阶段加权
repeat 次后聚合各维 / overall 的均值±标准差。

数据结构（方案 §6.3）：
  - CaseRun         单次生成的 per-run 结果（9 维分 + overall + meta）
  - BenchmarkResult 单个 case 的聚合结果（各维 mean/std + overall mean/std + runs）
  - BenchmarkReport run_suite 完整结果（agent × case 矩阵 + repeat + 汇总）
  - ComparisonReport compare 结果（agent 间总分对比 + 各维对比，供跑分卡）
"""
from dataclasses import dataclass, field
from statistics import fmean, stdev
from typing import Any

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.judge import Judge, JudgeScore
from novel_agent_eval.metrics import efficiency_score, weighted_score

EFFICIENCY_DIM = "efficiency"
STD_ZERO_WHEN_N1 = "repeat<2 时无样本方差，std 记 0.0"


# ── 数据结构 ────────────────────────────────────────────


@dataclass
class CaseRun:
    """单次生成的 per-run 结果：9 维分 + overall + 生成 meta（含效率信号）。"""

    run_index: int
    dimensions: dict[str, int]   # 9 维 = 8 质量维 + efficiency，各 0-100
    overall: float               # weighted_score(scores, case.stage)
    meta: dict[str, Any]         # 生成 meta（elapsed/tokens/evolution_rounds 等）


@dataclass
class BenchmarkResult:
    """单个 case 的聚合结果：repeat 次运行的各维 mean/std + overall mean/std + runs。"""

    agent: str
    case: str
    stage: str
    repeat: int
    dims_mean: dict[str, float]
    dims_std: dict[str, float]
    overall_mean: float
    overall_std: float
    runs: list[CaseRun] = field(default_factory=list)   # 每次运行的 per-run 结果

    @property
    def dims(self) -> list[str]:
        return list(self.dims_mean)


@dataclass
class BenchmarkReport:
    """run_suite 完整结果：agent × case 矩阵 + repeat + 汇总。"""

    results: list[BenchmarkResult]
    repeat: int
    agents: list[str]
    cases: list[str]

    def matrix(self) -> dict[str, dict[str, BenchmarkResult]]:
        """agent -> case -> BenchmarkResult 二维矩阵。"""
        m: dict[str, dict[str, BenchmarkResult]] = {}
        for r in self.results:
            m.setdefault(r.agent, {})[r.case] = r
        return m

    def agent_overall(self) -> dict[str, float]:
        """每个 agent 跨全部 case 的 overall 均值（横评总览）。"""
        by_agent: dict[str, list[float]] = {}
        for r in self.results:
            by_agent.setdefault(r.agent, []).append(r.overall_mean)
        return {a: round(fmean(v), 3) for a, v in by_agent.items()}


@dataclass
class ComparisonReport:
    """compare 结果：agent 间总分对比 + 各维对比（供跑分卡）。"""

    agents: list[str]                       # 保持输入结果出现顺序
    dims: list[str]                         # 9 维
    repeat: int
    overall_mean: dict[str, float]          # agent -> 跨结果 overall 均值
    overall_std: dict[str, float]           # agent -> 跨结果 overall 样本标准差
    dims_mean: dict[str, dict[str, float]]  # agent -> dim -> 均值
    dims_std: dict[str, dict[str, float]]   # agent -> dim -> 样本标准差

    def ranking(self) -> list[str]:
        """按 overall_mean 降序的 agent 排行。"""
        return sorted(self.agents, key=lambda a: self.overall_mean[a], reverse=True)


# ── 效率维度 helpers ────────────────────────────────────


def _normalize_tokens(tokens) -> int:
    """meta.tokens 归一化为 int：None→0；dict{input,output}→求和；int/str 原样转 int。

    efficiency_score 当前未在公式中使用该参数，仅保持类型稳定（防 dict 传入报错）。
    """
    if tokens is None:
        return 0
    if isinstance(tokens, dict):
        vals = [t for t in tokens.values() if isinstance(t, (int, float))]
        return int(sum(vals))
    try:
        return int(tokens)
    except (TypeError, ValueError):
        return 0


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = fmean(vals)
    std = stdev(vals) if len(vals) > 1 else 0.0
    return round(mean, 3), round(std, 3)


# ── BenchmarkRunner ─────────────────────────────────────

# 消融配置注册表：名称 → {desc, factory}。baseline 为完整进化对照。
# 当前仅 evolution_enabled 可落地（关闭进化走线性闭环）；其余需主仓库加细粒度开关后实现，
# 见 run_ablation 与 _NOT_IMPLEMENTED_ABLATIONS。
_ABLATION_REGISTRY: dict[str, dict[str, Any]] = {
    "baseline": {
        "desc": "完整进化流水线（对照基线）",
        "factory": lambda: NovelAgentAdapter(evolution_enabled=True),
    },
    "evolution_enabled": {
        "desc": "关闭进化：Writer→Editor→Continuity 线性闭环（legacy 路径）",
        "factory": lambda: NovelAgentAdapter(evolution_enabled=False),
    },
}

# 主仓库 build_chapter_graph_async(persist_dir, evolution_enabled) 目前只有
# evolution_enabled 一个开关（见 novel_agent/graph/chapter.py），以下消融需主仓库
# 加「无 Continuity / 无 Worldbuilding / 无 EvoOrchestrator LLM 增强」等细粒度开关后实现。
_NOT_IMPLEMENTED_ABLATIONS = (
    "no_continuity",
    "no_worldbuilding",
    "no_orchestrator_llm",
)


class BenchmarkRunner:
    """横评编排器：串 AgentAdapter + Judge + weighted_score。

    judge 为必须注入的 Judge（或 mock）；repeat 为方法缺省重复次数。
    """

    def __init__(self, judge: Judge, repeat: int = 3):
        self._judge = judge
        self._repeat = repeat

    # -- run_case：单 case × repeat 次 → 聚合均值±标准差 --

    async def run_case(self, agent, case, repeat: int | None = None) -> BenchmarkResult:
        """跑单个 case repeat 次，聚合各维与 overall 的均值±标准差。

        注：方案 brief 草拟签名为 `-> list[BenchmarkResult]`，但任务注记 #4 定义
        BenchmarkResult 即聚合结果（各维 mean/std + overall mean/std + runs），
        单 case 单配置即一个聚合对象，故返回单对象而非单元素 list。
        """
        n = self._repeat if repeat is None else repeat
        runs = [await self._run_once(agent, case, i) for i in range(n)]
        return self._aggregate(agent, case, runs)

    async def _run_once(self, agent, case, run_index: int) -> CaseRun:
        """单次生成 → Judge 8 质量维 + efficiency 第 9 维 → weighted_score 合成 overall。"""
        gen = await agent.generate(case)
        js: JudgeScore = await self._judge.score(gen.content, case)
        eff = efficiency_score(
            gen.meta.get("elapsed_seconds", 0.0),
            _normalize_tokens(gen.meta.get("tokens")),
            gen.meta.get("evolution_rounds", 0),
        )
        dims = {**js.dimensions, EFFICIENCY_DIM: eff}
        overall = weighted_score(dims, case.stage)
        return CaseRun(
            run_index=run_index,
            dimensions=dims,
            overall=overall,
            meta=gen.meta,
        )

    @staticmethod
    def _aggregate(agent, case, runs: list[CaseRun]) -> BenchmarkResult:
        dims = list(runs[0].dimensions) if runs else []
        dims_mean = {}
        dims_std = {}
        for d in dims:
            vals = [float(r.dimensions[d]) for r in runs]
            dims_mean[d], dims_std[d] = _mean_std(vals)
        overalls = [r.overall for r in runs]
        overall_mean, overall_std = _mean_std(overalls)
        return BenchmarkResult(
            agent=agent.name,
            case=case.name,
            stage=case.stage,
            repeat=len(runs),
            dims_mean=dims_mean,
            dims_std=dims_std,
            overall_mean=overall_mean,
            overall_std=overall_std,
            runs=runs,
        )

    # -- run_ablation：消融配置对比 --

    async def run_ablation(
        self,
        case,
        modules: list[str],
        repeat: int | None = None,
    ) -> dict[str, BenchmarkResult]:
        """跑基线（完整进化）+ 每个消融配置，返回 {配置名: BenchmarkResult}。

        modules 为配置名列表；当前可识别 "evolution_enabled"（→ NovelAgentAdapter
        (evolution_enabled=False)）。其余消融（无 Continuity / Worldbuilding /
        EvoOrchestrator LLM 增强）在主仓库加细粒度开关前明确 raise NotImplementedError，
        不假装实现。
        """
        n = self._repeat if repeat is None else repeat
        results: dict[str, BenchmarkResult] = {}

        # 先校验全部消融名，再跑基线 —— 未实现/未知配置在任何 run_case（付费生成）前即 raise。
        for name in modules:
            cfg = _ABLATION_REGISTRY.get(name)
            if cfg is None:
                extra = (
                    f"（已知未实现：{_NOT_IMPLEMENTED_ABLATIONS}）"
                    if name in _NOT_IMPLEMENTED_ABLATIONS
                    else "（未知配置）"
                )
                raise NotImplementedError(
                    f"消融配置 {name!r} 未实现{extra}：主仓库 build_chapter_graph_async "
                    f"目前只有 evolution_enabled 一个开关，需主仓库加细粒度开关后实现"
                )

        baseline_cfg = _ABLATION_REGISTRY["baseline"]
        results["baseline"] = await self.run_case(baseline_cfg["factory"](), case, n)

        for name in modules:
            results[name] = await self.run_case(_ABLATION_REGISTRY[name]["factory"](), case, n)
        return results

    # -- run_suite：agent × case 两重循环 --

    async def run_suite(self, agents, cases, repeat: int | None = None) -> BenchmarkReport:
        """agent × case 全遍历，每 (agent, case) 跑 repeat 次并聚合。"""
        n = self._repeat if repeat is None else repeat
        results = []
        for agent in agents:
            for case in cases:
                results.append(await self.run_case(agent, case, n))
        return BenchmarkReport(
            results=results,
            repeat=n,
            agents=[a.name for a in agents],
            cases=[c.name for c in cases],
        )

    # -- compare：跑分卡 --

    @staticmethod
    def compare(results: list[BenchmarkResult]) -> ComparisonReport:
        """把多个 BenchmarkResult（如 run_suite.results 或 run_ablation 的 dict values）
        按 agent 聚合：总分 + 各维均值，供跑分卡排序。"""
        by_agent: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_agent.setdefault(r.agent, []).append(r)
        agents = list(by_agent.keys())
        dims = list(results[0].dims_mean) if results else []
        repeat = results[0].repeat if results else 0

        overall_mean, overall_std = {}, {}
        dims_mean, dims_std = {}, {}
        for agent, rs in by_agent.items():
            overall_mean[agent], overall_std[agent] = _mean_std([r.overall_mean for r in rs])
            dims_mean[agent], dims_std[agent] = {}, {}
            for d in dims:
                vals = [float(r.dims_mean[d]) for r in rs]
                dims_mean[agent][d], dims_std[agent][d] = _mean_std(vals)
        return ComparisonReport(
            agents=agents,
            dims=dims,
            repeat=repeat,
            overall_mean=overall_mean,
            overall_std=overall_std,
            dims_mean=dims_mean,
            dims_std=dims_std,
        )
