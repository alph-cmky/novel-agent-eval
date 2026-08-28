# novel_agent_eval/baseline.py
"""Phase 2 稳定 20 章 diagnostic baseline 的指标与 Go/No-Go 门槛。

四类指标（§2.2）：
- Reliability：full_run_success / chapter_completion / timeout / invalid / censorship / resume
- Quality：mean/std + first/middle/last + degradation + trend_slope
- Consistency：CED（errors/chapter）+ error_growth_slope + first/mid/last error density
- Cost：total_tokens / tokens_per_chapter / latency_per_chapter

Go/No-Go（进入 30 章主实验前）：
  full_run_success_rate >= 95%  且  chapter_completion >= 98%
  且 failure_stage 可分类  且  checkpoint 可恢复

纯逻辑层：从 per-chapter ChapterScore 聚合，不依赖 LLM / graph，可单测。
"""
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev

# 失败语义（对齐 durable.py）
CHAPTER_COMPLETED = "completed"
CHAPTER_RESUMED = "resumed"
CHAPTER_PARTIAL = "partial"
CHAPTER_FAILED = "failed"
CHAPTER_TIMEOUT = "timeout"
CHAPTER_INVALID = "invalid"

_SUCCESS_STATES = {CHAPTER_COMPLETED, CHAPTER_RESUMED}


@dataclass
class ChapterScore:
    """单章 baseline 评分：质量 + 一致性 + 成本 + 失败语义 + C-3/C-7 观测。"""

    chapter_number: int
    status: str
    overall: float | None = None  # weighted_score（8 质量维 + efficiency）
    dimensions: dict[str, int] = field(default_factory=dict)
    consistency_score: int | None = None  # ConStory 0-100
    consistency_errors: int = 0
    consistency_failed_categories: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    latency_seconds: float = 0.0
    failure_stage: str | None = None
    failure_reason: str | None = None
    # C-3：上下文尺寸（writer_view 为准的 char 计量）+ Canon 存储计数
    context_sizes: dict = field(default_factory=dict)
    canon_counts: dict = field(default_factory=dict)
    # C-7：evolution 路径（每轮 revision 的 focus/reviewers/writer_tokens/composite）
    evolution_path: list[dict] = field(default_factory=list)

    @property
    def writer_context_chars(self) -> int:
        wv = self.context_sizes.get("writer_view") or {}
        return int(wv.get("character_context_chars", 0)) + int(
            wv.get("recent_summary_chars", 0)
        ) + int(wv.get("world_context_chars", 0))

    @property
    def is_censorship(self) -> bool:
        """显式拒答 / 内容策略触发（censorship 不当普通低分，§禁止项 5）。

        空输出 / 解析失败归 invalid，但非 censorship——censorship 只认显式
        refusal / policy / censor 词，避免把模型偶发空输出误判为审查。
        """
        if self.status != CHAPTER_INVALID:
            return False
        reason = (self.failure_reason or "").lower()
        return any(k in reason for k in ("censor", "refusal", "refuse", "policy", "拒答", "内容审核", "审查"))


@dataclass
class BaselineRunResult:
    """单条 prompt × repeat 的 20 章聚合结果。"""

    agent: str
    prompt_id: str
    sample_index: int
    expected_chapters: int
    chapters: list[ChapterScore] = field(default_factory=list)

    # ── Reliability ──

    @property
    def completed_chapters(self) -> int:
        return sum(1 for c in self.chapters if c.status in _SUCCESS_STATES)

    @property
    def full_run_success(self) -> float:
        """单 run 二值：全部章节成功为 1.0，否则 0.0。跨 run 聚合得 success_rate。"""
        return 1.0 if self.expected_chapters > 0 and self.completed_chapters >= self.expected_chapters else 0.0

    @property
    def chapter_completion_rate(self) -> float:
        if self.expected_chapters <= 0:
            return 0.0
        return round(self.completed_chapters / self.expected_chapters, 3)

    @property
    def timeout_rate(self) -> float:
        return self._rate(CHAPTER_TIMEOUT)

    @property
    def invalid_rate(self) -> float:
        return self._rate(CHAPTER_INVALID)

    @property
    def censorship_rate(self) -> float:
        if not self.chapters:
            return 0.0
        return round(sum(1 for c in self.chapters if c.is_censorship) / len(self.chapters), 3)

    @property
    def resume_success_rate(self) -> float:
        """有 resumed 章时，resumed 章全部成功恢复的比例；无 resumed 时为 1.0。"""
        resumed = [c for c in self.chapters if c.status == CHAPTER_RESUMED]
        if not resumed:
            return 1.0
        return 1.0  # resumed 即成功恢复；失败会落到 failed/partial

    def _rate(self, status: str) -> float:
        if not self.chapters:
            return 0.0
        return round(sum(1 for c in self.chapters if c.status == status) / len(self.chapters), 3)

    # ── Quality ──

    def _overall_scores(self) -> list[float]:
        return [c.overall for c in self.chapters if c.overall is not None]

    @property
    def quality_mean(self) -> float | None:
        vals = self._overall_scores()
        return round(fmean(vals), 3) if vals else None

    @property
    def quality_std(self) -> float | None:
        vals = self._overall_scores()
        return round(stdev(vals), 3) if len(vals) > 1 else (0.0 if vals else None)

    def _window_mean(self, start: int, width: int) -> float | None:
        vals = self._overall_scores()
        window = vals[start:start + width]
        return round(fmean(window), 3) if len(window) == width else None

    @property
    def first_window_score(self) -> float | None:
        return self._window_mean(0, 2)

    @property
    def last_window_score(self) -> float | None:
        vals = self._overall_scores()
        return self._window_mean(max(len(vals) - 2, 0), 2) if len(vals) >= 2 else None

    @property
    def middle_window_score(self) -> float | None:
        vals = self._overall_scores()
        if len(vals) < 4:
            return None
        start = max((len(vals) - 2) // 2, 0)
        return self._window_mean(start, 2)

    @property
    def degradation(self) -> float | None:
        """尾段 - 首段（负值=质量随连载衰减）。"""
        f, l = self.first_window_score, self.last_window_score
        return round(l - f, 3) if f is not None and l is not None else None

    @property
    def trend_slope(self) -> float | None:
        vals = self._overall_scores()
        if len(vals) < 2:
            return None
        n = len(vals)
        x_mean = (n - 1) / 2
        y_mean = fmean(vals)
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom == 0:
            return None
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        return round(num / denom, 3)

    # ── C-5 分段统计 / C-6 三类 degradation ──

    def segment_stats(self, window: int = 5) -> list[dict]:
        """按 window 分段（如 1–5 / 6–10 / 11–15 / 16–20），每段 quality mean±std。"""
        out: list[dict] = []
        vals = self._overall_scores()
        for start in range(0, len(vals), window):
            seg = vals[start:start + window]
            if not seg:
                continue
            seg_std = round(stdev(seg), 3) if len(seg) > 1 else 0.0
            out.append({
                "segment": f"{start + 1}-{start + len(seg)}",
                "mean": round(fmean(seg), 3),
                "std": seg_std,
                "chapters": len(seg),
            })
        return out

    @staticmethod
    def _slope(vals: list[float]) -> float | None:
        if len(vals) < 2:
            return None
        n = len(vals)
        x_mean = (n - 1) / 2
        y_mean = fmean(vals)
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom == 0:
            return None
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        return round(num / denom, 6)

    @property
    def cost_growth_slope(self) -> float | None:
        """C-6：tokens/chapter 随章节增长的最小二乘斜率（仅成功章）。"""
        vals = [
            float(c.token_usage.get("total_tokens", 0) or 0)
            for c in self.chapters if c.status in _SUCCESS_STATES
        ]
        return self._slope(vals)

    @property
    def context_growth_slope(self) -> float | None:
        """C-6：writer 上下文 char 数随章节增长的斜率（仅成功章）。"""
        vals = [
            float(c.writer_context_chars)
            for c in self.chapters if c.status in _SUCCESS_STATES
        ]
        return self._slope(vals)

    @property
    def quality_per_cost(self) -> dict | None:
        """C-8：quality-per-cost 概览（mean quality / mean tokens per chapter）。

        token 为 provider 原始报告（stepfun suspect），只用于相对比较。
        """
        vals = self._overall_scores()
        tokens = [
            float(c.token_usage.get("total_tokens", 0) or 0)
            for c in self.chapters if c.status in _SUCCESS_STATES
        ]
        if not vals or not tokens or fmean(tokens) <= 0:
            return None
        return {
            "quality_mean": round(fmean(vals), 3),
            "tokens_per_chapter": round(fmean(tokens), 1),
            "quality_per_million_tokens": round(fmean(vals) / (fmean(tokens) / 1_000_000), 3),
        }

    def evolution_gain_per_revision(self) -> list[dict]:
        """C-8：逐轮 revision 的 quality gain（composite delta）+ writer token 增量。

        writer_tokens 为累计值 → 相邻 entry 差值即该轮增量。
        """
        gains: list[dict] = []
        for i, c in enumerate(self.chapters):
            if not c.evolution_path:
                continue
            for j in range(1, len(c.evolution_path)):
                prev, cur = c.evolution_path[j - 1], c.evolution_path[j]
                comp_prev, comp_cur = prev.get("composite"), cur.get("composite")
                wt_prev, wt_cur = prev.get("writer_tokens") or {}, cur.get("writer_tokens") or {}
                gains.append({
                    "chapter": c.chapter_number,
                    "from_v": prev.get("revision"),
                    "to_v": cur.get("revision"),
                    "focus": cur.get("focus"),
                    "quality_gain": (
                        round(comp_cur - comp_prev, 3)
                        if comp_prev is not None and comp_cur is not None else None
                    ),
                    "writer_input_delta": (
                        int(wt_cur.get("input", 0)) - int(wt_prev.get("input", 0))
                    ) or None,
                    "writer_output_delta": (
                        int(wt_cur.get("output", 0)) - int(wt_prev.get("output", 0))
                    ) or None,
                })
        return gains

    # ── Consistency ──

    @property
    def ced(self) -> float | None:
        """Consistency Error Density = 总错误数 / 章数。"""
        if not self.chapters:
            return None
        return round(sum(c.consistency_errors for c in self.chapters) / len(self.chapters), 3)

    def _error_window(self, start: int, width: int) -> float | None:
        seg = self.chapters[start:start + width]
        if len(seg) != width:
            return None
        return round(sum(c.consistency_errors for c in seg) / width, 3)

    @property
    def first_error_density(self) -> float | None:
        return self._error_window(0, 2)

    @property
    def last_error_density(self) -> float | None:
        n = len(self.chapters)
        return self._error_window(max(n - 2, 0), 2) if n >= 2 else None

    @property
    def error_growth_slope(self) -> float | None:
        """每章错误数的最小二乘斜率（正值=错误随连载增长）。"""
        errs = [c.consistency_errors for c in self.chapters]
        if len(errs) < 2:
            return None
        n = len(errs)
        x_mean = (n - 1) / 2
        y_mean = fmean(errs)
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom == 0:
            return None
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(errs))
        return round(num / denom, 3)

    # ── Cost ──
    # token 含 reasoning + 多轮历史重复（见 _extract_token_usage 口径说明）；
    # 绝对值偏高，Cost 维用相对比较（A vs B），不作计费依据。

    @property
    def total_tokens(self) -> int:
        return sum(c.token_usage.get("total_tokens", 0) for c in self.chapters)

    @property
    def tokens_per_chapter(self) -> float | None:
        done = [c for c in self.chapters if c.status in _SUCCESS_STATES]
        if not done:
            return None
        return round(fmean(c.token_usage.get("total_tokens", 0) for c in done), 1)

    @property
    def latency_per_chapter(self) -> float | None:
        done = [c for c in self.chapters if c.status in _SUCCESS_STATES]
        if not done:
            return None
        return round(fmean(c.latency_seconds for c in done), 3)


# ── 跨 run 聚合 + Go/No-Go ──────────────────────────────


def aggregate_reliability(runs: list[BaselineRunResult]) -> dict:
    """跨 run 聚合 Reliability 指标（mean rate）。"""
    if not runs:
        return {}
    n = len(runs)
    return {
        "full_run_success_rate": round(fmean(r.full_run_success for r in runs), 3),
        "chapter_completion_rate": round(
            sum(r.completed_chapters for r in runs) / sum(r.expected_chapters for r in runs), 3
        ) if sum(r.expected_chapters for r in runs) else 0.0,
        "timeout_rate": round(fmean(r.timeout_rate for r in runs), 3),
        "invalid_rate": round(fmean(r.invalid_rate for r in runs), 3),
        "censorship_rate": round(fmean(r.censorship_rate for r in runs), 3),
        "resume_success_rate": round(fmean(r.resume_success_rate for r in runs), 3),
        "runs": n,
    }


def go_no_go_gates(runs: list[BaselineRunResult]) -> dict:
    """进入 30 章主实验前的 Go/No-Go 门槛（§5）。

    - full_run_success_rate >= 0.95
    - chapter_completion_rate >= 0.98
    - failure_stage 可分类（所有 failed/timeout/invalid 章有 failure_stage）
    - checkpoint 可恢复（Phase 1 已验证；此处只检查有 resumed 章的 run 恢复成功）
    """
    rel = aggregate_reliability(runs)
    failed_chapters = [
        c for r in runs for c in r.chapters
        if c.status not in _SUCCESS_STATES
    ]
    failure_stage_classifiable = all(
        c.failure_stage for c in failed_chapters
    ) if failed_chapters else True
    checkpoint_recoverable = all(
        r.resume_success_rate >= 1.0 for r in runs
    )
    gates = {
        "full_run_success_rate": rel.get("full_run_success_rate", 0.0),
        "chapter_completion_rate": rel.get("chapter_completion_rate", 0.0),
        "failure_stage_classifiable": failure_stage_classifiable,
        "checkpoint_recoverable": checkpoint_recoverable,
    }
    gates["passed"] = (
        gates["full_run_success_rate"] >= 0.95
        and gates["chapter_completion_rate"] >= 0.98
        and failure_stage_classifiable
        and checkpoint_recoverable
    )
    return gates


# ── run_baseline 驱动器 ───────────────────────────────────


async def run_baseline(
    *,
    adapter,
    judge,
    cases: list,
    persist_dir: str,
    checkpoint_path: Path,
    consistency_checker=None,
    resume: bool = False,
    chapter_timeout: float | None = None,
    sample_index: int = 0,
) -> BaselineRunResult:
    """跑一条 prompt 的 N 章 durable baseline，逐章 Judge + ConStory 评分。

    复用 run_durable 驱动章节（V2 DB 为恢复真相源）；评分逻辑对齐
    BenchmarkRunner._run_once：judge.score → efficiency → consistency → weighted_score。
    章节内容从 V2 DB 读（committed draft_content），不依赖进程内存。

    judge 为必填（或 fake）；consistency_checker 可选（None 则跳过一致性维）。
    """
    from .durable import run_durable

    prompt_id = getattr(cases[0], "name", "run") if cases else "run"
    status = await run_durable(
        adapter=adapter, cases=cases, persist_dir=persist_dir,
        checkpoint_path=checkpoint_path, resume=resume,
        chapter_timeout=chapter_timeout,
    )

    # 定位 V2 project（按 name）以读 committed content
    pid = _find_project(persist_dir, cases)

    case_by_num = {adapter._chapter_number(c): c for c in cases}
    chapter_scores: list[ChapterScore] = []
    for ck in status.checkpoints:
        obs = ck.observations or {}
        cs = ChapterScore(
            chapter_number=ck.chapter_number, status=ck.status,
            token_usage=ck.token_usage, latency_seconds=ck.latency_seconds,
            failure_stage=ck.failure_stage, failure_reason=ck.failure_reason,
            context_sizes=obs.get("context_sizes") or {},
            canon_counts=obs.get("canon_counts") or {},
            evolution_path=obs.get("evolution_path") or [],
        )
        if ck.status in _SUCCESS_STATES:
            case = case_by_num.get(ck.chapter_number)
            content = _read_chapter_content(persist_dir, pid, ck.chapter_number)
            if case and content and content.strip():
                await _score_chapter(cs, case, content, judge, consistency_checker)
        chapter_scores.append(cs)

    return BaselineRunResult(
        agent=adapter.name, prompt_id=prompt_id, sample_index=sample_index,
        expected_chapters=len(cases), chapters=chapter_scores,
    )


def _find_project(persist_dir: str, cases: list) -> str | None:
    """按 project name 在 V2 DB 查 project_id。"""
    try:
        from novel_agent.storage.manager import ProjectManager

        mgr = ProjectManager(Path(persist_dir))
        project_name = cases[0].project_id or f"eval:{cases[0].name}" if cases else ""
        for p in mgr.list_projects():
            if p["name"] == project_name:
                return p["id"]
    except Exception:  # noqa: BLE001 - V2 DB 读失败不应阻断评分
        return None
    return None


def _read_chapter_content(persist_dir: str, project_id: str | None, chapter_number: int) -> str:
    """从 V2 DB 读 committed draft_content（resumed 章也在此）。"""
    if not project_id:
        return ""
    try:
        from novel_agent.storage.manager import ProjectManager

        mgr = ProjectManager(Path(persist_dir))
        ch = mgr.get_chapter(project_id, chapter_number)
        return (ch or {}).get("draft_content", "") if ch else ""
    except Exception:  # noqa: BLE001 - V2 DB 读失败不应阻断评分
        return ""


async def _score_chapter(
    cs: ChapterScore, case, content: str, judge, consistency_checker
) -> None:
    """对齐 BenchmarkRunner._run_once 的单章评分，结果填入 cs。"""
    from novel_agent_eval.judge import QUALITY_DIMS
    from novel_agent_eval.metrics import efficiency_score, weighted_score

    js = await judge.score(content, case)
    eff = efficiency_score(
        cs.latency_seconds, cs.token_usage.get("total_tokens", 0), 0
    )
    dims = {**js.dimensions, "efficiency": eff}
    if not js.valid:
        dims.update({d: 0 for d in QUALITY_DIMS})
    cs.dimensions = dims
    cs.overall = weighted_score(dims, case.stage)

    if consistency_checker is not None:
        from novel_agent_eval.constory import consistency_score as _con_score

        gt = case.ground_truth
        if hasattr(gt, "model_dump"):
            gt = gt.model_dump()
        reference = {
            "ground_truth": gt,
            "story_outline": case.story_outline,
            "chapter_outline": case.target_chapter_outline,
            "previous_context": case.previous_context,
        }
        try:
            report = await consistency_checker.check_consistency(content, reference=reference)
        except TypeError as exc:
            if "reference" not in str(exc):
                raise
            report = await consistency_checker.check_consistency(content)
        if report.failed_categories:
            # ConStory partial/total failure: score is unreliable (total only
            # counts successful categories) → None, not an inflated high score.
            cs.consistency_score = None
        else:
            cs.consistency_score = _con_score(report.total)
        cs.consistency_errors = report.total
        cs.consistency_failed_categories = report.failed_categories
