# novel_agent_eval/longform.py
"""EQ-Bench Longform 编排：bridge 产出 plan → 逐章生成 + EQBenchJudge 评分 →
聚合 0-100 总分 + degradation（长程质量衰减）。

一条 prompt 跑 8 章连载：每章把前文累积回填 previous_context，让被测模型在
完整前文下续写（对齐官方多轮历史）。逐章用 14 维 EQBenchJudge 评分，单章分
= eqbench_chapter_score(原始 14 维)，整体 = 8 章均值 → 0-100（×5）。

degradation 为横评自建指标（官方无此定义）：尾段均值 - 首段均值，负值=质量随
连载衰减。window 控制首/尾各取几章。
"""

import asyncio
import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

from novel_agent_eval.eqbench_bridge import LongformPlan, plan_to_cases
from novel_agent_eval.eqbench_judge import EQBenchJudge, eqbench_chapter_score


def _retryable_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "rate_limited",
            "concurrency",
            "remoteprotocol",
            "incomplete chunked",
            "connection reset",
            "server disconnected",
        )
    )


@dataclass
class ChapterResult:
    """单章结果：14 维原始分 + 0-20 加权分 + 正文 + 生成 meta。"""

    chapter_index: int          # 1-based
    scores: dict[str, float]    # 14 维原始分（0-20，未反转）
    eqbench_score: float | None  # 0-20 加权单章分
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def content_length(self) -> int:
        return len(self.content)


@dataclass
class LongformResult:
    """单 agent × 单 prompt 的 8 章聚合结果。"""

    agent: str
    prompt_id: str
    title: str
    plan: LongformPlan
    chapters: list[ChapterResult]
    mean_score: float | None     # 任一章节 invalid 时为 None
    eqbench_0_100: float | None  # mean × 5
    degradation: float | None    # 任一章节 invalid 时为 None
    sample_index: int = 0
    valid_chapters: int = 0
    completion_rate: float = 0.0
    first_window_score: float | None = None
    middle_window_score: float | None = None
    last_window_score: float | None = None
    trend_slope: float | None = None

    @property
    def chapter_scores(self) -> list[float]:
        return [c.eqbench_score for c in self.chapters if c.eqbench_score is not None]


def degradation_score(scores: list[float], window: int = 2) -> float | None:
    """尾段均值 - 首段均值。scores 为按章序的 0-20 单章分，长度不足 2*window 时返回 None。"""
    if len(scores) < 2 * window:
        return None
    head = sum(scores[:window]) / window
    tail = sum(scores[-window:]) / window
    return round(tail - head, 3)


def _window_mean(scores: list[float], start: int, width: int) -> float | None:
    window = scores[start:start + width]
    return round(sum(window) / width, 3) if len(window) == width else None


def _trend_slope(scores: list[float]) -> float | None:
    """Return the least-squares score trend per chapter."""
    if len(scores) < 2:
        return None
    x_mean = (len(scores) - 1) / 2
    y_mean = sum(scores) / len(scores)
    denominator = sum((i - x_mean) ** 2 for i in range(len(scores)))
    numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(scores))
    return round(numerator / denominator, 3) if denominator else None


async def run_longform(
    *,
    agent,                       # AgentAdapter
    judge: EQBenchJudge,
    plan: LongformPlan,
    word_target: int = 1000,
    degradation_window: int = 2,
    sample_index: int = 0,
    max_story_outline_chars: int | None = None,
) -> LongformResult:
    """跑一条 prompt 的 8 章连载并逐章评分，聚合 0-100 + degradation。"""
    cases = plan_to_cases(
        plan,
        word_target=word_target,
        max_story_outline_chars=max_story_outline_chars,
    )
    chapters: list[ChapterResult] = []
    context = ""
    story_persist_dir = tempfile.mkdtemp(prefix="novel_longform_")
    story_project_id = f"longform_{plan.prompt_id}_{sample_index}_{agent.name}"
    try:
        for i, case in enumerate(cases, start=1):
            case.previous_context = context
            case.project_id = story_project_id
            case.persist_dir = story_persist_dir
            gen = None
            for attempt in range(5):
                try:
                    gen = await agent.generate(case)
                    break
                except Exception as e:
                    if _retryable_error(e):
                        await asyncio.sleep(3.0 * (attempt + 1))
                        continue
                    raise

            if hasattr(agent, "index_chapter"):
                agent.index_chapter(
                    project_id=story_project_id,
                    persist_dir=story_persist_dir,
                    chapter_number=i,
                    content=gen.content,
                )

            scores = None
            for attempt in range(5):
                try:
                    scores = await judge.score_chapter(
                        writing_prompt=plan.writing_prompt,
                        final_plan=plan.final_plan,
                        character_profiles=plan.character_profiles,
                        chapter_number=i,
                        chapter_text=gen.content,
                    )
                    break
                except Exception as e:
                    if _retryable_error(e):
                        await asyncio.sleep(3.0 * (attempt + 1))
                        continue
                    raise
            eq = eqbench_chapter_score(scores)
            chapter_meta = dict(gen.meta)
            chapter_meta["previous_context_length"] = len(context)
            chapter_meta["target_word_count"] = case.word_target
            chapter_meta["memory_project_id"] = story_project_id
            chapter_meta["memory_persist_dir"] = story_persist_dir
            chapters.append(
                ChapterResult(
                    chapter_index=i,
                    scores=scores,
                    eqbench_score=eq,
                    content=gen.content,
                    meta=chapter_meta,
                )
            )
            context += f"\n\n[Chapter {i}]\n{gen.content}"
    finally:
        shutil.rmtree(story_persist_dir, ignore_errors=True)

    valid = [c.eqbench_score for c in chapters]
    complete = all(score is not None for score in valid)
    values = [score for score in valid if score is not None]
    mean = round(sum(values) / len(values), 3) if complete and values else None
    deg = degradation_score(values, window=degradation_window) if complete else None
    first = _window_mean(values, 0, degradation_window) if complete else None
    middle_start = max((len(values) - degradation_window) // 2, 0)
    middle = _window_mean(values, middle_start, degradation_window) if complete else None
    last = _window_mean(
        values, len(values) - degradation_window, degradation_window
    ) if complete else None
    return LongformResult(
        agent=agent.name,
        prompt_id=plan.prompt_id,
        title=plan.title,
        plan=plan,
        chapters=chapters,
        mean_score=mean,
        eqbench_0_100=round(mean * 5, 3) if mean is not None else None,
        degradation=deg,
        sample_index=sample_index,
        valid_chapters=len(values),
        completion_rate=round(len(values) / len(cases), 3) if cases else 0.0,
        first_window_score=first,
        middle_window_score=middle,
        last_window_score=last,
        trend_slope=_trend_slope(values) if complete else None,
    )


def render_longform_table(results: list[LongformResult]) -> str:
    """agent × prompt 的 degradation 报告表（纯文本 markdown，横评脚本打印用）。

    列：agent / title / 0-100 总分 / 单章均值(0-20) / degradation(尾-首) / 每章分。
    degradation 为负表示长程质量随连载衰减，是 EQ-Bench Longform 的核心观测。
    """
    lines = [
        "| agent | title | eqbench(0-100) | completion | first/mid/last | trend | degradation | per-chapter |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for r in results:
        per_chapter = "/".join(
            f"{s:.0f}" if s is not None else "—" for s in
            [c.eqbench_score for c in r.chapters]
        )
        score = f"{r.eqbench_0_100:.1f}" if r.eqbench_0_100 is not None else "—"
        windows = (
            f"{r.first_window_score:.1f}/{r.middle_window_score:.1f}/"
            f"{r.last_window_score:.1f}"
            if r.first_window_score is not None
            else "—"
        )
        trend = f"{r.trend_slope:+.2f}" if r.trend_slope is not None else "—"
        degradation = f"{r.degradation:+.2f}" if r.degradation is not None else "—"
        lines.append(
            f"| {r.agent} | {r.title} | {score} | {r.completion_rate:.1%} "
            f"| {windows} | {trend} | {degradation} | {per_chapter} |"
        )
    return "\n".join(lines)
