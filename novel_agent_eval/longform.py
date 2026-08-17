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
from dataclasses import dataclass, field
from typing import Any

from novel_agent_eval.eqbench_bridge import LongformPlan, plan_to_cases
from novel_agent_eval.eqbench_judge import EQBenchJudge, eqbench_chapter_score


@dataclass
class ChapterResult:
    """单章结果：14 维原始分 + 0-20 加权分 + 正文 + 生成 meta。"""

    chapter_index: int          # 1-based
    scores: dict[str, float]    # 14 维原始分（0-20，未反转）
    eqbench_score: float | None  # 0-20 加权单章分
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LongformResult:
    """单 agent × 单 prompt 的 8 章聚合结果。"""

    agent: str
    prompt_id: str
    title: str
    plan: LongformPlan
    chapters: list[ChapterResult]
    mean_score: float            # 8 章均值（0-20）
    eqbench_0_100: float         # mean × 5
    degradation: float           # 尾段均值 - 首段均值（负=衰减）

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


async def run_longform(
    *,
    agent,                       # AgentAdapter
    judge: EQBenchJudge,
    plan: LongformPlan,
    word_target: int = 1000,
    degradation_window: int = 2,
) -> LongformResult:
    """跑一条 prompt 的 8 章连载并逐章评分，聚合 0-100 + degradation。"""
    cases = plan_to_cases(plan, word_target=word_target)
    chapters: list[ChapterResult] = []
    context = ""
    for i, case in enumerate(cases, start=1):
        case.previous_context = context
        gen = None
        for attempt in range(5):
            try:
                gen = await agent.generate(case)
                break
            except Exception as e:
                if "429" in str(e) or "concurrency" in str(e).lower():
                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                raise

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
                if "429" in str(e) or "concurrency" in str(e).lower():
                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                raise
        eq = eqbench_chapter_score(scores)
        chapters.append(
            ChapterResult(
                chapter_index=i,
                scores=scores,
                eqbench_score=eq,
                content=gen.content,
                meta=gen.meta,
            )
        )
        context += f"\n\n[Chapter {i}]\n{gen.content}"

    valid = [c.eqbench_score for c in chapters if c.eqbench_score is not None]
    mean = round(sum(valid) / len(valid), 3) if valid else 0.0
    deg = degradation_score(valid, window=degradation_window)
    return LongformResult(
        agent=agent.name,
        prompt_id=plan.prompt_id,
        title=plan.title,
        plan=plan,
        chapters=chapters,
        mean_score=mean,
        eqbench_0_100=round(mean * 5, 3),
        degradation=deg if deg is not None else 0.0,
    )


def render_longform_table(results: list[LongformResult]) -> str:
    """agent × prompt 的 degradation 报告表（纯文本 markdown，横评脚本打印用）。

    列：agent / title / 0-100 总分 / 单章均值(0-20) / degradation(尾-首) / 每章分。
    degradation 为负表示长程质量随连载衰减，是 EQ-Bench Longform 的核心观测。
    """
    lines = [
        "| agent | title | eqbench(0-100) | mean(0-20) | degradation | per-chapter |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        per_chapter = "/".join(
            f"{s:.0f}" for s in r.chapter_scores
        )
        lines.append(
            f"| {r.agent} | {r.title} | {r.eqbench_0_100:.1f} | {r.mean_score:.1f} "
            f"| {r.degradation:+.2f} | {per_chapter} |"
        )
    return "\n".join(lines)
