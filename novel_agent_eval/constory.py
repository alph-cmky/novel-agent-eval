# novel_agent_eval/constory.py
"""ConStory-Checker 连贯性检测适配器。

复用 vendored 官方 ConStory-Checker（novel_agent_eval/vendor/constory/judge.py）：
JudgeLLMClient + load_prompt_templates + EVALUATION_CRITERIA + parse_criteria_response。

对 5 类评估并发请求，把 19 个子类型的错误对象按 3 类映射表
（character / timeline / worldbuilding）聚合进 ConsistencyReport；
narrative_style 3 个子类型只保留在 raw，不进 3 类聚合。
某一类评估失败时该类子类型记空列表（降级，不抛异常）。
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import aiohttp
from pydantic import BaseModel

from novel_agent_eval.vendor.constory.judge import (
    EVALUATION_CRITERIA,
    JudgeLLMClient,
    load_prompt_templates,
    parse_criteria_response,
)

_DEFAULT_JUDGE_MODEL = "step-3.7-flash"

# 19 子类型 → 代码 continuity 3 类映射表（硬性表，见 task-12 brief §3 类映射）
_SUBTYPE_TO_CATEGORY: dict[str, str] = {
    "characterization_memory_contradictions": "character",
    "characterization_knowledge_contradictions": "character",
    "characterization_skill_power_fluctuations": "character",
    "characterization_forgotten_abilities": "character",
    "factual_detail_appearance_mismatches": "character",
    "timeline_plot_absolute_time_contradictions": "timeline",
    "timeline_plot_duration_timeline_contradictions": "timeline",
    "timeline_plot_simultaneity_contradictions": "timeline",
    "timeline_plot_causeless_effects": "timeline",
    "timeline_plot_causal_logic_violations": "timeline",
    "timeline_plot_abandoned_plot_elements": "timeline",
    "world_building_core_rules_violations": "worldbuilding",
    "world_building_social_norms_violations": "worldbuilding",
    "world_building_geographical_contradictions": "worldbuilding",
    "factual_detail_nomenclature_confusions": "worldbuilding",
    "factual_detail_quantitative_mismatches": "worldbuilding",
}

# 19 子类型全量（含不映射 3 类的文风维）
_ALL_SUBTYPES: list[str] = [
    f"{cat}_{sc}"
    for cat, cfg in EVALUATION_CRITERIA.items()
    for sc in cfg["sub_criteria"]
]

# vendor prompts 目录相对本文件位置
_VENDOR_PROMPTS_DIR = Path(__file__).parent / "vendor" / "constory" / "prompts"


class ConsistencyError(BaseModel):
    exact_quote: str
    location: str = ""
    contradiction_pair: str | None = None
    contradiction_location: str | None = None
    context: str = ""
    subtype: str  # 19 子类型名


class ConsistencyReport(BaseModel):
    character: list[ConsistencyError]
    timeline: list[ConsistencyError]
    worldbuilding: list[ConsistencyError]
    raw: dict[str, list[dict]]  # 19 子类型 → 官方格式错误对象（含 narrative_style 3 类）
    total: int


def _to_error(subtype: str, e: dict) -> ConsistencyError:
    """把官方格式错误对象规范化为 ConsistencyError（字段缺失兜底）。"""
    return ConsistencyError(
        exact_quote=e.get("exact_quote") or "",
        location=e.get("location") or "",
        contradiction_pair=e.get("contradiction_pair"),
        contradiction_location=e.get("contradiction_location"),
        context=e.get("context") or "",
        subtype=subtype,
    )


class ConStoryCheckerAdapter:
    """ConStory-Checker 适配器：对 narrative 做连贯性检测，输出 3 类聚合报告。"""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrent: int = 5,
    ):
        self._api_base = api_base or os.environ["STEPFUN_BASE_URL"]
        self._api_key = api_key or os.environ["STEPFUN_API_KEY"]
        self._model = model or os.environ.get("STEPFUN_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
        self._max_concurrent = max_concurrent
        self._templates = load_prompt_templates(str(_VENDOR_PROMPTS_DIR))
        self._logger = logging.getLogger(__name__)

    async def check_consistency(self, narrative: str) -> ConsistencyReport:
        client = JudgeLLMClient(
            api_base=self._api_base,
            api_key=self._api_key,
            model=self._model,
            max_concurrent=self._max_concurrent,
            logger=self._logger,
        )
        raw: dict[str, list[dict]] = {full: [] for full in _ALL_SUBTYPES}

        async with aiohttp.ClientSession() as session:
            tasks = {
                cat: client.evaluate_criteria(
                    session, self._templates[cat], narrative, cat
                )
                for cat in EVALUATION_CRITERIA
            }
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for cat, resp in zip(tasks, gathered):
                cfg = EVALUATION_CRITERIA[cat]
                if isinstance(resp, Exception) or not resp.get("success"):
                    # 该类评估失败 → 子类型记空列表（降级，不抛异常）
                    continue
                content = resp.get("content", "")
                parsed = parse_criteria_response(content, cfg["sub_criteria"], cat)
                for sc in cfg["sub_criteria"]:
                    full = f"{cat}_{sc}"
                    try:
                        items = json.loads(parsed.get(sc, "[]") or "[]")
                    except json.JSONDecodeError:
                        items = []
                    if not isinstance(items, list):
                        items = []
                    raw[full] = [it for it in items if isinstance(it, dict)]

        buckets: dict[str, list[ConsistencyError]] = {
            "character": [], "timeline": [], "worldbuilding": []
        }
        for full, items in raw.items():
            cat = _SUBTYPE_TO_CATEGORY.get(full)
            if cat is None:
                continue  # narrative_style 文风维，仅保留在 raw
            buckets[cat].extend(_to_error(full, e) for e in items)

        total = sum(len(v) for v in buckets.values())
        return ConsistencyReport(
            character=buckets["character"],
            timeline=buckets["timeline"],
            worldbuilding=buckets["worldbuilding"],
            raw=raw,
            total=total,
        )
