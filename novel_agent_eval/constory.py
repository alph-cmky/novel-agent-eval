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
import os
import re
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from novel_agent_eval.vendor.constory.judge import (
    EVALUATION_CRITERIA,
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
    failed_categories: list[str] = Field(default_factory=list)


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


def _split_chatml(template: str) -> tuple[str, str]:
    """把 ConStory prompt 模板（ChatML 格式）拆成 (system, user) 两条消息。

    官方 JudgeLLMClient 把整个 ChatML 文本塞进单条 user message；StepFun 走 OpenAI
    兼容协议，更规范的做法是拆成 system/user 两条 role，避免模型把 `<|im_start|>`
    当普通文本。
    """
    m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", template, re.DOTALL)
    system = m.group(1).strip() if m else ""
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", template, re.DOTALL)
    user = m.group(1).strip() if m else template
    return system, user


def consistency_score(total_errors: int, penalty_per_error: int = 20) -> int:
    """把 ConStory 检出的矛盾总数折算为 0-100 连贯性分（每矛盾扣 penalty 分，下限 0）。"""
    return max(0, 100 - penalty_per_error * total_errors)


class ConStoryCheckerAdapter:
    """ConStory-Checker 适配器：对 narrative 做连贯性检测，输出 3 类聚合报告。

    client 为 OpenAI 兼容 async client（StepFun），可注入 mock 供测试（与 judge.py
    的构造模式一致）。对 step-3.7-flash（reasoning 模型）注入 reasoning_effort=low +
    temperature=0，并把 ChatML 模板拆成 system/user 两条消息——官方 JudgeLLMClient
    的默认档（无 reasoning_effort、temperature=0.5、ChatML 塞单条 user）会让
    reasoning 模型 overthinking 编造矛盾，正好违背 ConStory「不编造」的初衷。
    """

    def __init__(
        self,
        client=None,
        model: str | None = None,
        prompts_dir: str | None = None,
    ):
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("STEPFUN_API_KEY"),
            base_url=os.environ.get("STEPFUN_BASE_URL"),
        )
        self._model = model or os.environ.get("STEPFUN_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
        self._templates = load_prompt_templates(str(prompts_dir or _VENDOR_PROMPTS_DIR))

    async def _evaluate_category(self, template: str, narrative: str, cat: str) -> str:
        """对单个类别发起一次评估，返回原始 content（失败时 raise，由上层降级）。"""
        prompt = template.replace("{{ Content }}", narrative).replace(
            "{{ Query }}", f"{EVALUATION_CRITERIA[cat]['name']} Analysis"
        )
        system, user = _split_chatml(prompt)
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=8192,
            reasoning_effort="low",
        )
        return resp.choices[0].message.content or ""

    async def check_consistency(self, narrative: str) -> ConsistencyReport:
        raw: dict[str, list[dict]] = {full: [] for full in _ALL_SUBTYPES}
        failed_categories: list[str] = []

        tasks = {
            cat: self._evaluate_category(self._templates[cat], narrative, cat)
            for cat in EVALUATION_CRITERIA
        }
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for cat, resp in zip(tasks, gathered):
            cfg = EVALUATION_CRITERIA[cat]
            if isinstance(resp, Exception):
                # 保留空列表供诊断，但显式标记失败，不能等价于零矛盾。
                failed_categories.append(cat)
                continue
            content = resp or ""
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
            failed_categories=failed_categories,
        )
