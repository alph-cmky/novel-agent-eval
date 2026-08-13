# novel_agent_eval/judge.py
"""LLM Judge — 第三方小说评审，对生成章节按 8 个质量维打分。

使用 StepFun 单模型（OpenAI 兼容协议），不用 Claude/GPT 双 Judge。
测试用 mock client，不消耗真实 API。

输入 = EvalCase（target_chapter_outline / story_outline / previous_context）+ draft。
输出 = JudgeScore（dimensions 8 维各 0-100 + overall）。
"""

import json
import os
import re

from openai import AsyncOpenAI
from pydantic import BaseModel

from novel_agent_eval.dataset.schema import EvalCase

QUALITY_DIMS = [
    "consistency", "writing", "ai_flavor", "dialogue",
    "plot", "instruction", "creativity", "controllability",
]

# 真实调用通过环境变量覆盖；默认值取主仓库已在用的 StepFun 模型名
DEFAULT_JUDGE_MODEL = "step-3.7-flash"

# 8 质量维 × 5 档 Rubric（方案 §6.1），逐字用于构造 prompt
_RUBRIC_BANDS = ["0-20", "21-40", "41-60", "61-80", "81-100"]
_RUBRIC_TABLE = [
    ("consistency", "连贯性", ["多处致命矛盾", "有明显矛盾", "少量矛盾", "基本一致", "完全一致"]),
    ("writing", "文笔", ["生硬难读", "平淡", "通顺", "有亮点", "文学性强"]),
    ("ai_flavor", "AI味", ["机器感极强", "较明显", "轻微", "很自然", "几乎无 AI 味"]),
    ("dialogue", "对话质量", ["所有角色说话一样", "偶有区分", "基本可分辨", "各有特点", "闻声识人"]),
    ("plot", "情节结构", ["流水账，无起伏", "有基本结构", "节奏尚可", "有高潮铺垫", "结构精巧，环环相扣"]),
    ("instruction", "指令遵循", ["偏离大纲", "部分覆盖", "基本覆盖", "较好覆盖", "精准还原"]),
    ("creativity", "创意性", ["明显套路模板", "较平淡", "有亮点", "有意外转折", "新颖且合理"]),
    ("controllability", "可操控性", ["无视反馈", "表面修改，未触及核心", "部分响应", "大部分改到位", "精准执行，举一反三"]),
]


class JudgeScore(BaseModel):
    dimensions: dict[str, int]   # 8 质量维，各 0-100
    overall: int


def _build_judge_prompt(draft: str, case: EvalCase) -> str:
    """组装 Judge prompt：角色 + Rubric 表 + 四段输入 + 输出格式。"""
    rubric = "\n".join(
        f"- {key} {label}：\n"
        + "\n".join(f"    {band}：{desc}" for band, desc in zip(_RUBRIC_BANDS, levels))
        for key, label, levels in _RUBRIC_TABLE
    )
    dims_json = ", ".join(f'"{d}": 0-100整数' for d in QUALITY_DIMS)
    return f"""你是第三方小说评审，请对本章生成正文逐维打分。每个维度 0-100 分整数，严格参照以下评分标准（按区间给分）：

{rubric}

# 输入材料

## 全书大纲（story_outline）
{case.story_outline}

## 本章大纲（target_chapter_outline）
{case.target_chapter_outline}

## 前文上下文（previous_context）
{case.previous_context}

## 本章生成正文（draft）
{draft}

# 输出要求

只输出一个 JSON 对象，不要输出任何其他文字，格式严格如下：
{{"dimensions": {{{dims_json}}}, "overall": 0-100整数}}

"overall" 是你的综合评分（0-100 整数）。
"""


# ---------- 健壮 JSON 解析（借鉴主仓库 novel_agent/schema/parser.py 的候选文本策略，独立实现） ----------

def _strip_none(obj):
    """递归移除 None 值（dict 值 + list 元素），保证下游 .get(key, default) 生效。"""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj if v is not None]
    return obj


def _repair_json(text: str) -> str | None:
    """迭代修复常见 LLM JSON 语法错误：尾逗号 / 漏左引号的字符串值，最多 6 轮。"""
    t = text.strip()
    for _ in range(6):
        try:
            json.loads(t)
            return t
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r",\s*([}\]])", r"\1", t)  # 尾逗号 ,} 或 ,]
        fixed = re.sub(
            r':\s*([^"{}\[\],\s][^"{}]*?)"(?=\s*[,}\]])', r': "\1"', fixed
        )  # 漏左引号的字符串值 10枚" -> "10枚"
        if fixed == t:
            return None
        t = fixed
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        return None


def _try_load(text: str):
    """json.loads，失败时先走 _repair_json 修复再试。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = _repair_json(text)
    if repaired is not None:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


def _parse_judge_json(text: str) -> dict | None:
    """健壮解析 Judge 输出：直接解析 → markdown 代码块 → 最外层 {...} → 修复后重试。

    返回 dict（已 strip_none）；全部失败返回 None。
    """
    if not text or not text.strip():
        return None
    candidates = [text]
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        data = _try_load(cand)
        if isinstance(data, dict):
            return _strip_none(data)
    return None


# ---------- Judge ----------

def _coerce_score(v) -> int | None:
    """把任意值强制为 0-100 整数；None/非法 → None（由调用方决定兜底）。"""
    if v is None:
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return max(0, min(100, i))


def _has_full_dims(data: dict) -> bool:
    dims = data.get("dimensions")
    return isinstance(dims, dict) and all(d in dims for d in QUALITY_DIMS)


class Judge:
    """第三方小说评审。client 为 OpenAI 兼容 async client（StepFun）。

    不传 client 时按环境变量构造：STEPFUN_API_KEY / STEPFUN_BASE_URL /
    STEPFUN_JUDGE_MODEL（默认 DEFAULT_JUDGE_MODEL）。
    """

    def __init__(self, client=None, model: str | None = None, max_attempts: int = 3):
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("STEPFUN_API_KEY"),
            base_url=os.environ.get("STEPFUN_BASE_URL"),
        )
        self._model = model or os.environ.get("STEPFUN_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self._max_attempts = max_attempts  # 初次 + 最多 2 次重试

    async def _request(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _to_score(data: dict | None) -> JudgeScore:
        """把（可能不完整的）解析结果规范化为 JudgeScore：缺失维度给 0；overall 缺失退化为 8 维平均。"""
        dims = (data or {}).get("dimensions")
        if not isinstance(dims, dict):
            dims = {}
        normalized = {d: (_coerce_score(dims.get(d)) or 0) for d in QUALITY_DIMS}
        overall = _coerce_score((data or {}).get("overall"))
        if overall is None:
            overall = round(sum(normalized.values()) / len(QUALITY_DIMS))
        return JudgeScore(dimensions=normalized, overall=overall)

    async def score(self, draft: str, case: EvalCase) -> JudgeScore:
        """对生成章节 draft 打分，输入来自 case。解析失败或维度缺失时重试，最多 2 次。"""
        prompt = _build_judge_prompt(draft, case)
        data = None
        for _ in range(self._max_attempts):
            content = await self._request(prompt)
            data = _parse_judge_json(content)
            if data is not None and _has_full_dims(data):
                return self._to_score(data)
        # 重试耗尽：缺失维度 0 分兜底（data 为 None 时整体 0 分）
        return self._to_score(data)
