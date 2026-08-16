# novel_agent_eval/judge.py
"""LLM Judge — 第三方小说评审，对生成章节按 8 个质量维打分。

使用 StepFun 单模型（OpenAI 兼容协议），不用 Claude/GPT 双 Judge。
测试用 mock client，不消耗真实 API。

输入 = EvalCase（target_chapter_outline / story_outline / previous_context）+ draft。
输出 = JudgeScore（dimensions 8 维各 0-100 + overall）。
"""

import asyncio
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

# 8 质量维 × 5 档严苛 Rubric（参考 EQ-Bench 14维文学判据进行严苛化校准）
# 80-100 档代表出版级/顶级网文质感；61-80 档为合格；41-60 档为明显瑕疵；40 以下为劣质
_RUBRIC_BANDS = ["0-20", "21-40", "41-60", "61-80", "81-100"]
_RUBRIC_TABLE = [
    ("consistency", "连贯性 (Coherence & Profile Fidelity)", [
        "人物改名/身世崩塌/核心设定严重吃书",
        "存在 2 处以上明显的时间线或因果冲突",
        "有 1 处主要矛盾或数处次要细节冲突",
        "基本自洽，人物行为符合已知性格，无关键破绽",
        "严丝合缝，前后伏笔呼应自然，设定完全自洽且具有跨章连续性"
    ]),
    ("writing", "文笔与画面感 (Show Don't Tell)", [
        "充斥大量流水账与抽象情绪告知 (Tell)，无任何微动作与五感细节",
        "文笔平淡生硬，修辞堆砌过度 (Purple Prose) 或语言干瘪",
        "通顺连贯，但描写较为常规套路，缺少令人眼前一亮的镜头感",
        "动作细节生动，善于通过环境与微反应烘托氛围 (Good Show)",
        "具有强烈的画面张力与高级文学质感，炼字精准，情绪递进层次分明"
    ]),
    ("ai_flavor", "去AI味 (Anti-AI Flavor)", [
        "机器味极浓：大量「总之/可以说/然而/不禁/宛如/仿佛」，末尾强行哲学升华或大团圆总结",
        "存在明显公文腔、说教腔、排比句泛滥，千人一面",
        "轻微 AI 套路，结尾有轻微概括，但主体叙事尚属自然",
        "自然流畅的网文/小说口吻，无明显 AI 习惯性套话与升华总结",
        "完全消除 AI 味，语言纯正地道，富有作者个人风格与生活化质感"
    ]),
    ("dialogue", "对话质量 (Nuanced Dialogue)", [
        "所有角色说话千篇一律 (Weak Dialogue)，充斥播音腔与说明性台词",
        "对话生硬尴尬，无潜台词，仅作为信息交代工具",
        "对话能基本区分身份，但缺少语气交锋与人物性格张力",
        "个性鲜明，对白有来有往且带有明确动机与潜台词",
        "闻声识人！对话充满智商交锋与幽微拉扯，语带机锋，极具张力"
    ]),
    ("plot", "情节张力 (Compelling Plot & Pacing)", [
        "流水账且毫无波澜，节奏失控，事件无因果推进",
        "情节平铺直叙，缺少有效冲突铺垫与悬念设计",
        "节奏尚可，有基本的起承转合与情节推进，但转折较为生硬",
        "起伏有致，场景冲突激烈，有扣人心弦的情绪高潮或意外转折",
        "节奏把控炉火纯青，环环相扣，悬念与断章 (Cliffhanger) 恰到好处"
    ]),
    ("instruction", "大纲与指令还原 (Fidelity to Outline)", [
        "严重偏离大纲核心事件，擅自篡改关键人物行动与结果",
        "仅覆盖不足一半大纲节点，关键事件被一笔带过或遗漏",
        "基本覆盖大纲主线事件，但细节展开不充分或有所妥协",
        "较好还原大纲全部要点，主线与支线衔接顺畅",
        "精准完美还原大纲全部事件与细节指令，且艺术化展开极其饱满"
    ]),
    ("creativity", "创意与意外感 (Unsurprising vs Creative)", [
        "通篇陈词滥调、毫无新意的模板化套路 (Uncreative)",
        "剧情走向完全在读者意料之中，缺少新颖视角或设定亮点",
        "有局部小亮点或设定巧思，但整体框架较为常规",
        "情节发展有意外转折且合情合理，细节呈现新颖视角",
        "惊艳的构思与破局方式！脑洞大开同时逻辑自洽，极具创新深度"
    ]),
    ("controllability", "可操控与反馈响应 (Controllability)", [
        "完全无视前置指导或修改要求，我行我素",
        "仅做表面文字微调，未触及任何核心缺陷或指令重点",
        "对重点反馈有部分响应，但仍有关键意见未落实",
        "准确执行大部分修改意见与篇幅/风格约束，改到位",
        "完美执行全部精细化指令，举一反三且主动强化薄弱维度"
    ]),
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


def _extract_dims(data: dict | None) -> dict:
    """从 Judge 输出提取 8 维分值 dict，兼容嵌套与平铺两种格式。

    step-3.7-flash（reasoning 模型）常把 8 维平铺到顶层（漏 dimensions 外壳、
    漏 overall），实测 content 形如 {"consistency": 20, "writing": 10, ...}。
    此处把嵌套 {"dimensions": {...}} 与平铺格式统一成维度 dict。
    """
    if not isinstance(data, dict):
        return {}
    dims = data.get("dimensions")
    if isinstance(dims, dict):
        return dims
    # 平铺格式：8 维键直接在顶层
    return {d: data.get(d) for d in QUALITY_DIMS if d in data}


def _has_full_dims(data: dict) -> bool:
    dims = _extract_dims(data)
    return isinstance(dims, dict) and all(d in dims for d in QUALITY_DIMS)


class Judge:
    """第三方小说评审。client 为 OpenAI 兼容 async client（StepFun）。

    不传 client 时按环境变量构造：STEPFUN_API_KEY / STEPFUN_BASE_URL /
    STEPFUN_JUDGE_MODEL（默认 DEFAULT_JUDGE_MODEL）。
    """

    def __init__(
        self,
        client=None,
        model: str | None = None,
        max_attempts: int = 3,
        n_samples: int = 1,
    ):
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("STEPFUN_API_KEY"),
            base_url=os.environ.get("STEPFUN_BASE_URL"),
        )
        self._model = model or os.environ.get("STEPFUN_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self._max_attempts = max_attempts  # 初次 + 最多 2 次重试
        if n_samples < 1:
            raise ValueError(f"n_samples 必须 >= 1，收到 {n_samples}")
        self._n_samples = n_samples  # 中位数采样次数（>1 时取各维中位数，抑制偶发极端分）

    async def _request(self, prompt: str) -> str:
        # step-3.7-flash 是 reasoning 模型，这里的三项设置都是实测校准：
        # - max_tokens=8192：2048 会让真实 Judge prompt（rubric + 全文 draft）的
        #   content 被 reasoning 挤空（实测 4096 仍 finish=length，8192 才稳定产出）。
        # - reasoning_effort="low"：官方入参（low/medium/high），把 reasoning 从 ~6800
        #   压到 ~2700 token，减少间歇空 content 与 step-3.7 已知的 overthinking
        #   （default 档会把 instruction 维打到 35 这类极端分），打分质量不降反稳。
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8192,
            reasoning_effort="low",
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _to_score(data: dict | None) -> JudgeScore:
        """把（可能不完整的）解析结果规范化为 JudgeScore：缺失维度给 0；overall 缺失退化为 8 维平均。"""
        dims = _extract_dims(data)
        normalized = {d: (_coerce_score(dims.get(d)) or 0) for d in QUALITY_DIMS}
        overall = _coerce_score((data or {}).get("overall"))
        if overall is None:
            overall = round(sum(normalized.values()) / len(QUALITY_DIMS))
        return JudgeScore(dimensions=normalized, overall=overall)

    async def score(self, draft: str, case: EvalCase) -> JudgeScore:
        """对生成章节 draft 打分，输入来自 case。

        n_samples>1 时并发连打 n_samples 次取各维中位数，抑制 reasoning 模型偶发
        极端分（探针确认 consistency 维 range=55，其余 7 维稳定）；单次采样内部仍
        保留「解析失败/维度缺失」的重试兜底。
        """
        if self._n_samples <= 1:
            return await self._score_once(draft, case)
        samples = await asyncio.gather(
            *(self._score_once(draft, case) for _ in range(self._n_samples))
        )
        return self._median_scores(samples)

    async def _score_once(self, draft: str, case: EvalCase) -> JudgeScore:
        """单次打分：解析失败或维度缺失时重试，最多 max_attempts 次。"""
        prompt = _build_judge_prompt(draft, case)
        data = None
        for _ in range(self._max_attempts):
            content = await self._request(prompt)
            data = _parse_judge_json(content)
            if data is not None and _has_full_dims(data):
                return self._to_score(data)
        # 重试耗尽：缺失维度 0 分兜底（data 为 None 时整体 0 分）
        return self._to_score(data)

    @staticmethod
    def _median_scores(samples: list[JudgeScore]) -> JudgeScore:
        """对多次采样取各维中位数（奇数取中位，偶数取上中位，整数分无需取平均）。"""
        n = len(samples)

        def _med(vals: list[int]) -> int:
            return sorted(vals)[n // 2]

        dims = {d: _med([s.dimensions[d] for s in samples]) for d in QUALITY_DIMS}
        overall = _med([s.overall for s in samples])
        return JudgeScore(dimensions=dims, overall=overall)
