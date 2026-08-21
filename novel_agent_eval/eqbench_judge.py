# novel_agent_eval/eqbench_judge.py
"""EQ-Bench Longform 14 维逐章 judge（DeepSeek provider）。

复用 EQ-bench 官方资产（novel_agent_eval/dataset/eqbench/）：
- criteria_chapter.txt / negative_criteria_chapter.txt：14 维（8 正向 + 6 负向）
- criteria_weights.json：forced poetry or metaphor=5 / purple prose=1
- judging_prompt_chapter.txt：逐章评分模板，输出 "Metric Name: [Score]" 0-20 分制

评分公式对齐官方 core/scoring.py：
- 负向维反转 new = 20 - old（官方 invert_if_negative）
- forced poetry or metaphor 反转后 (v/20)^1.7 * 20 缩放
- 单章加权平均 = Σ(processed*weight) / Σ(weight)

judge 走 DeepSeek 官方直连（deepseek-v4-pro）：DeepSeek V4 默认 thinking 开启，
须 extra_body={"thinking":{"type":"disabled"}} 关闭，否则 reasoning 挤空/污染评分
输出。测试用 mock client，不消耗真实 API。
"""

import asyncio
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI

_ASSETS_DIR = Path(__file__).parent / "dataset" / "eqbench"

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"

SCORE_MAX = 20  # EQ-bench 0-20 分制


def _load_lines(filename: str) -> list[str]:
    return [
        ln.strip()
        for ln in (_ASSETS_DIR / filename).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _load_weights() -> dict[str, float]:
    raw = json.loads((_ASSETS_DIR / "criteria_weights.json").read_text(encoding="utf-8"))
    return {k.lower().strip(): float(v) for k, v in raw.items()}


CRITERIA = _load_lines("criteria_chapter.txt")                      # 14 维（8 正向 + 6 负向）
NEGATIVE_CRITERIA = _load_lines("negative_criteria_chapter.txt")    # 6 负向（lower is better）
CRITERIA_WEIGHTS = _load_weights()
_CRITERIA_KEYS = {c.lower().strip() for c in CRITERIA}


def invert_if_negative(metric: str, score: float, negative: list[str] | None = None) -> float:
    """负向维（lower is better）反转：new = 20 - old。非负向维原样返回。"""
    neg = negative if negative is not None else NEGATIVE_CRITERIA
    if metric.lower().strip() in {n.lower().strip() for n in neg}:
        return SCORE_MAX - score
    return score


def eqbench_chapter_score(scores: dict[str, float]) -> float | None:
    """单章 14 维原始分 → 0-20 加权分（对齐官方 calculate_task_score 逐章逻辑）。"""
    # 缺维度是 invalid，不允许用剩余维度算出看似正常的高分。
    if not isinstance(scores, dict) or {k.lower().strip() for k in scores} != _CRITERIA_KEYS:
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for metric, value in scores.items():
        if not isinstance(value, (int, float)):
            continue
        processed = invert_if_negative(metric, value)
        weight = CRITERIA_WEIGHTS.get(metric.lower().strip(), 1.0)
        if metric.lower().strip() == "forced poetry or metaphor":
            processed = (processed / SCORE_MAX) ** 1.7 * SCORE_MAX
        weighted_sum += processed * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None


def parse_eqbench_scores(text: str) -> dict[str, float]:
    """解析 judge 响应 "Metric Name: [Score]" 格式 → {metric: 0-20}。

    对齐官方 parse_judge_scores_longform：优先 [Scores] 段，否则全行扫描；
    过滤常见非指标行，clamp 到 [0, 20]。
    """
    if not text:
        return {}
    collected: list[str] = []
    in_scores = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if "[Scores]" in s or "--- Scores ---" in s:
            in_scores = True
            continue
        if in_scores and ("---" in s or "[End Scores]" in s):
            in_scores = False
            continue
        collected.append(s)

    scores: dict[str, float] = {}
    pattern = re.compile(r"^\s*([^:]+?)\s*:\s*\[?(-?\d+(?:\.\d+)?)\]?")
    inline = re.compile(r"([^:]+?)\s*:\s*\[?(-?\d+(?:\.\d+)?)\]?")
    skip = {"overall assessment", "summary", "reasoning", "critique", "feedback", "notes"}
    for line in collected:
        m = pattern.match(line) or inline.search(line)
        if not m:
            continue
        name = m.group(1).strip()
        if name.lower() in skip:
            continue
        try:
            scores[name] = max(0.0, min(float(SCORE_MAX), float(m.group(2))))
        except ValueError:
            continue
    return scores


class EQBenchJudge:
    """EQ-Bench Longform 逐章 judge（14 维，0-20 分制，DeepSeek）。

    构造与 judge.Judge 一致：client 为 OpenAI 兼容 async client（DeepSeek），
    可注入 mock 供测试。n_samples>1 时并发采样取各维中位数，抑制偶发极端分。
    """

    def __init__(
        self,
        client=None,
        model: str | None = None,
        n_samples: int = 1,
        max_attempts: int = 3,
    ):
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL),
        )
        self._model = model or os.environ.get("DEEPSEEK_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        if n_samples < 1:
            raise ValueError(f"n_samples 必须 >= 1，收到 {n_samples}")
        self._n_samples = n_samples
        self._max_attempts = max_attempts  # 解析为空时重试
        self._template = (_ASSETS_DIR / "judging_prompt_chapter.txt").read_text(encoding="utf-8")

    async def _request(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8192,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return resp.choices[0].message.content or ""

    def _build_prompt(
        self,
        *,
        writing_prompt: str,
        final_plan: str,
        character_profiles: str,
        chapter_number: int,
        chapter_text: str,
    ) -> str:
        criteria = "\n".join(f"- {c}" for c in CRITERIA)
        neg = ", ".join(NEGATIVE_CRITERIA)
        return (
            self._template.replace("{writing_prompt}", writing_prompt)
            .replace("{final_plan}", final_plan)
            .replace("{character_profiles}", character_profiles)
            .replace("{chapter_number}", str(chapter_number))
            .replace("{chapter_text}", chapter_text)
            .replace("{creative_writing_criteria}", criteria)
            .replace("{lower_is_better_criteria}", neg)
        )

    async def _score_once(
        self,
        *,
        writing_prompt: str,
        final_plan: str,
        character_profiles: str,
        chapter_number: int,
        chapter_text: str,
    ) -> dict[str, float]:
        prompt = self._build_prompt(
            writing_prompt=writing_prompt,
            final_plan=final_plan,
            character_profiles=character_profiles,
            chapter_number=chapter_number,
            chapter_text=chapter_text,
        )
        scores: dict[str, float] = {}
        for _ in range(self._max_attempts):
            content = await self._request(prompt)
            scores = parse_eqbench_scores(content)
            if {k.lower().strip() for k in scores} == _CRITERIA_KEYS:
                return scores
        return {}  # 重试耗尽或维度不完整：显式 invalid

    async def score_chapter(
        self,
        *,
        writing_prompt: str,
        final_plan: str,
        character_profiles: str,
        chapter_number: int,
        chapter_text: str,
    ) -> dict[str, float]:
        """逐章 14 维原始分（0-20）。n_samples>1 时并发采样取各维中位数。"""
        kwargs = {
            "writing_prompt": writing_prompt,
            "final_plan": final_plan,
            "character_profiles": character_profiles,
            "chapter_number": chapter_number,
            "chapter_text": chapter_text,
        }
        if self._n_samples <= 1:
            return await self._score_once(**kwargs)
        samples = await asyncio.gather(*(self._score_once(**kwargs) for _ in range(self._n_samples)))
        return self._median(samples)

    @staticmethod
    def _median(samples: list[dict[str, float]]) -> dict[str, float]:
        keys: set[str] = set()
        for s in samples:
            keys.update(s.keys())
        result: dict[str, float] = {}
        for k in keys:
            vals = sorted(s[k] for s in samples if k in s)
            if vals:
                result[k] = vals[len(vals) // 2]
        return result
