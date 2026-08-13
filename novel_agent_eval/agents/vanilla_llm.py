# novel_agent_eval/agents/vanilla_llm.py
"""Vanilla LLM 基线：无 Agent 框架，一次 prompt 直接生成章节。

被测对象 = 裸模型。OpenAI 兼容客户端，模型走环境变量：
  BASELINE_API_KEY / BASELINE_BASE_URL / BASELINE_MODEL（缺省 DEFAULT_BASELINE_MODEL）。
client 参数可注入 mock，便于测试不消耗真实 API（与 judge 的构造模式一致）。
"""
import os
import time

from openai import AsyncOpenAI

from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter

DEFAULT_BASELINE_MODEL = "step-3.7-flash"


def build_vanilla_prompt(case: EvalCase) -> str:
    """把 story_outline + previous_context + target_chapter_outline 拼成一次生成的 prompt。"""
    return f"""你是小说作者，请根据以下素材，直接写出本章的小说正文。

## 全书大纲（story_outline）
{case.story_outline}

## 前文上下文（previous_context）
{case.previous_context}

## 本章大纲（target_chapter_outline）
{case.target_chapter_outline}

要求：
- 用中文书写本章正文，目标约 {case.word_target} 字。
- 严格遵循本章大纲，并衔接前文上下文。
- 只输出正文本身，不要任何解释、标题或注释。"""


class VanillaLLMAdapter:
    """Vanilla 基线：单次 LLM 调用生成章节。"""

    name = "vanilla_llm"

    def __init__(self, client=None, model: str | None = None):
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("BASELINE_API_KEY"),
            base_url=os.environ.get("BASELINE_BASE_URL"),
        )
        self._model = model or os.environ.get("BASELINE_MODEL", DEFAULT_BASELINE_MODEL)

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        prompt = build_vanilla_prompt(case)
        # 与主仓库 Writer 对齐：字数目标 ×3 作为 token 上限
        max_tokens = max(2048, int(case.word_target * 3))

        start = time.monotonic()
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=max_tokens,
        )
        elapsed = time.monotonic() - start

        content = (resp.choices[0].message.content or "").strip()

        usage = getattr(resp, "usage", None)
        tokens = None
        if usage is not None:
            tokens = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
            }

        return GeneratedChapter(
            content=content,
            meta={
                "adapter": self.name,
                "model": self._model,
                "elapsed_seconds": round(elapsed, 3),
                "tokens": tokens,
            },
        )
