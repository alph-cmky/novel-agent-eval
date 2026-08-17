# novel_agent_eval/agents/story_bible.py
"""StoryBible (Huxiuzhi / Novel Agent Workspace) 风格对手适配器：
模拟基于全局故事圣经 (Story Bible)、动态角色卡管理与全局世界观上下文注入的创作流水线。
"""
import os
import time

from openai import AsyncOpenAI

from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter, ModelConfig
from novel_agent_eval.dataset.schema import EvalCase


class StoryBibleAdapter(AgentAdapter):
    """策略模拟器（非官方实现）：模拟全局故事圣经与角色卡约束，不产生对应项目的真实成绩。"""

    name = "story_bible"

    def __init__(self, model: ModelConfig | None = None):
        self._model = model or ModelConfig(
            base_url=os.environ.get("STEPFUN_BASE_URL", "https://api.stepfun.com/step_plan/v1"),
            api_key=os.environ.get("STEPFUN_API_KEY", ""),
            model="step-3.7-flash",
        )
        self._client = AsyncOpenAI(
            api_key=self._model.api_key,
            base_url=self._model.base_url,
        )

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        start = time.monotonic()

        # Step 1: 故事圣经提纯 (Story Bible Extraction)
        bible_prompt = f"""你是一位小说架构师与设定集编纂官。
根据全书大纲与前文，整理出本章专用的【故事圣经（Story Bible）】，包含：
1. 涉及角色的当前心理状态与核心动机卡片；
2. 本章触发的世界观底层规则与禁忌；
3. 本章主线推进的关键锚点。

全书大纲：{case.story_outline}
前文梗概：{case.previous_context}
本章大纲：{case.target_chapter_outline}
请输出结构化的小说故事圣经。"""

        bible_resp = await self._client.chat.completions.create(
            model=self._model.model,
            messages=[{"role": "user", "content": bible_prompt}],
            temperature=0.7,
            max_tokens=2048,
            extra_body={"reasoning_effort": "low"},
        )
        story_bible_text = bible_resp.choices[0].message.content or ""

        # Step 2: 注入故事圣经撰写章节正文 (Bible-Guided Generation)
        write_prompt = f"""你是一位职业小说家。请严格遵循以下【故事圣经】中的人设与世界观规则，撰写本章正文：

【故事圣经 (Story Bible)】：
{story_bible_text}

【前文剧情】：
{case.previous_context}

【本章大纲】：
{case.target_chapter_outline}

目标字数：约 {case.word_target} 字。请直接输出小说正文，禁止解释。"""

        write_resp = await self._client.chat.completions.create(
            model=self._model.model,
            messages=[{"role": "user", "content": write_prompt}],
            temperature=0.7,
            max_tokens=8192,
            extra_body={"reasoning_effort": "low"},
        )
        content = write_resp.choices[0].message.content or ""
        elapsed = time.monotonic() - start

        return GeneratedChapter(
            content=content.strip(),
            meta={
                "adapter": self.name,
                "model": self._model.model,
                "elapsed_seconds": round(elapsed, 3),
                "tokens": None,
            },
        )
