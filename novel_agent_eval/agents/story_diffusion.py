# novel_agent_eval/agents/story_diffusion.py
"""StoryDiffusion / LongWriter-Pipeline 对手适配器：
模拟开源经典 Multi-Pass Planning & Expanding 故事创作流水线（大纲生成 -> 场景扩写 -> 章节润色）。
"""
import os
import time
from openai import AsyncOpenAI

from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter, ModelConfig
from novel_agent_eval.dataset.schema import EvalCase


class StoryDiffusionAdapter(AgentAdapter):
    """基于经典两阶段 (Outline Planning -> Multi-Scene Expansion) 的开源故事生成 Agent。"""

    name = "story_diffusion"

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

        # 阶段 1: 细化场景分镜大纲 (Scene Outline Planning)
        plan_prompt = f"""你是一个小说分镜规划师。根据以下信息，将本章规划为 3-4 个具体的场景分镜：
全书背景：{case.story_outline}
前文梗概：{case.previous_context}
本章大纲：{case.target_chapter_outline}
请输出清晰的分镜规划（场景一、场景二、场景三...）。"""

        plan_resp = await self._client.chat.completions.create(
            model=self._model.model,
            messages=[{"role": "user", "content": plan_prompt}],
            temperature=0.7,
            max_tokens=2048,
            extra_body={"reasoning_effort": "low"},
        )
        scenes_plan = plan_resp.choices[0].message.content or ""

        # 阶段 2: 依据分镜进行长篇正文扩写 (Multi-Scene Expansion)
        write_prompt = f"""你是一位职业小说家。根据以下分镜大纲，撰写出完整、生动的小说章节正文。
全书大纲：{case.story_outline}
前文：{case.previous_context}
分镜规划：{scenes_plan}
目标篇幅：约 {case.word_target} 字。请直接输出小说正文，禁止解释。"""

        write_resp = await self._client.chat.completions.create(
            model=self._model.model,
            messages=[{"role": "user", "content": write_prompt}],
            temperature=0.7,
            max_tokens=8192,
            extra_body={"reasoning_effort": "low"},
        )
        content = (write_resp.choices[0].message.content or "").strip()
        elapsed = time.monotonic() - start

        return GeneratedChapter(
            content=content,
            meta={
                "adapter": "story_diffusion",
                "elapsed_seconds": round(elapsed, 3),
                "scenes_plan_chars": len(scenes_plan),
            },
        )
