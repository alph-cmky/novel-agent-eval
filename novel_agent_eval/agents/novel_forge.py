# novel_agent_eval/agents/novel_forge.py
"""NovelForge 对手适配器：
模拟开源 NovelForge 的核心范式 —— 三层账本管理 (Bookkeeping System) 与对抗式审查修复 (Adversarial Review & Fixer)。
"""
import os
import time
from openai import AsyncOpenAI

from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter, ModelConfig
from novel_agent_eval.dataset.schema import EvalCase


class NovelForgeAdapter(AgentAdapter):
    """基于三层账本管理与对抗式审查修复 (Adversarial Review) 的 NovelForge 风格 Agent。"""

    name = "novel_forge"

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

        # Step 1: 账本提取与情节规划 (Bookkeeping & Chapter Planning)
        plan_prompt = f"""你是一个小说项目账本管理员与大纲规划师。
根据以下信息，提取本章的核心事实账本（关键角色、关键道具、地理位置、设定规则），并规划本章正文大纲：
全书背景：{case.story_outline}
前文梗概：{case.previous_context}
本章大纲：{case.target_chapter_outline}
请输出：1. 核心事实账本；2. 详细章节大纲。"""

        plan_resp = await self._client.chat.completions.create(
            model=self._model.model,
            messages=[{"role": "user", "content": plan_prompt}],
            temperature=0.7,
            max_tokens=2048,
            extra_body={"reasoning_effort": "low"},
        )
        ledger_plan = plan_resp.choices[0].message.content or ""

        # Step 2: 依据账本进行草稿生成 (Draft Generation)
        draft_prompt = f"""你是一位小说创作者。根据事实账本和大纲，撰写本章初稿：
事实账本与大纲：{ledger_plan}
目标字数：约 {case.word_target} 字。请直接输出章节正文，不要输出解释。"""

        draft_resp = await self._client.chat.completions.create(
            model=self._model.model,
            messages=[{"role": "user", "content": draft_prompt}],
            temperature=0.7,
            max_tokens=8192,
            extra_body={"reasoning_effort": "low"},
        )
        draft_text = draft_resp.choices[0].message.content or ""

        # Step 3: 对抗式事实审查与修复 (Adversarial Evaluator & Fixer)
        fix_prompt = f"""你是一位严苛的对抗式小说审稿人（Adversarial Evaluator）。
请对照以下核心事实账本，审查初稿中是否存在设定违背、事实冲突或逻辑硬伤：
核心账本：{ledger_plan}
初稿正文：{draft_text}

请在保留初稿优点的基础上，修正所有事实硬伤与情节违背，输出最终精修后的章节正文（直接输出小说正文，禁止解释）。"""

        final_resp = await self._client.chat.completions.create(
            model=self._model.model,
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.7,
            max_tokens=8192,
            extra_body={"reasoning_effort": "low"},
        )
        final_text = final_resp.choices[0].message.content or draft_text
        elapsed = time.monotonic() - start

        return GeneratedChapter(
            content=final_text.strip(),
            meta={
                "adapter": self.name,
                "model": self._model.model,
                "elapsed_seconds": round(elapsed, 3),
                "tokens": None,
            },
        )
