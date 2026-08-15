# novel_agent_eval/eqbench_bridge.py
"""EQ-Bench Longform bridge 层：用独立 LLM 跑官方 5 步 planning prompt，
产出 final_plan（step4）+ character_profiles（step5），并映射为 8 个 EvalCase。

对齐官方 core/conversation.py 的 planning 流程：
  step1 brainstorm → step2 plan → step3 critique → step4 final_plan → step5 characters
五步是同一段多轮对话：第 i 步的 messages = 前面每步的 user 模板 + assistant 输出，
最后 append 当前步 user 模板。final_plan 取 step4 输出，character_profiles 取 step5。

bridge 用「独立 LLM」：与被测 agent、judge 解耦，走 BRIDGE_* 环境变量（缺省
fallback 到 DeepSeek，与 judge 同 key 但模型可单独覆盖）。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from novel_agent_eval.dataset.schema import EvalCase

_ASSETS_DIR = Path(__file__).parent / "dataset" / "eqbench"
_PLAN_PROMPTS_DIR = _ASSETS_DIR / "plan_prompts"

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_BRIDGE_MODEL = "deepseek-v4-pro"

# 官方 plan prompt 文件名 → 步序号（1-based，对齐 prompt1..5）
_STEP_FILES = {
    1: "1_brainstorm.txt",
    2: "2_plan.txt",
    3: "3_critique.txt",
    4: "4_final_plan.txt",
    5: "5_characters.txt",
}
NUM_PLANNING_STEPS = 5


@dataclass
class LongformPlan:
    """bridge 产出的 planning 结果：final_plan + character_profiles + 每步原始输出。"""

    prompt_id: str
    title: str
    category: str
    writing_prompt: str
    n_chapters: int
    final_plan: str                       # step4 输出（# Intention + # Chapter Planning）
    character_profiles: str               # step5 输出（# Character N ...）
    step_outputs: dict[str, str] = field(default_factory=dict)  # 原始 5 步输出（调试）


class EQBenchBridge:
    """独立 LLM 跑官方 planning 5 步，产出 LongformPlan。client 可注入 mock 供测试。"""

    def __init__(self, client=None, model: str | None = None, n_chapters: int = 8):
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("BRIDGE_API_KEY", os.environ.get("DEEPSEEK_API_KEY")),
            base_url=os.environ.get("BRIDGE_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL),
        )
        self._model = model or os.environ.get("BRIDGE_MODEL", DEFAULT_BRIDGE_MODEL)
        self._n_chapters = n_chapters
        self._templates = {
            i: (_PLAN_PROMPTS_DIR / f).read_text(encoding="utf-8")
            for i, f in _STEP_FILES.items()
        }

    def _render(self, template: str, writing_prompt: str) -> str:
        return template.replace("{writing_prompt}", writing_prompt).replace(
            "{n_chapters}", str(self._n_chapters)
        )

    def _build_messages(self, step_num: int, writing_prompt: str, outputs: dict[str, str]) -> list[dict]:
        """第 step_num 步的完整多轮历史（对齐官方 _build_messages_for_step）。"""
        messages = []
        for i in range(1, step_num):
            prev = self._render(self._templates[i], writing_prompt)
            messages.append({"role": "user", "content": prev})
            if str(i) in outputs:
                messages.append({"role": "assistant", "content": outputs[str(i)]})
        messages.append(
            {"role": "user", "content": self._render(self._templates[step_num], writing_prompt)}
        )
        return messages

    async def _request(self, messages: list[dict]) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.9,      # planning 发散创作
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""

    async def plan(
        self,
        writing_prompt: str,
        *,
        prompt_id: str = "",
        title: str = "",
        category: str = "",
    ) -> LongformPlan:
        """跑 5 步 planning，返回 LongformPlan（final_plan=step4，characters=step5）。"""
        outputs: dict[str, str] = {}
        for step_num in range(1, NUM_PLANNING_STEPS + 1):
            messages = self._build_messages(step_num, writing_prompt, outputs)
            outputs[str(step_num)] = await self._request(messages)
        return LongformPlan(
            prompt_id=prompt_id,
            title=title,
            category=category,
            writing_prompt=writing_prompt,
            n_chapters=self._n_chapters,
            final_plan=outputs["4"],
            character_profiles=outputs["5"],
            step_outputs=outputs,
        )


def plan_to_cases(plan: LongformPlan, *, word_target: int = 1000) -> list[EvalCase]:
    """LongformPlan → n_chapters 个 EvalCase（连载链，previous_context 运行时填充）。

    对齐官方 chapter 生成：被测模型无多轮历史，须把完整 plan（final_plan +
    character_profiles）塞进 story_outline，target_chapter_outline 只给「按计划写
    第 i 章」指令。每章 previous_context 留空，由 run_longform 逐章累积正文回填。
    """
    story_outline = "\n\n".join(
        s
        for s in [
            plan.writing_prompt.strip(),
            f"## Character Profiles\n{plan.character_profiles}".strip(),
            f"## Writing Plan\n{plan.final_plan}".strip(),
        ]
        if s
    )
    cases: list[EvalCase] = []
    for i in range(1, plan.n_chapters + 1):
        cases.append(
            EvalCase(
                name=f"eqbench_p{plan.prompt_id}_ch{i:02d}",
                stage="opening",  # 8 章短篇 → 前几章语义；EQ-Bench 评测不走 weighted_score，仅占位
                genre=plan.category or "general",
                story_outline=story_outline,
                previous_context="",  # run_longform 逐章回填
                target_chapter_outline=(
                    f"Follow your plan. Write chapter {i} of {plan.n_chapters}. "
                    f"~{word_target} words."
                ),
                word_target=word_target,
            )
        )
    return cases
