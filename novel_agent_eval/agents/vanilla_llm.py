# novel_agent_eval/agents/vanilla_llm.py
"""Vanilla LLM 基线：无 Agent 框架，一次 prompt 直接生成章节。

被测对象 = 裸模型。OpenAI 兼容客户端，模型走环境变量：
  BASELINE_API_KEY / BASELINE_BASE_URL / BASELINE_MODEL（缺省 DEFAULT_BASELINE_MODEL）。
client 参数可注入 mock，便于测试不消耗真实 API（与 judge 的构造模式一致）。

50 章 durable 对照（Phase 5）：与 NovelAgentAdapter 同一条 run_baseline 评分链路，
需要（1）`_chapter_number`（durable.py 契约）；（2）V2 DB 写入——baseline 的
`_read_chapter_content` 从 chapters 表读 committed 正文，Vanilla 用 save_chapter
UPSERT 最小写入，不建 writing_run / canon（架构对照的意义正是无这些机制）。
resume 语义与 durable 对齐：chapters 表已有正文即跳过（跨进程续跑）。

多轮记忆模式（memory_mode）：
  "none"   纯大纲生成，previous_context 恒空——裸模型基线（默认）
  "full"   拼接全部前章全文——朴素长上下文基线（对照 Canon 结构化投影）
"""
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

from novel_agent_eval.agents.base import GeneratedChapter
from novel_agent_eval.dataset.schema import EvalCase

DEFAULT_BASELINE_MODEL = "step-3.7-flash"


def build_vanilla_prompt(case: EvalCase, previous_text: str = "") -> str:
    """把 story_outline + previous_context + target_chapter_outline 拼成一次生成的 prompt。"""
    prev_block = previous_text or (case.previous_context or "")
    return f"""你是小说作者，请根据以下素材，直接写出本章的小说正文。

## 全书大纲（story_outline）
{case.story_outline}

## 前文上下文（previous_context）
{prev_block}

## 本章大纲（target_chapter_outline）
{case.target_chapter_outline}

要求：
- 用中文书写本章正文，目标约 {case.word_target} 字。
- 严格遵循本章大纲，并衔接前文上下文。
- 只输出正文本身，不要任何解释、标题或注释。"""


class VanillaLLMAdapter:
    """Vanilla 基线：单次 LLM 调用生成章节。"""

    name = "vanilla_llm"

    def __init__(
        self,
        client=None,
        model: str | None = None,
        persist_dir: str | None = None,
        resume: bool = False,
        memory_mode: str = "none",
    ):
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("BASELINE_API_KEY"),
            base_url=os.environ.get("BASELINE_BASE_URL"),
        )
        self._model = model or os.environ.get("BASELINE_MODEL", DEFAULT_BASELINE_MODEL)
        self._persist_dir = persist_dir
        self.resume = resume
        if memory_mode not in ("none", "full"):
            raise ValueError(f"memory_mode 须为 none/full，收到 {memory_mode!r}")
        self.memory_mode = memory_mode
        self._project_id: str | None = None
        self._manager = None

    @staticmethod
    def _chapter_number(case: EvalCase) -> int:
        """durable.py 契约：从 case.name 尾部解析章号（baseline{N}_s{k}_ch{NN}）。"""
        return int(case.name.rsplit("_ch", 1)[-1])

    def _ensure_project(self, case: EvalCase):
        """懒初始化 V2 project（评分链路需要 chapters 表有正文）。"""
        if self._project_id is not None:
            return self._project_id
        if self._persist_dir is None:
            return None
        from novel_agent.storage.manager import ProjectManager

        self._manager = ProjectManager(Path(self._persist_dir))
        project_name = case.project_id or f"eval:{case.name}"
        if self.resume:
            for p in self._manager.list_projects():
                if p.get("name") == project_name:
                    self._project_id = p["id"]
                    return self._project_id
        self._project_id = self._manager.init_project(
            name=project_name,
            title=project_name,
            story_length="long",
            target_chapter_words=case.word_target,
        )
        return self._project_id

    def _resumed_content(self, project_id: str, chapter_number: int) -> str | None:
        """resume：chapters 表已有正文 → 返回，跳过重复生成。"""
        ch = self._manager.get_chapter(project_id, chapter_number)
        content = (ch or {}).get("draft_content", "") if ch else ""
        return content if (content or "").strip() else None

    def _previous_text(self, project_id: str, chapter_number: int) -> str:
        """memory_mode=full：拼接全部前章全文（朴素长上下文）。"""
        if self.memory_mode != "full" or self._manager is None or project_id is None:
            return ""
        parts = []
        for n in range(1, chapter_number):
            ch = self._manager.get_chapter(project_id, n)
            text = (ch or {}).get("draft_content", "") if ch else ""
            if text.strip():
                parts.append(f"### 第{n}章\n{text.strip()}")
        return "\n\n".join(parts)

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        chapter_number = self._chapter_number(case)
        project_id = self._ensure_project(case)

        if project_id and self.resume:
            resumed = self._resumed_content(project_id, chapter_number)
            if resumed is not None:
                return GeneratedChapter(
                    content=resumed,
                    meta={
                        "adapter": self.name,
                        "model": self._model,
                        "resumed": True,
                        "tokens": None,
                        "elapsed_seconds": 0.0,
                    },
                )

        previous_text = self._previous_text(project_id, chapter_number) if project_id else ""
        prompt = build_vanilla_prompt(case, previous_text)
        # 与主仓库 Writer 对齐：字数目标 ×3 作为 token 上限
        max_tokens = max(2048, int(case.word_target * 3))

        start = time.monotonic()
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.85,
            "max_tokens": max_tokens,
        }
        # thinking 控制与主仓库/Judge 同口径：LLM_THINKING_DISABLED → extra_body
        # （DeepSeek v4 / qwen3.8-flash 等 hybrid 模型）；reasoning 模型走 low effort。
        if os.environ.get("LLM_THINKING_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["reasoning_effort"] = "low"
        resp = await self._client.chat.completions.create(**kwargs)
        elapsed = time.monotonic() - start

        content = (resp.choices[0].message.content or "").strip()

        usage = getattr(resp, "usage", None)
        tokens = None
        if usage is not None:
            tokens = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
            }

        if project_id and self._manager is not None:
            self._manager.save_chapter(
                project_id,
                chapter_number,
                outline=case.target_chapter_outline,
                draft_content=content,
                status="approved",
            )

        return GeneratedChapter(
            content=content,
            meta={
                "adapter": self.name,
                "model": self._model,
                "memory_mode": self.memory_mode,
                "elapsed_seconds": round(elapsed, 3),
                # durable.py checkpoint_from_meta 读 "token_usage" 键
                "token_usage": {"total_tokens": (tokens or {}).get("input", 0) + (tokens or {}).get("output", 0), **(tokens or {})},
                "tokens": tokens,
            },
        )
