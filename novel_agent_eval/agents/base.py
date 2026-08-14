# novel_agent_eval/agents/base.py
"""被测 Agent 统一接口：把不同被测对象（novel-agent 流水线 / Vanilla LLM 基线）
收敛成同一个 `generate(case) -> GeneratedChapter` 接口，供 runner 横评调用。
"""
from dataclasses import dataclass
from typing import Protocol

from novel_agent_eval.dataset.schema import EvalCase


class GeneratedChapter:
    """一次生成的章节产出。

    meta 采集生成过程的可观测信号，供 Task 9 internal_signals 使用，
    至少含：elapsed_seconds / tokens / composite_score / evolution_rounds /
    evolution_termination（novel-agent 有进化字段，Vanilla 无）。
    """

    def __init__(self, content: str, meta: dict | None = None):
        self.content = content
        self.meta = meta or {}


@dataclass
class ModelConfig:
    """对手适配器用模型配置（OpenAI 兼容），§5.3 同模型消融 / 最优模型对比的注入点。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7


class AgentAdapter(Protocol):
    name: str

    async def generate(self, case: EvalCase) -> GeneratedChapter: ...
