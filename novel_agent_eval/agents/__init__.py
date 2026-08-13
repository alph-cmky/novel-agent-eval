# novel_agent_eval/agents — 被测 Agent 适配器包
from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter

__all__ = [
    "AgentAdapter",
    "GeneratedChapter",
    "NovelAgentAdapter",
    "VanillaLLMAdapter",
]
