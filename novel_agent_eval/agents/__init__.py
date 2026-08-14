# novel_agent_eval/agents — 被测 Agent 适配器包
from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter, ModelConfig
from novel_agent_eval.agents.inkos import InkOSAdapter
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.agents.novel_writing import NovelWritingAgentAdapter
from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter

__all__ = [
    "AgentAdapter",
    "GeneratedChapter",
    "InkOSAdapter",
    "ModelConfig",
    "NovelAgentAdapter",
    "NovelWritingAgentAdapter",
    "VanillaLLMAdapter",
]
