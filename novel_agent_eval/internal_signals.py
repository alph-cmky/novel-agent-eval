# novel_agent_eval/internal_signals.py
"""内部信号采集 — 从原始 LangGraph graph state 提取 novel-agent 的自我评估信号。

供后续报告分析「novel-agent 的自评分是否与外部 Judge 分一致」。评分公式
完全复用主仓库 novel_agent.graph.evolution 的 extract_scores / composite_score，
不重复实现（全局约束）。

歧义解析（task-9 brief 权威版）：
  A. 优先用最终选定版本的报告（evolution_best_editor_report /
     evolution_best_continuity_report）；仅当 evolution_enabled 为真且两个 best
     报告都存在（非空 dict）时采用。否则回退当前轮 editor_report / continuity_report
     （如 evolution_enabled=False 的 legacy 路径）。
  B. Continuity 原始数据没有 per-category 数值分，只有 inconsistencies 列表 →
     continuity_by_category 为各 category 的不一致条数（不发明 severity 权重）。
"""
from dataclasses import dataclass

from novel_agent.graph.evolution import composite_score, extract_scores

# Continuity 3 类不一致（对齐 novel_agent/agents/continuity.py 的 category 枚举）
CONTINUITY_CATEGORIES = ("character", "timeline", "worldbuilding")


def _as_int(value, default: int = 0) -> int:
    """防御性 int 转换；None / 非法值 → default。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class InternalSignals:
    """novel-agent 对最终输出的内部自评估信号。"""

    composite_score: float
    evolution_round: int
    termination_reason: str
    editor_overall: int
    editor_dimensions: dict[str, int]  # 5 维：rhythm/ai_flavor/dialogue/logic/writing
    continuity_overall: int
    continuity_by_category: dict[str, int]  # 3 类：character/timeline/worldbuilding 条数


class InternalSignalCollector:
    """从 graph state dict 提取内部信号；对部分填充的 state 防御式退化。"""

    @staticmethod
    def _pick_reports(state: dict) -> tuple[dict, dict]:
        """选择评分来源报告：优先最佳版本报告，缺省回退当前轮报告。"""
        if state.get("evolution_enabled"):
            best_editor = state.get("evolution_best_editor_report") or {}
            best_continuity = state.get("evolution_best_continuity_report") or {}
            if best_editor and best_continuity:
                return best_editor, best_continuity
        editor_report = state.get("editor_report") or {}
        continuity_report = state.get("continuity_report") or {}
        return editor_report, continuity_report

    @staticmethod
    def _count_by_category(continuity_report: dict) -> dict[str, int]:
        """按 category 统计 inconsistencies 条数（原始数据无 per-category 数值分）。"""
        counts = {c: 0 for c in CONTINUITY_CATEGORIES}
        for item in continuity_report.get("inconsistencies") or []:
            category = item.get("category") if isinstance(item, dict) else None
            if category in counts:
                counts[category] += 1
        return counts

    def collect(self, graph_state: dict) -> InternalSignals:
        state = graph_state or {}
        editor_report, continuity_report = self._pick_reports(state)
        # extract_scores 读取字面键 editor_report / continuity_report，
        # 统一用合成 dict 喂入（best 或当前轮报告均可）。
        scores = extract_scores(
            {"editor_report": editor_report, "continuity_report": continuity_report}
        )
        return InternalSignals(
            composite_score=composite_score(scores),
            evolution_round=_as_int(state.get("evolution_round")),
            termination_reason=str(state.get("evolution_termination") or ""),
            editor_overall=scores["editor_overall"],
            editor_dimensions=scores["dimensions"],
            continuity_overall=scores["continuity_overall"],
            continuity_by_category=self._count_by_category(continuity_report),
        )
