"""External benchmark dimensions mapped to internal evolution capabilities.

This module deliberately maps evidence and repair targets; it does not merge
external scores into the production composite score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionMapping:
    source: str
    external_dimension: str
    internal_dimensions: tuple[str, ...]
    hard_constraint: bool
    repair_instruction: str


EQBENCH_MAPPINGS = (
    DimensionMapping("eqbench", "Nuanced Characters", ("writing", "character_fidelity"), False, "增加角色层次和稳定的行为动机"),
    DimensionMapping("eqbench", "Emotionally Engaging", ("writing", "emotional_arc"), False, "补足情绪递进和可观察的情绪变化"),
    DimensionMapping("eqbench", "Compelling Plot", ("rhythm", "logic", "plot_progression"), False, "加强因果推进、冲突升级和场景节奏"),
    DimensionMapping("eqbench", "Coherent", ("logic", "timeline_consistency"), True, "修复事件顺序和因果链，不改变已确认事实"),
    DimensionMapping("eqbench", "Characters Consistent with Profile", ("character_fidelity",), True, "保持角色档案中的身份、能力和动机"),
    DimensionMapping("eqbench", "Followed Chapter Plan", ("outline_adherence",), True, "保留并完成本章大纲中的全部必需节点"),
    DimensionMapping("eqbench", "Faithful to Writing Prompt", ("instruction",), True, "严格遵循原始创作要求和体裁约束"),
    DimensionMapping("eqbench", "Weak Dialogue", ("dialogue",), False, "增加潜台词、角色声音差异和动作反馈"),
    DimensionMapping("eqbench", "Tell-Don't-Show", ("writing", "show_dont_tell"), False, "用动作、感官和场景反应替代抽象情绪告知"),
    DimensionMapping("eqbench", "Unsurprising or Uncreative", ("creativity",), False, "避免模板化转折，增加合乎逻辑的意外性"),
    DimensionMapping("eqbench", "Amateurish", ("writing",), False, "修复基础表达、段落组织和叙述成熟度问题"),
    DimensionMapping("eqbench", "Purple Prose", ("writing", "ai_flavor"), False, "减少堆砌修辞，保留精准有效的意象"),
    DimensionMapping("eqbench", "Forced Poetry or Metaphor", ("writing", "metaphor_coherence"), False, "删除不服务于场景和情绪的强行比喻"),
)


CONSTORY_MAPPINGS = (
    DimensionMapping("constory", "characterization", ("character_fidelity",), True, "修复角色记忆、知识、能力和外貌冲突"),
    DimensionMapping("constory", "factual_detail", ("entity_fidelity", "worldbuilding_consistency"), True, "保持名称、数量、属性和实体状态一致"),
    DimensionMapping("constory", "narrative_style", ("ai_flavor", "style_consistency"), False, "保持视角、语气和文风连续"),
    DimensionMapping("constory", "timeline_plot", ("timeline_consistency", "logic", "plot_progression"), True, "修复时间顺序、因果关系和未完成剧情线"),
    DimensionMapping("constory", "world_building", ("worldbuilding_consistency",), True, "遵守世界规则、地理关系和社会规范"),
)


def mapping_for(source: str, external_dimension: str) -> DimensionMapping:
    """Return a mapping or raise a clear error for an unmapped dimension."""
    candidates = EQBENCH_MAPPINGS if source == "eqbench" else CONSTORY_MAPPINGS
    for mapping in candidates:
        if mapping.external_dimension == external_dimension:
            return mapping
    raise KeyError(f"未映射的外部维度: {source}/{external_dimension}")


def evidence_to_repair_instruction(
    source: str,
    external_dimension: str,
    *,
    evidence: str = "",
) -> dict:
    """Convert external evidence into a structured internal repair target."""
    mapping = mapping_for(source, external_dimension)
    instruction = mapping.repair_instruction
    if evidence:
        instruction = f"{instruction}。具体证据：{evidence}"
    return {
        "source": source,
        "external_dimension": external_dimension,
        "internal_dimensions": list(mapping.internal_dimensions),
        "hard_constraint": mapping.hard_constraint,
        "repair_instruction": instruction,
    }
