# novel_agent_eval/dataset/loader.py
import json
from pathlib import Path

from .schema import EvalCase


def load_cases(dir_path: str) -> list[EvalCase]:
    cases = []
    for p in sorted(Path(dir_path).glob("*.json")):
        cases.append(EvalCase.model_validate_json(p.read_text(encoding="utf-8")))
    return cases


def load_external_constory_cases(json_path: str, max_cases: int = 10) -> list[EvalCase]:
    """从 constory JSON 数据集加载为标准 EvalCase 列表。"""
    p = Path(json_path)
    if not p.exists():
        return []
    raw_list = json.loads(p.read_text(encoding="utf-8"))
    cases = []
    for item in raw_list[:max_cases]:
        case = EvalCase(
            name=f"constory_{item.get('id', len(cases)+1)}",
            stage="long",
            genre="通用故事",
            story_outline=item.get("prompt", ""),
            previous_context="",
            target_chapter_outline=item.get("prompt", ""),
            word_target=2000,
        )
        cases.append(case)
    return cases
