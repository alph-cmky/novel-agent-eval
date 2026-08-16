# novel_agent_eval/dataset/loader.py
import json
from pathlib import Path
from .schema import EvalCase


def load_cases(dir_path: str) -> list[EvalCase]:
    cases = []
    for p in sorted(Path(dir_path).glob("*.json")):
        cases.append(EvalCase.model_validate_json(p.read_text(encoding="utf-8")))
    return cases


def load_external_benchmark_cases(json_path: str, benchmark_type: str = "litbench") -> list[EvalCase]:
    """从 external_benchmarks.json 加载指定开源评测集（litbench / longwriter / storybench）。"""
    p = Path(json_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get(benchmark_type.lower().strip(), [])
    cases = []
    for item in items:
        case = EvalCase(
            name=f"{benchmark_type}_{item.get('id', len(cases)+1)}",
            stage="long" if benchmark_type in ("longwriter", "storybench") else "opening",
            genre=item.get("genre", "通用故事"),
            story_outline=item.get("prompt", ""),
            previous_context="",
            target_chapter_outline=item.get("prompt", ""),
            word_target=item.get("word_target", 3000),
        )
        cases.append(case)
    return cases
