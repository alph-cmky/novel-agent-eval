# novel_agent_eval/dataset/loader.py
from pathlib import Path
from .schema import EvalCase

def load_cases(dir_path: str) -> list[EvalCase]:
    cases = []
    for p in sorted(Path(dir_path).glob("*.json")):
        cases.append(EvalCase.model_validate_json(p.read_text(encoding="utf-8")))
    return cases
