# tests/test_dataset.py
from pathlib import Path

from novel_agent_eval.dataset.loader import load_cases

_SELF_BUILT_DIR = Path(__file__).resolve().parents[1] / "novel_agent_eval" / "dataset" / "self_built"

def test_load_all_self_built_cases():
    cases = load_cases(str(_SELF_BUILT_DIR))
    assert len(cases) == 12
    stages = [c.stage for c in cases]
    assert stages.count("opening") == 4
    assert stages.count("middle") == 4
    assert stages.count("long") == 4
