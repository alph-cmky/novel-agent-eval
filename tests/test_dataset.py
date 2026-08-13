# tests/test_dataset.py
from novel_agent_eval.dataset.loader import load_cases

def test_load_nine_cases():
    cases = load_cases("novel_agent_eval/dataset/self_built")
    assert len(cases) == 9
    stages = [c.stage for c in cases]
    assert stages.count("opening") == 3
    assert stages.count("middle") == 3
    assert stages.count("long") == 3
