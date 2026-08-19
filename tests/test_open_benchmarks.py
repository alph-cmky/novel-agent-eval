import json

from novel_agent_eval.dataset import open_benchmarks


def test_load_constory_cases_filters_and_limits(monkeypatch, tmp_path):
    data = [
        {"id": 1, "language": "zh", "task_type": "generation", "prompt": "a"},
        {"id": 2, "language": "en", "task_type": "continuation", "prompt": "b"},
        {"id": 3, "language": "zh", "task_type": "expansion", "prompt": "c"},
    ]
    monkeypatch.setattr(open_benchmarks, "CONSTORY_PATH", _write_json(tmp_path, data))
    cases = open_benchmarks.load_constory_cases(language="zh", limit=1)
    assert [case.name for case in cases] == ["constory_1"]


def test_load_eqbench_prompts_is_numeric(monkeypatch, tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"2": {"title": "b"}, "1": {"title": "a"}}), encoding="utf-8")
    monkeypatch.setattr(open_benchmarks, "EQBENCH_PATH", path)
    prompts = open_benchmarks.load_eqbench_prompts()
    assert [item["prompt_id"] for item in prompts] == ["1", "2"]


def test_audit_marks_invalid_doc_re3(monkeypatch, tmp_path):
    constory = tmp_path / "constory.json"
    constory.write_text(json.dumps([]), encoding="utf-8")
    eqbench = tmp_path / "eqbench.json"
    eqbench.write_text(json.dumps({"1": {}}), encoding="utf-8")
    doc = tmp_path / "doc.json"
    doc.write_text("429: Too Many Requests", encoding="utf-8")
    monkeypatch.setattr(open_benchmarks, "CONSTORY_PATH", constory)
    monkeypatch.setattr(open_benchmarks, "EQBENCH_PATH", eqbench)
    monkeypatch.setattr(open_benchmarks, "DOC_RE3_PATH", doc)
    result = open_benchmarks.audit_open_assets()
    assert result["doc_re3"]["status"] == "invalid"
    assert result["doc_re3"]["eligible_for_scoring"] is False


def _write_json(directory, data):
    path = directory / "data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
