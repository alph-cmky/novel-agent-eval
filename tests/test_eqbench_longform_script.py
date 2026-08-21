"""长篇入口的增量落盘测试。"""
import importlib.util
import json
from pathlib import Path


def _load_runner(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("STEPFUN_API_KEY", "test")
    path = Path(__file__).parents[1] / "scripts" / "run_eqbench_longform.py"
    spec = importlib.util.spec_from_file_location("run_eqbench_longform", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_json_atomic_replaces_existing_file(tmp_path, monkeypatch):
    runner = _load_runner(monkeypatch)
    output = tmp_path / "nested" / "partial_results.json"

    runner._write_json_atomic(output, {"version": 1})
    runner._write_json_atomic(output, {"version": 2, "results": []})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "version": 2,
        "results": [],
    }
    assert not output.with_name(f".{output.name}.tmp").exists()
