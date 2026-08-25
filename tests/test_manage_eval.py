import json
from pathlib import Path

from scripts.manage_eval import _PRESETS, _paths, _progress, _write_json


def test_status_paths_are_next_to_output(tmp_path):
    paths = _paths(tmp_path / "run.json")

    assert paths["status"] == tmp_path / "run.status.json"
    assert paths["log"] == tmp_path / "run.log"
    assert paths["pid"] == tmp_path / "run.pid"


def test_write_json_is_atomic_and_utf8(tmp_path):
    path = tmp_path / "nested" / "status.json"

    _write_json(path, {"status": "running", "title": "长篇"})

    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "长篇"
    assert not Path(str(path) + ".tmp").exists()


def test_progress_reports_results_failures_and_saved_chapters(tmp_path):
    output = tmp_path / "run.json"
    output.with_name("run.partial_results.json").write_text(
        '{"results":[{},{}]}', encoding="utf-8"
    )
    output.with_name("run.failures.json").write_text(
        '{"failures":[{}]}', encoding="utf-8"
    )
    chapters = output.with_name("run.chapters") / "agent" / "1" / "0"
    chapters.mkdir(parents=True)
    (chapters / "chapter_01.txt").write_text("text", encoding="utf-8")

    assert _progress(output, {"agents": "a", "prompts": 2, "repeat": 2}) == {
        "completed_samples": 2,
        "failures": 1,
        "chapters_saved": 1,
        "target_samples": 4,
        "sample_percent": 50.0,
    }


def test_presets_keep_longform_protocols_explicit():
    assert _PRESETS["prompt_v2_20"] == {
        "prompts": 2,
        "repeat": 1,
        "chapters": 20,
        "agents": "novel_agent",
        "max_rounds": 0,
        "prompt_profile": "v2",
    }
