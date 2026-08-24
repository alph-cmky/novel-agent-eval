import json
from pathlib import Path

from scripts.manage_eval import _paths, _write_json


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
