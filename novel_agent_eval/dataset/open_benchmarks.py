"""Load and audit the open benchmark assets used by the evaluation runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from novel_agent_eval.dataset.schema import EvalCase

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTORY_PATH = REPO_ROOT / "novel_agent_eval" / "dataset" / "external" / "constory" / "constory_prompts_longform.json"
EQBENCH_PATH = REPO_ROOT / "novel_agent_eval" / "dataset" / "eqbench" / "prompts.json"
DOC_RE3_PATH = REPO_ROOT / "novel_agent_eval" / "dataset" / "external" / "doc_re3" / "prompts.json"


def _read_json(path: Path) -> Any:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, str):
        raise TypeError(f"{path} contains a JSON string, not benchmark records")
    return data


def load_constory_cases(
    *,
    limit: int | None = None,
    language: str | None = None,
    task_types: tuple[str, ...] = ("generation", "continuation"),
) -> list[EvalCase]:
    """Load a deterministic slice of ConStory prompts as single-chapter cases."""
    raw = _read_json(CONSTORY_PATH)
    if not isinstance(raw, list):
        raise TypeError(f"ConStory asset must be a list: {CONSTORY_PATH}")

    selected = [
        item
        for item in raw
        if isinstance(item, dict)
        and (language is None or item.get("language") == language)
        and item.get("task_type") in task_types
    ]
    if limit is not None:
        selected = selected[:limit]

    return [
        EvalCase(
            name=f"constory_{item.get('id', index + 1)}",
            stage="long",
            genre="general",
            story_outline=str(item.get("prompt", "")),
            previous_context="",
            target_chapter_outline=str(item.get("prompt", "")),
            word_target=2000,
        )
        for index, item in enumerate(selected)
    ]


def load_eqbench_prompts(limit: int | None = None) -> list[dict[str, str]]:
    """Load EQ-Bench Longform prompts in numeric order."""
    raw = _read_json(EQBENCH_PATH)
    if not isinstance(raw, dict):
        raise TypeError(f"EQ-Bench asset must be an object: {EQBENCH_PATH}")
    prompts = []
    for key in sorted(raw, key=int):
        item = raw[key]
        if not isinstance(item, dict):
            continue
        prompts.append(
            {
                "prompt_id": str(key),
                "title": str(item.get("title", "")),
                "category": str(item.get("category", "")),
                "writing_prompt": str(item.get("writing_prompt", "")),
            }
        )
    return prompts if limit is None else prompts[:limit]


def audit_open_assets() -> dict[str, Any]:
    """Return an offline manifest and reject known invalid downloaded assets."""
    constory = _read_json(CONSTORY_PATH)
    eqbench = _read_json(EQBENCH_PATH)
    doc_re3_status = "missing"
    if DOC_RE3_PATH.exists():
        try:
            doc_re3 = _read_json(DOC_RE3_PATH)
            doc_re3_status = "valid" if isinstance(doc_re3, (list, dict)) else "invalid"
        except (OSError, json.JSONDecodeError, ValueError):
            doc_re3_status = "invalid"

    constory_records = constory if isinstance(constory, list) else []
    task_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for item in constory_records:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task_type", "unknown"))
        language = str(item.get("language", "unknown"))
        task_counts[task] = task_counts.get(task, 0) + 1
        language_counts[language] = language_counts.get(language, 0) + 1

    return {
        "constory": {
            "records": len(constory_records),
            "task_counts": task_counts,
            "language_counts": language_counts,
        },
        "eqbench_longform": {
            "prompts": len(eqbench) if isinstance(eqbench, dict) else 0,
            "official_runner": "scripts/run_eqbench_longform.py",
        },
        "doc_re3": {
            "status": doc_re3_status,
            "eligible_for_scoring": doc_re3_status == "valid",
        },
    }
