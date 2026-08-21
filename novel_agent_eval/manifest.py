"""Reproducibility metadata for evaluation runs."""

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for file in files:
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        file_hash = _sha256(file)
        if file_hash:
            digest.update(file_hash.encode("ascii"))
    return digest.hexdigest()


def build_run_manifest(config: dict[str, Any], prompt_path: Path) -> dict[str, Any]:
    """Build run metadata without including credentials or full prompt text."""
    eval_root = Path(__file__).resolve().parents[1]
    novel_root = Path(
        os.environ.get("NOVEL_AGENT_PATH", str(eval_root.parent / "novel-agent"))
    )
    safe_config = {
        key: value
        for key, value in config.items()
        if "key" not in key.lower()
        and "token" not in key.lower()
        and "secret" not in key.lower()
    }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "eval_repo_commit": _git_head(eval_root),
        "eval_repo_dirty": _git_dirty(eval_root),
        "novel_agent_commit": _git_head(novel_root),
        "novel_agent_dirty": _git_dirty(novel_root),
        "dataset_hash": _tree_sha256(prompt_path.parent),
        "prompt_file": str(prompt_path),
        "prompt_hash": _sha256(prompt_path),
        "eval_lock_hash": _sha256(eval_root / "uv.lock"),
        "config": safe_config,
    }
