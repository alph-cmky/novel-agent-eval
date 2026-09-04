"""评测产物层：章节导出、人工盲测 schema、与 Judge 的 kappa 校准。

正文权威仍在 V2 SQLite；本模块只把 committed 草稿写成可读 md，
并把人工 8 维分与 Judge 对齐。不进入生产 graph。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from novel_agent_eval.judge import QUALITY_DIMS
from novel_agent_eval.longform import save_chapter_text

SCHEMA_VERSION = 1
N_BANDS = 5


def score_to_band(score: float) -> int:
    """把 0–100 分映射到 Judge rubric 的 5 档（0..4）。"""
    value = round(float(score))
    if value <= 20:
        return 0
    if value <= 40:
        return 1
    if value <= 60:
        return 2
    if value <= 80:
        return 3
    return 4


def quadratic_weighted_kappa(
    left: list[int],
    right: list[int],
    *,
    n_categories: int = N_BANDS,
) -> float | None:
    """Cohen's kappa with quadratic weights. Categories are 0..n_categories-1."""
    if len(left) != len(right) or len(left) < 2:
        return None
    size = n_categories
    observed = [[0] * size for _ in range(size)]
    for a, b in zip(left, right):
        if not (0 <= a < size and 0 <= b < size):
            return None
        observed[a][b] += 1
    n = len(left)
    hist_left = [sum(observed[i][j] for j in range(size)) for i in range(size)]
    hist_right = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    expected = [
        [hist_left[i] * hist_right[j] / n for j in range(size)] for i in range(size)
    ]
    denom = size - 1
    weights = [[((i - j) / denom) ** 2 for j in range(size)] for i in range(size)]
    num = sum(weights[i][j] * observed[i][j] for i in range(size) for j in range(size))
    den = sum(weights[i][j] * expected[i][j] for i in range(size) for j in range(size))
    if den == 0:
        return 1.0 if num == 0 else None
    return 1.0 - num / den


def empty_human_scores(
    *,
    run_tag: str,
    rater_id: str,
    chapter_numbers: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "rater_id": rater_id,
        "blind": True,
        "saw_judge": False,
        "scores": [
            {
                "chapter_number": n,
                "dimensions": {d: None for d in QUALITY_DIMS},
                "overall": None,
                "notes": "",
            }
            for n in chapter_numbers
        ],
    }


def load_human_scores(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("human_scores.json must be an object")
    scores = payload.get("scores")
    if not isinstance(scores, list):
        raise TypeError("human_scores.json missing scores[]")
    return payload


def save_human_scores(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_chapter_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def render_chapter_markdown(
    *,
    chapter_number: int,
    outline: str,
    content: str,
    previous: str = "",
) -> str:
    """Blind-readable chapter file: outline + optional previous, never Judge scores."""
    parts = [f"# 第{chapter_number}章", ""]
    if (outline or "").strip():
        parts.extend(["## 本章大纲", outline.strip(), ""])
    if (previous or "").strip():
        parts.extend(["## 前文摘要", previous.strip(), ""])
    parts.extend(["## 正文", (content or "").strip(), ""])
    return "\n".join(parts)


def export_chapter_markdown(
    path: Path,
    *,
    chapter_number: int,
    outline: str,
    content: str,
    previous: str = "",
) -> None:
    save_chapter_text(
        path,
        render_chapter_markdown(
            chapter_number=chapter_number,
            outline=outline,
            content=content,
            previous=previous,
        ),
    )


def judge_dims_from_baseline(result: dict[str, Any]) -> dict[int, dict[str, int]]:
    """chapter_number → 8 维 Judge 分（缺维或非数字的章跳过该维）。"""
    out: dict[int, dict[str, int]] = {}
    for row in result.get("chapters") or []:
        if not isinstance(row, dict):
            continue
        number = row.get("chapter") or row.get("chapter_number")
        dims = row.get("dimensions") or {}
        if not isinstance(number, int) or not isinstance(dims, dict):
            continue
        cleaned: dict[str, int] = {}
        for dim in QUALITY_DIMS:
            value = dims.get(dim)
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                cleaned[dim] = round(float(value))
        if cleaned:
            out[number] = cleaned
    return out


def calibrate_human_vs_judge(
    human: dict[str, Any],
    judge_by_chapter: dict[int, dict[str, int]],
) -> dict[str, Any]:
    """按 5 档 rubric 计算每维 quadratic weighted kappa。"""
    per_dim: dict[str, float | None] = {}
    paired_chapters: set[int] = set()
    for dim in QUALITY_DIMS:
        human_bands: list[int] = []
        judge_bands: list[int] = []
        for row in human.get("scores") or []:
            if not isinstance(row, dict):
                continue
            number = row.get("chapter_number")
            dims = row.get("dimensions") or {}
            human_val = dims.get(dim) if isinstance(dims, dict) else None
            judge_dims = judge_by_chapter.get(number) if isinstance(number, int) else None
            judge_val = judge_dims.get(dim) if judge_dims else None
            if human_val is None or judge_val is None:
                continue
            if not isinstance(human_val, (int, float)) or not isinstance(judge_val, (int, float)):
                continue
            human_bands.append(score_to_band(human_val))
            judge_bands.append(score_to_band(judge_val))
            paired_chapters.add(number)
        per_dim[dim] = quadratic_weighted_kappa(human_bands, judge_bands)
    present = [v for v in per_dim.values() if v is not None]
    return {
        "n_paired": len(paired_chapters),
        "blind": bool(human.get("blind", True)),
        "saw_judge": bool(human.get("saw_judge", False)),
        "rater_id": human.get("rater_id") or "",
        "dimensions": per_dim,
        "mean_kappa": (sum(present) / len(present)) if present else None,
    }


def _outline_map(run_dir: Path) -> dict[int, str]:
    prompts_path = run_dir / "prompts.json"
    if not prompts_path.exists():
        return {}
    try:
        payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = payload.get("chapter_outlines") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return out


def _project_id(persist_dir: Path) -> str | None:
    try:
        from novel_agent.storage.manager import ProjectManager

        projects = ProjectManager(persist_dir).list_projects()
    except Exception:  # noqa: BLE001 - 导出失败不应阻断评分落盘
        return None
    if not projects:
        return None
    return projects[0]["id"]


def export_run_chapters(run_dir: Path, *, persist_dir: Path | None = None) -> int:
    """从 V2 DB 导出 ``chapters/chXX.md``。返回写出的章数。"""
    run_dir = Path(run_dir)
    persist = Path(persist_dir) if persist_dir is not None else run_dir / "v2_state"
    result_path = run_dir / "baseline_result.json"
    if not result_path.exists():
        return 0
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    pid = _project_id(persist)
    if not pid:
        return 0
    try:
        from novel_agent.storage.manager import ProjectManager

        mgr = ProjectManager(persist)
    except Exception:  # noqa: BLE001
        return 0
    outlines = _outline_map(run_dir)
    chapter_dir = run_dir / "chapters"
    written = 0
    previous_tail = ""
    rows = sorted(
        (row for row in (result.get("chapters") or []) if isinstance(row, dict)),
        key=lambda row: int(row.get("chapter") or 0),
    )
    for row in rows:
        number = row.get("chapter")
        if not isinstance(number, int):
            continue
        try:
            record = mgr.get_chapter(pid, number)
        except Exception:  # noqa: BLE001
            record = None
        content = ((record or {}).get("draft_content") or "").strip()
        if not content:
            continue
        export_chapter_markdown(
            chapter_dir / f"ch{number:02d}.md",
            chapter_number=number,
            outline=outlines.get(number, ""),
            content=content,
            previous=previous_tail,
        )
        previous_tail = content[-400:]
        written += 1
    return written
