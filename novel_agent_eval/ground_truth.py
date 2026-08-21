"""Deterministic evidence metrics for self-built evaluation cases."""

import re
from collections.abc import Iterable
from typing import Any


def _terms(value: str) -> list[str]:
    chunks = re.findall(r"[\u3400-\u9fff]{2,}|[A-Za-z0-9][A-Za-z0-9_-]+", value)
    return list(dict.fromkeys(chunks))


def _matches(text: str, item: str) -> bool:
    terms = _terms(item)
    if not terms:
        return False
    normalized = text.casefold()
    hits = 0
    total = 0
    for term in terms:
        if term.casefold() in normalized:
            hits += 1
            total += 1
            continue
        grams = [term[i:i + 2] for i in range(len(term) - 1)]
        total += len(grams)
        hits += sum(gram.casefold() in normalized for gram in grams)
    # Require multiple pieces for long descriptions, while allowing concise
    # ground-truth labels such as "火莲印" to count as evidence.
    required = 1 if total <= 2 else max(2, (total + 2) // 3)
    return hits >= required


def _coverage(items: Iterable[str], text: str) -> dict[str, Any]:
    values = [str(item) for item in items if str(item).strip()]
    matched = [item for item in values if _matches(text, item)]
    return {
        "matched": len(matched),
        "total": len(values),
        "rate": round(len(matched) / len(values), 3) if values else None,
        "matched_items": matched,
    }


def ground_truth_metrics(text: str, ground_truth) -> dict[str, Any]:
    """Return interpretable coverage and contradiction-exposure evidence.

    ``continuity_bug_exposure`` is intentionally not a quality score: it is
    the fraction of injected bug keyword sets reproduced by the output. Lower
    is better. Outline and foreshadowing rates measure evidence presence.
    """
    outline = getattr(ground_truth, "outline_points", None)
    foreshadowings = getattr(ground_truth, "foreshadowings", None)
    bugs = getattr(ground_truth, "continuity_bugs", None)
    if isinstance(ground_truth, dict):
        outline = ground_truth.get("outline_points", [])
        foreshadowings = ground_truth.get("foreshadowings", [])
        bugs = ground_truth.get("continuity_bugs", [])
    outline = outline or []
    foreshadowings = foreshadowings or []
    bugs = bugs or []

    exposed = []
    for bug in bugs:
        if not isinstance(bug, dict):
            continue
        keywords = [str(k) for k in bug.get("keywords", []) if str(k).strip()]
        if keywords and all(k.casefold() in text.casefold() for k in keywords):
            exposed.append({
                "category": bug.get("category", "unknown"),
                "severity": bug.get("severity", "unknown"),
                "keywords": keywords,
            })

    return {
        "available": bool(outline or foreshadowings or bugs),
        "outline_coverage": _coverage(outline, text),
        "foreshadowing_coverage": _coverage(foreshadowings, text),
        "continuity_bug_exposure": {
            "exposed": len(exposed),
            "total": len(bugs),
            "rate": round(len(exposed) / len(bugs), 3) if bugs else None,
            "items": exposed,
        },
    }
