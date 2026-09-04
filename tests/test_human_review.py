# tests/test_human_review.py
"""章节导出、人工分 schema、与 Judge 的 5 档 quadratic weighted kappa。"""

from novel_agent_eval.human_review import (
    calibrate_human_vs_judge,
    empty_human_scores,
    export_chapter_markdown,
    judge_dims_from_baseline,
    load_human_scores,
    quadratic_weighted_kappa,
    render_chapter_markdown,
    save_human_scores,
    score_to_band,
)
from novel_agent_eval.judge import QUALITY_DIMS
from novel_agent_eval.report import render_scorecard
from novel_agent_eval.runner import BenchmarkReport, BenchmarkResult, CaseRun


def test_score_to_band_matches_rubric():
    assert score_to_band(0) == 0
    assert score_to_band(20) == 0
    assert score_to_band(21) == 1
    assert score_to_band(80) == 3
    assert score_to_band(81) == 4
    assert score_to_band(100) == 4


def test_quadratic_weighted_kappa_perfect_and_swap():
    assert quadratic_weighted_kappa([3, 3, 2, 4], [3, 3, 2, 4]) == 1.0
    # 两端完全对调、边际相同 → 二次加权 kappa = -1
    assert quadratic_weighted_kappa([0, 4], [4, 0]) == -1.0


def test_quadratic_weighted_kappa_too_short():
    assert quadratic_weighted_kappa([1], [1]) is None


def test_chapter_markdown_omits_judge_scores():
    md = render_chapter_markdown(
        chapter_number=2,
        outline="取星髓铁",
        content="沈舟入涧。",
        previous="上一章结尾",
    )
    assert md.startswith("# 第2章")
    assert "取星髓铁" in md
    assert "沈舟入涧" in md
    assert "上一章结尾" in md
    assert "Judge" not in md
    assert "overall" not in md


def test_export_and_human_scores_roundtrip(tmp_path):
    path = tmp_path / "chapters" / "ch01.md"
    export_chapter_markdown(
        path, chapter_number=1, outline="开篇", content="正文一" * 10
    )
    text = path.read_text(encoding="utf-8")
    assert "正文一" in text

    scores = empty_human_scores(run_tag="demo", rater_id="r1", chapter_numbers=[1, 2])
    scores["scores"][0]["dimensions"] = {d: 70 for d in QUALITY_DIMS}
    scores["scores"][0]["overall"] = 70
    out = tmp_path / "annotations" / "human_scores.json"
    save_human_scores(out, scores)
    loaded = load_human_scores(out)
    assert loaded["rater_id"] == "r1"
    assert loaded["scores"][0]["dimensions"]["plot"] == 70


def test_calibrate_human_vs_judge_sample():
    dims = {d: 78 for d in QUALITY_DIMS}
    human = {
        "blind": True,
        "saw_judge": False,
        "rater_id": "r1",
        "scores": [
            {"chapter_number": 1, "dimensions": {d: 76 for d in QUALITY_DIMS}},
            {"chapter_number": 2, "dimensions": {d: 82 for d in QUALITY_DIMS}},
        ],
    }
    judge = {1: dict(dims), 2: {d: 84 for d in QUALITY_DIMS}}
    report = calibrate_human_vs_judge(human, judge)
    assert report["n_paired"] == 2
    assert report["mean_kappa"] == 1.0
    assert all(v == 1.0 for v in report["dimensions"].values())


def test_judge_dims_from_baseline_skips_nulls():
    result = {
        "chapters": [
            {"chapter": 1, "dimensions": {d: 80 for d in QUALITY_DIMS}},
            {"chapter": 2, "dimensions": {d: None for d in QUALITY_DIMS}},
        ]
    }
    mapping = judge_dims_from_baseline(result)
    assert 1 in mapping
    assert 2 not in mapping


def test_scorecard_renders_kappa_table():
    run = CaseRun(
        run_index=0,
        dimensions={d: 80 for d in QUALITY_DIMS},
        overall=80.0,
        meta={},
    )
    res = BenchmarkResult(
        agent="novel_agent",
        case="c1",
        stage="opening",
        repeat=1,
        dims_mean={d: 80.0 for d in QUALITY_DIMS},
        dims_std={d: 0.0 for d in QUALITY_DIMS},
        overall_mean=80.0,
        overall_std=0.0,
        runs=[run],
    )
    report = BenchmarkReport(
        results=[res], repeat=1, agents=["novel_agent"], cases=["c1"]
    )
    md = render_scorecard(
        report,
        calibration={
            "n_paired": 2,
            "blind": True,
            "saw_judge": False,
            "rater_id": "r1",
            "mean_kappa": 0.5,
            "dimensions": {d: 0.5 for d in QUALITY_DIMS},
        },
    )
    assert "配对章数 **2**" in md
    assert "0.500" in md
    assert "待采集（需人工盲测 ground truth）" not in md
