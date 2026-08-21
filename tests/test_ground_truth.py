from novel_agent_eval.ground_truth import ground_truth_metrics


def test_ground_truth_reports_evidence_and_bug_exposure():
    metrics = ground_truth_metrics(
        "秦照夜把火种托付给洛千秋。火莲印焚出血路。北墙由黑曜石砌成，青石开裂。",
        {
            "outline_points": ["秦照夜托付火种", "洛千秋以火莲印突围"],
            "foreshadowings": ["火种", "火莲印"],
            "continuity_bugs": [
                {"category": "worldbuilding", "severity": "critical", "keywords": ["黑曜石", "青石"]}
            ],
        },
    )

    assert metrics["available"] is True
    assert metrics["outline_coverage"]["rate"] == 1.0
    assert metrics["foreshadowing_coverage"]["rate"] == 1.0
    assert metrics["continuity_bug_exposure"]["rate"] == 1.0


def test_ground_truth_without_annotations_is_unavailable():
    metrics = ground_truth_metrics("正文", {})
    assert metrics["available"] is False
    assert metrics["outline_coverage"]["rate"] is None
