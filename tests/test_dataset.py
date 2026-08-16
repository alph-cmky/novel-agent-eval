# tests/test_dataset.py
import json
import tempfile
from pathlib import Path

from novel_agent_eval.dataset.loader import load_cases, load_external_constory_cases

_SELF_BUILT_DIR = Path(__file__).resolve().parents[1] / "novel_agent_eval" / "dataset" / "self_built"


def test_load_all_self_built_cases():
    cases = load_cases(str(_SELF_BUILT_DIR))
    assert len(cases) == 13
    stages = [c.stage for c in cases]
    assert stages.count("opening") == 4
    assert stages.count("middle") == 5
    assert stages.count("long") == 4


def test_self_built_cases_schema_integrity():
    """验证所有自建用例字段完整性与有效性。"""
    cases = load_cases(str(_SELF_BUILT_DIR))
    for c in cases:
        assert c.name, "用例名称不可为空"
        assert c.stage in ("opening", "middle", "long")
        assert c.genre, f"{c.name} 缺少题材标签"
        assert len(c.target_chapter_outline) >= 20, f"{c.name} 本章大纲过短"
        assert c.word_target >= 1000, f"{c.name} 目标字数过低"

        # 中段和长程必须有前文上下文或全书大纲
        if c.stage in ("middle", "long"):
            assert len(c.story_outline) > 0 or len(c.previous_context) > 0, f"{c.name} 缺少历史上下文"


def test_self_built_cases_ground_truth_quality():
    """验证自建评测集的 Ground Truth 结构质量（防吃书注入、伏笔与大纲点）。"""
    cases = load_cases(str(_SELF_BUILT_DIR))
    valid_categories = {"character", "timeline", "worldbuilding"}
    valid_severities = {"critical", "major", "minor"}

    total_bugs = 0
    total_foreshadowings = 0
    for c in cases:
        gt = c.ground_truth
        # 验证大纲要点
        assert len(gt.outline_points) >= 2, f"{c.name} 大纲要点不足 2 条"

        # 验证注入的 continuity bug 规范
        for bug in gt.continuity_bugs:
            total_bugs += 1
            assert bug.get("category") in valid_categories, f"{c.name} bug category 异常: {bug}"
            assert bug.get("severity") in valid_severities, f"{c.name} bug severity 异常: {bug}"
            assert "description" in bug and len(bug["description"]) > 5
            assert "keywords" in bug and isinstance(bug["keywords"], list)

        # 统计伏笔数
        total_foreshadowings += len(gt.foreshadowings)

    # 确保评测集拥有充足的防吃书陷阱与伏笔生命周期追踪点
    assert total_bugs >= 20, f"总注入 bug 数过少: {total_bugs}"
    assert total_foreshadowings >= 15, f"总伏笔数过少: {total_foreshadowings}"


def test_self_built_genres_diversity():
    """验证评测集涵盖多样化题材（包含古言百合、仙侠耽美、赛博朋克、女主无CP修仙等）。"""
    cases = load_cases(str(_SELF_BUILT_DIR))
    genres = {c.genre for c in cases}
    assert "古言百合" in genres
    assert "仙侠耽美" in genres
    assert "赛博耽美" in genres
    assert "女主无CP修仙" in genres
    assert "玄幻" in genres
    assert "都市" in genres


def test_load_external_constory_cases():
    """测试外部 ConStory 数据集的加载与转换。"""
    sample_data = [
        {"id": "cs_001", "prompt": "一个关于时空穿梭者的长篇故事大纲"},
        {"id": "cs_002", "prompt": "一个关于深海探险队的长篇故事大纲"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps(sample_data, ensure_ascii=False))
        tmp_path = f.name

    try:
        cases = load_external_constory_cases(tmp_path, max_cases=10)
        assert len(cases) == 2
        assert cases[0].name == "constory_cs_001"
        assert cases[0].stage == "long"
        assert "时空穿梭者" in cases[0].target_chapter_outline
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_external_constory_cases_missing_file():
    """测试文件不存在时安全降级返回空列表。"""
    cases = load_external_constory_cases("/path/to/non_existent_file.json")
    assert cases == []
