# tests/test_evaluation_config.py
"""Phase 0.3 EvaluationConfig 契约测试。

验证实验变量显式化、manifest 记录完整配置、凭证不落盘。
"""

import pytest

from novel_agent_eval.evaluation_config import EvaluationConfig
from novel_agent_eval.manifest import build_eval_manifest


def test_defaults_match_current_behavior():
    cfg = EvaluationConfig(model="step-3.7-flash", judge_model="step-3.7-flash", chapter_count=20)
    assert cfg.prompt_version == "v1"
    assert cfg.repeat == 1
    assert cfg.max_rounds == 2
    # B-1 parity：机制开关默认与 Production 入口一致（routes.py:636-648）
    assert cfg.deterministic_gate_first is False
    assert cfg.memory_protocol == "structured_narrative_state"
    assert cfg.scene_first is False
    assert cfg.synthetic_context is False
    assert cfg.context_mode == "bounded_memory"
    # C-1 叙事参数默认
    assert cfg.narrative_mode is None
    assert cfg.target_chapter_words == 3000
    assert cfg.review_interval == 1
    assert cfg.story_length == "long"
    assert cfg.timeout is None
    assert cfg.resume is False
    assert cfg.seed is None
    assert cfg.temperature is None


def test_to_manifest_config_contains_all_fields():
    cfg = EvaluationConfig(
        model="m", judge_model="j", chapter_count=30, repeat=3, max_rounds=1,
        deterministic_gate_first=False, scene_first=False,
        memory_protocol="previous_context_only", context_mode="full_context",
        timeout=600.0, resume=True, seed=42, temperature=0.7,
    )
    d = cfg.to_manifest_config()
    for key in (
        "model", "judge_model", "prompt_version", "chapter_count", "repeat",
        "max_rounds", "deterministic_gate_first", "memory_protocol",
        "scene_first", "context_mode", "timeout", "resume", "seed", "temperature",
    ):
        assert key in d, f"{key} 应写入 manifest"
    assert d["max_rounds"] == 1
    assert d["scene_first"] is False


def test_build_eval_manifest_records_config_without_credentials(tmp_path):
    prompt = tmp_path / "prompts.json"
    prompt.write_text('{"1": {"title": "T"}}', encoding="utf-8")
    cfg = EvaluationConfig(model="step-3.7-flash", judge_model="step-3.7-flash", chapter_count=20)

    manifest = build_eval_manifest(cfg, prompt)

    # 完整配置写入
    assert manifest["config"]["model"] == "step-3.7-flash"
    assert manifest["config"]["chapter_count"] == 20
    assert manifest["config"]["max_rounds"] == 2
    assert manifest["config"]["scene_first"] is False
    # 复现性锚点
    assert manifest["prompt_hash"]
    assert manifest["dataset_hash"]
    assert manifest["eval_lock_hash"]
    # 凭证绝不落盘
    assert "STEPFUN_API_KEY" not in str(manifest)
    assert "api_key" not in manifest["config"]


def test_build_eval_manifest_rejects_implicit_env_only_config(tmp_path):
    """不允许用隐式环境变量替代实验协议：必须给 EvaluationConfig 或 dict。"""
    prompt = tmp_path / "prompts.json"
    prompt.write_text("{}", encoding="utf-8")

    with pytest.raises(TypeError):
        build_eval_manifest("not-a-config", prompt)  # type: ignore[arg-type]
