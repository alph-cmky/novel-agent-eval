# tests/test_token_truth.py
"""Token Truth 单元测试（纯逻辑，不耗 API）。

覆盖 EVAL-P0 验收点：
- input+output 数学正确；cached/reasoning 不重复计入
- 每个 request 只累计一次（聚合不重复加总）
- suspect usage 不污染 trusted aggregate
- raw 永不被归一化覆盖；规则按 provider+model 解析且带 version
"""
from novel_agent_eval.token_truth import (
    VALIDITY_ESTIMATED,
    VALIDITY_SUSPECT,
    UsageRaw,
    aggregate_token_usages,
    build_token_usage,
    estimate_tokens,
    resolve_rule,
)


def test_usage_raw_total_is_input_plus_output_only():
    raw = UsageRaw(input_tokens=100, output_tokens=50, cached_input_tokens=30, reasoning_tokens=20)
    assert raw.total_tokens == 150  # cached/reasoning 是子集，不入 total


def test_build_trusted_when_no_rule_matches():
    u = build_token_usage(
        provider="deepseek", model="deepseek-chat", role="writer",
        raw=UsageRaw(input_tokens=1000, output_tokens=500, cached_input_tokens=100, reasoning_tokens=200),
    )
    assert u.validity == "trusted"
    assert u.normalized_input == 1000  # identity：normalized == raw
    assert u.total_normalized == 1500
    assert u.total_raw == 1500
    assert u.normalization_version == "identity-v1"


def test_step37flash_marked_suspect_and_raw_preserved():
    u = build_token_usage(
        provider="stepfun", model="step-3.7-flash", role="writer",
        raw=UsageRaw(input_tokens=2_180_000, output_tokens=842_000),
    )
    assert u.input_validity == VALIDITY_SUSPECT
    assert u.output_validity == VALIDITY_SUSPECT
    assert u.validity == VALIDITY_SUSPECT
    # 无实验证据前不 transform：normalized == raw，但被标记 suspect
    assert u.normalized_input == 2_180_000
    # raw 永不覆盖
    assert u.raw.input_tokens == 2_180_000
    assert u.normalization_version == "step37flash-suspect-v1"


def test_rule_prefix_matching_longest_wins():
    rule = resolve_rule("stepfun", "step-3.7-flash")
    assert rule.version == "step37flash-suspect-v1"  # exact 优先于 family 前缀
    family = resolve_rule("stepfun", "step-9.9-mini")
    assert family.version == "stepfun-family-suspect-v1"
    other = resolve_rule("deepseek", "deepseek-v4-pro")
    assert other.input_validity == "trusted"  # 无规则 → 默认 trusted


def test_zero_usage_falls_back_to_estimated():
    u = build_token_usage(
        provider="stepfun", model="step-3.7-flash", role="editor",
        raw=UsageRaw(),
        estimate_input=500,
        estimate_output=300,
    )
    assert u.validity == VALIDITY_ESTIMATED
    assert u.normalized_input == 500
    assert u.total_normalized == 800


def test_suspect_does_not_pollute_trusted_aggregate():
    trusted = build_token_usage(
        provider="deepseek", model="deepseek-chat", role="writer",
        raw=UsageRaw(input_tokens=1000, output_tokens=500),
    )
    suspect = build_token_usage(
        provider="stepfun", model="step-3.7-flash", role="orchestrator",
        raw=UsageRaw(input_tokens=10_000_000, output_tokens=5_000_000),
    )
    agg = aggregate_token_usages([trusted, suspect])

    assert agg["trusted"]["total_tokens"] == 1500  # 只含 trusted
    assert agg["trusted"]["count"] == 1
    assert agg["suspect"]["total_tokens"] == 15_000_000  # suspect 单独可见
    assert agg["suspect"]["count"] == 1
    # raw 全量保留（trusted + suspect）
    assert agg["raw"]["input_tokens"] == 1_000 + 10_000_000
    assert agg["raw"]["total_tokens"] == 1500 + 15_000_000


def test_aggregate_counts_each_usage_once():
    usages = [
        build_token_usage(provider="p", model="m", role=f"r{i}", raw=UsageRaw(input_tokens=10, output_tokens=5))
        for i in range(4)
    ]
    agg = aggregate_token_usages(usages)
    assert agg["trusted"]["count"] == 4
    assert agg["trusted"]["input_tokens"] == 40  # 4 × 10，不是 8 × 10
    assert agg["raw"]["total_tokens"] == 60


def test_estimate_tokens_cjk_vs_ascii():
    cjk = estimate_tokens("风起云涌" * 100)  # 400 CJK chars
    assert 280 <= cjk <= 320  # ~0.75/char
    ascii_text = "word " * 100  # 500 ascii chars ≈ 100 words
    assert 120 <= estimate_tokens(ascii_text) <= 140
    assert estimate_tokens("") == 0


def test_to_dict_roundtrip_fields():
    u = build_token_usage(
        provider="stepfun", model="step-3.7-flash", role="writer",
        raw=UsageRaw(input_tokens=100, output_tokens=50, cached_input_tokens=10, reasoning_tokens=5),
        estimate_input=90,
    )
    d = u.to_dict()
    assert d["raw"]["input_tokens"] == 100
    assert d["validity"] == VALIDITY_SUSPECT
    assert d["total_raw"] == 150
    assert d["total_normalized"] == 150
    assert d["estimate_input"] == 90
