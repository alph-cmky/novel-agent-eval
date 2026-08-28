# novel_agent_eval/token_truth.py
"""Provider Token Truth — 统一 token 记账结构 + provider/model 归一化层。

背景（EVAL-1 实测）：step-3.7-flash 的 usage_metadata 存在严重 provider 端膨胀
（600 字章节 reported input ≈ 180-480× 本地估算）。本模块提供：

- UsageRaw：provider 原始四字段，**永不覆盖**
- TokenUsage：raw + normalized + local estimate + validity 的单一结构
- NormalizationRule 注册表：按 (provider, model) 字段级标记/变换，带 version；
  禁止全局固定除数
- Validity 三态：trusted / suspect / estimated
- aggregate：trusted-only 聚合与 raw 聚合严格分离，suspect 不污染 trusted

design 参考 joint plan Phase A（EVAL-P0）任务书。
"""
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

# ── Validity 三态 ────────────────────────────────────────

VALIDITY_TRUSTED = "trusted"      # provider 报告可信，直接进入正式 cost aggregate
VALIDITY_SUSPECT = "suspect"      # provider 报告已知异常（如 step-3.7-flash 膨胀）
VALIDITY_ESTIMATED = "estimated"  # 无 provider 报告，仅本地估算

_NORMALIZATION_VERSION = "token-truth-v1"


@dataclass(frozen=True)
class UsageRaw:
    """provider 原始 usage 四字段。raw 是真相存档，任何规则都不得改写它。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0   # 语义：cached ⊂ input（telemetry，不入 total）
    reasoning_tokens: int = 0      # 语义：reasoning ⊂ output（telemetry，不入 total）

    @property
    def total_tokens(self) -> int:
        """total = input + output；cached/reasoning 是子集，不重复计入。"""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class NormalizationRule:
    """单条 (provider, model) 级归一化规则。

    - field_validity: 逐字段 validity 标记（input/output 分开，允许不对称）
    - transform: 可选变换函数 (raw_value) -> normalized_value；None = 仅标记不变换。
      禁止在无实验证据时提供 transform —— 先 suspect，等 A-5 实验确定规律后再加。
    - version: 规则版本号；transform 结果必须能通过 version 追溯。
    - evidence: 结论来源（实验/文档引用）。
    """

    input_validity: str = VALIDITY_TRUSTED
    output_validity: str = VALIDITY_TRUSTED
    input_transform: Callable[[int], int] | None = None
    output_transform: Callable[[int], int] | None = None
    version: str = "identity-v1"
    evidence: str = ""


# ── 归一化规则注册表：(provider, model_prefix) → rule ──────
# model_prefix 匹配：exact 优先，然后最长前缀。新增规则必须附 evidence。
# 当前所有 step-3.7-flash 字段仅标记 suspect（EVAL-1 实测膨胀 ~180-480×，
# 规律未定，禁止拍脑袋 transform —— 待 A-5 实验输出后按 (provider, model, field) 补充）。

_NORMALIZATION_RULES: dict[tuple[str, str], NormalizationRule] = {
    ("stepfun", "step-3.7-flash"): NormalizationRule(
        input_validity=VALIDITY_SUSPECT,
        output_validity=VALIDITY_SUSPECT,
        version="step37flash-suspect-v2",
        evidence="A-5 probe 2026-08-28: identical messages+params report 922 tokens "
        "standalone vs 1,137,997 (1832x) inside LangGraph execution context; "
        "reported output exceeds max_tokens cap ~477x; isolated calls "
        "(direct client / LangChain / standalone agent) all ratio 0.95-1.13x. "
        "Provider-side metering defect triggered by graph-context requests; "
        "pattern non-deterministic (659x-2873x), no safe transform derivable yet.",
    ),
    ("stepfun", "step-"): NormalizationRule(
        input_validity=VALIDITY_SUSPECT,
        output_validity=VALIDITY_SUSPECT,
        version="stepfun-family-suspect-v1",
        evidence="family-level precaution: graph-context metering defect confirmed "
        "on step-3.7-flash; other step-* models unverified",
    ),
}


def resolve_rule(provider: str, model: str) -> NormalizationRule:
    """按 (provider, model) 查规则：exact 命中 > 最长前缀 > 默认 trusted。"""
    key = (provider, model)
    if key in _NORMALIZATION_RULES:
        return _NORMALIZATION_RULES[key]
    best: tuple[int, NormalizationRule] | None = None
    for (p, prefix), rule in _NORMALIZATION_RULES.items():
        if p == provider and model.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), rule)
    if best is not None:
        return best[1]
    return NormalizationRule()  # 无规则 → trusted / identity


@dataclass
class TokenUsage:
    """单次生成（一个 role / 一个 chapter）的统一 token 记账。

    - raw：provider 原始值，永不覆盖
    - normalized：按规则变换后的值（无 transform 时等于 raw）
    - estimate_input/output：本地字符估算（独立于 provider，用于 ratio 诊断）
    - validity：input/output 各自的 validity（取规则标记；无 provider 报告时 estimated）
    """

    provider: str
    model: str
    role: str
    raw: UsageRaw = field(default_factory=UsageRaw)
    normalized_input: int = 0
    normalized_output: int = 0
    normalized_cached: int = 0
    normalized_reasoning: int = 0
    input_validity: str = VALIDITY_TRUSTED
    output_validity: str = VALIDITY_TRUSTED
    estimate_input: int | None = None
    estimate_output: int | None = None
    normalization_version: str = "identity-v1"

    @property
    def validity(self) -> str:
        """整体 validity = input/output 中较差者（suspect > estimated 优先级：先 suspect）。"""
        if VALIDITY_SUSPECT in (self.input_validity, self.output_validity):
            return VALIDITY_SUSPECT
        if VALIDITY_ESTIMATED in (self.input_validity, self.output_validity):
            return VALIDITY_ESTIMATED
        return VALIDITY_TRUSTED

    @property
    def total_raw(self) -> int:
        return self.raw.total_tokens

    @property
    def total_normalized(self) -> int:
        return self.normalized_input + self.normalized_output

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw"] = asdict(self.raw)
        d["total_raw"] = self.total_raw
        d["total_normalized"] = self.total_normalized
        d["validity"] = self.validity
        return d


def build_token_usage(
    *,
    provider: str,
    model: str,
    role: str,
    raw: UsageRaw,
    estimate_input: int | None = None,
    estimate_output: int | None = None,
) -> TokenUsage:
    """从 raw usage + 归一化规则构造 TokenUsage。

    - 有 provider 报告：按规则逐字段标记 validity / transform；raw 原样保留
    - provider 报告全零：validity = estimated，normalized 用本地估算（若有）
    """
    if raw.input_tokens == 0 and raw.output_tokens == 0:
        return TokenUsage(
            provider=provider,
            model=model,
            role=role,
            raw=raw,
            normalized_input=estimate_input or 0,
            normalized_output=estimate_output or 0,
            input_validity=VALIDITY_ESTIMATED,
            output_validity=VALIDITY_ESTIMATED,
            estimate_input=estimate_input,
            estimate_output=estimate_output,
            normalization_version=_NORMALIZATION_VERSION,
        )

    rule = resolve_rule(provider, model)
    in_val = rule.input_validity
    out_val = rule.output_validity
    normalized_input = rule.input_transform(raw.input_tokens) if rule.input_transform else raw.input_tokens
    normalized_output = rule.output_transform(raw.output_tokens) if rule.output_transform else raw.output_tokens
    return TokenUsage(
        provider=provider,
        model=model,
        role=role,
        raw=raw,
        normalized_input=normalized_input,
        normalized_output=normalized_output,
        normalized_cached=raw.cached_input_tokens,
        normalized_reasoning=raw.reasoning_tokens,
        input_validity=in_val,
        output_validity=out_val,
        estimate_input=estimate_input,
        estimate_output=estimate_output,
        normalization_version=rule.version,
    )


# ── 本地估算（诊断用，非计费依据）──────────────────────────

def estimate_tokens(text: str) -> int:
    """粗估 token 数：CJK ≈ 0.75 token/char，非 CJK ≈ 1.3 token/word。

    用于 ratio 诊断与 provider 报告异常检测，禁止作为计费/正式指标。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    non_cjk = "".join(" " if "\u3400" <= ch <= "\u9fff" else ch for ch in text)
    words = len(non_cjk.split())
    return int(cjk * 0.75 + words * 1.3)


# ── provider 识别 + adapter 集成 ─────────────────────────

def detect_provider(base_url: str) -> str:
    """从 base_url 粗判 provider（规则注册表的 key 用）。"""
    url = (base_url or "").lower()
    if "stepfun" in url:
        return "stepfun"
    if "deepseek" in url:
        return "deepseek"
    if "openai" in url:
        return "openai"
    return "unknown"


def annotate_usage(
    usage: dict,
    *,
    provider: str,
    models: list[str],
) -> dict:
    """给 adapter 的扁平 usage dict 注入 validity / normalization 元数据（原地合并返回）。

    - validity 按 (provider, model) 规则解析；多 model 取较差者
    - raw 数值不被改写；normalized 无实验证据前等于 raw（仅标记）
    """
    rule = NormalizationRule()
    for m in models:
        r = resolve_rule(provider, m)
        worse = r if r.input_validity == VALIDITY_SUSPECT or r.output_validity == VALIDITY_SUSPECT else rule
        rule = worse
    usage["provider"] = provider
    usage["models"] = sorted(set(models))
    usage["input_validity"] = rule.input_validity
    usage["output_validity"] = rule.output_validity
    usage["validity"] = (
        VALIDITY_SUSPECT
        if VALIDITY_SUSPECT in (rule.input_validity, rule.output_validity)
        else VALIDITY_TRUSTED
    )
    usage["normalization_version"] = rule.version
    return usage


# ── 聚合：trusted 与 raw 严格分离 ─────────────────────────

def aggregate_token_usages(usages: list[TokenUsage]) -> dict:
    """聚合多条 TokenUsage。

    返回：
    - trusted: 只含 validity=trusted 的 normalized 计量（正式 cost 指标唯一来源）
    - suspect / estimated: 各自单独计数，绝不混入 trusted
    - raw: 全量原始值（telemetry 存档，永远保留）
    """
    out = {
        "trusted": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "count": 0},
        "suspect": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "count": 0},
        "estimated": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "count": 0},
        "raw": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0},
    }
    for u in usages:
        bucket = out[u.validity]
        bucket["input_tokens"] += u.normalized_input
        bucket["output_tokens"] += u.normalized_output
        bucket["total_tokens"] += u.total_normalized
        bucket["count"] += 1
        out["raw"]["input_tokens"] += u.raw.input_tokens
        out["raw"]["output_tokens"] += u.raw.output_tokens
        out["raw"]["cached_tokens"] += u.raw.cached_input_tokens
        out["raw"]["reasoning_tokens"] += u.raw.reasoning_tokens
        out["raw"]["total_tokens"] += u.raw.total_tokens
    return out
