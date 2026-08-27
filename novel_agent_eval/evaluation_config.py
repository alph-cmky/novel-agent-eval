# novel_agent_eval/evaluation_config.py
"""冻结的评测协议配置 — 把实验变量显式化，不依赖隐式环境变量。

Phase 0.3 契约：任何正式实验（baseline / 消融 / 主实验 / 横评）都必须构造一份
EvaluationConfig 并写入 manifest，保证 run 可复现、变量可审计。环境变量只用于
凭证（STEPFUN_API_KEY / OPENAI_API_KEY），绝不作为实验变量的隐式来源。

字段对齐执行方案 §0.3：
  - model / judge_model / prompt_version：生成与判分模型 + prompt 版本
  - chapter_count / repeat / max_rounds：实验规模与进化预算
  - deterministic_gate_first / memory_protocol / scene_first / context_mode：
    单变量消融的四个机制开关（Phase 3/4）
  - timeout / resume / seed / temperature：运行控制
"""

from typing import Literal

from pydantic import BaseModel, Field

MemoryProtocol = Literal["structured_narrative_state", "previous_context_only"]
ContextMode = Literal["bridge", "full_context", "bounded_memory"]


class EvaluationConfig(BaseModel):
    """一次正式评测的完整协议配置。"""

    # ── 模型与判分 ──
    model: str = Field(description="被测生成模型（novel-agent 走 ModelRouter，vanilla 走 BASELINE_*）")
    judge_model: str = Field(description="LLM-judge 模型（StepFun）")
    prompt_version: str = Field(default="v1", description="prompt 版本标签，变更 prompt 必须改版本号")

    # ── 实验规模 ──
    chapter_count: int = Field(ge=1, description="单条 prompt 连载章数")
    repeat: int = Field(default=1, ge=1, description="独立采样重复次数")
    max_rounds: int = Field(default=2, ge=0, description="单章自动进化预算；0 = 无进化对照")

    # ── 机制开关（单变量消融对象）──
    deterministic_gate_first: bool = Field(default=True, description="确定性硬门禁先行，过则跳过昂贵 LLM 审查")
    memory_protocol: MemoryProtocol = Field(
        default="structured_narrative_state",
        description="结构化叙事状态 vs 仅前文上下文（Phase 4.2 核心消融）",
    )
    scene_first: bool = Field(default=True, description="scene-first 拆场生成 vs 整章生成")
    context_mode: ContextMode = Field(
        default="bounded_memory",
        description="上下文协议：bridge / full_context / bounded_memory",
    )

    # ── 运行控制 ──
    timeout: float | None = Field(default=None, description="单章超时秒数；None = 不限时")
    resume: bool = Field(default=False, description="是否断点续跑（跳过已完成章节）")
    seed: int | None = Field(default=None, description="随机种子；None = 不固定")
    temperature: float | None = Field(default=None, ge=0, le=2, description="采样温度；None = 各 adapter 默认")

    def to_manifest_config(self) -> dict:
        """转成可写 manifest 的 dict（不含凭证；凭证只经 env 传）。

        manifest.build_run_manifest 会再过滤含 key/token/secret 的键，这里
        本就不放任何凭证字段，双重保险。
        """
        return self.model_dump()
