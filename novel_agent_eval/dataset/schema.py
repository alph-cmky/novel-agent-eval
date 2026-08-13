# novel_agent_eval/dataset/schema.py
from typing import Literal
from pydantic import BaseModel, Field

Stage = Literal["opening", "middle", "long"]   # 开局(1-10章) / 中段(30-50) / 长程(80-100)

class GroundTruth(BaseModel):
    continuity_bugs: list[dict] = Field(default_factory=list)  # 注入的一致性错误（对齐 continuity 3 类）
    foreshadowings: list[str] = Field(default_factory=list)    # 应回收的伏笔
    outline_points: list[str] = Field(default_factory=list)    # 本章必须覆盖的大纲点

class EvalCase(BaseModel):
    name: str
    stage: Stage
    genre: str = "玄幻"
    story_outline: str                 # 全书大纲（长程 case 需跨章节世界观）
    previous_context: str              # 前文章节上下文（供连贯性检查）
    target_chapter_outline: str        # 本章大纲
    word_target: int = 3000
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)
    narrative_mode: str | None = None  # 激活验证用，API 直传
    narrative_perspective: str | None = None
