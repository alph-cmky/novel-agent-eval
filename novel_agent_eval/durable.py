# novel_agent_eval/durable.py
"""Phase 1 durable longform 契约：逐章 checkpoint + 统一失败语义。

V2 durable state（Project/ChapterVersion/Canon）是跨进程恢复的真相源；
本模块提供 eval 侧的逐章 checkpoint（补全进程内消亡的 token/hash/score）和
多章 run 的统一失败分类，供 Rule Gate / 质量统计 / 成功率计算一致使用。

失败语义（§1.3）：
- completed：全部 expected 章节成功
- partial：部分成功后失败
- failed：无一成功且非 timeout/invalid
- timeout：超时为主因
- invalid：空输出 / 解析失败 / censorship 等无效产出为主因
"""
import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# 单章状态（比 run 级 outcome 更细，记录每章如何结束）
CHAPTER_COMPLETED = "completed"
CHAPTER_RESUMED = "resumed"  # 跨进程恢复命中，未重生
CHAPTER_PARTIAL = "partial"  # 生成了一部分但未 commit
CHAPTER_FAILED = "failed"
CHAPTER_TIMEOUT = "timeout"
CHAPTER_INVALID = "invalid"  # 空输出 / 解析失败 / censorship

# run 级 outcome
OUTCOME_COMPLETED = "completed"
OUTCOME_PARTIAL = "partial"
OUTCOME_FAILED = "failed"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_INVALID = "invalid"


@dataclass
class ChapterCheckpoint:
    """单章 durable checkpoint（§1.2 富字段）。

    失败不能导致之前章节数据丢失：save 每章原子落盘，run 中途崩溃可从
    已落盘 checkpoint 恢复，不依赖进程内存。
    """

    chapter_number: int
    status: str  # CHAPTER_* 之一
    content_path: str | None = None
    score: float | None = None  # overall（judge 加权后），未判分时 None
    token_usage: dict = field(default_factory=dict)
    latency_seconds: float = 0.0
    run_id: str | None = None
    version_id: str | None = None
    context_packet_hash: str | None = None
    failure_stage: str | None = None  # generation | judge | commit | timeout | ...
    failure_reason: str | None = None


@dataclass
class RunStatus:
    """多章 run 的统一状态（§1.3）。

    outcome 由 checkpoints + expected 推导；Rule Gate / 成功率 / 质量统计
    必须经此语义，不得把 timeout/invalid 当普通低分。
    """

    expected_chapters: int
    completed_chapters: int
    checkpoints: list[ChapterCheckpoint] = field(default_factory=list)
    failure_stage: str | None = None
    failure_reason: str | None = None

    @property
    def completion_rate(self) -> float:
        if self.expected_chapters <= 0:
            return 0.0
        return round(self.completed_chapters / self.expected_chapters, 3)

    @property
    def outcome(self) -> str:
        return classify_outcome(self.checkpoints, self.expected_chapters)


def classify_outcome(
    checkpoints: list[ChapterCheckpoint], expected: int
) -> str:
    """从逐章 checkpoint 推导 run 级 outcome。

    判定优先级：无一成功时按失败章主因区分 timeout/invalid/failed；
    部分成功 → partial；全部成功 → completed。
    """
    if expected <= 0:
        return OUTCOME_COMPLETED
    completed = [
        c for c in checkpoints
        if c.status in {CHAPTER_COMPLETED, CHAPTER_RESUMED}
    ]
    if len(completed) >= expected:
        return OUTCOME_COMPLETED
    if not completed:
        # 无一成功：按首个失败章主因分类
        for c in checkpoints:
            if c.status == CHAPTER_TIMEOUT:
                return OUTCOME_TIMEOUT
            if c.status == CHAPTER_INVALID:
                return OUTCOME_INVALID
        return OUTCOME_FAILED
    return OUTCOME_PARTIAL


# ── 原子 save / load ────────────────────────────────────


def save_run_checkpoint(path: Path, status: RunStatus) -> None:
    """原子写 run checkpoint：tmp.replace，崩溃不丢已完成章。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "expected_chapters": status.expected_chapters,
        "completed_chapters": status.completed_chapters,
        "failure_stage": status.failure_stage,
        "failure_reason": status.failure_reason,
        "checkpoints": [asdict(c) for c in status.checkpoints],
    }
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_run_checkpoint(path: Path) -> RunStatus | None:
    """读 run checkpoint；不存在/损坏返回 None（不抛，恢复路径要 robust）。"""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checkpoints = [
        ChapterCheckpoint(
            chapter_number=c.get("chapter_number", 0),
            status=c.get("status", CHAPTER_FAILED),
            content_path=c.get("content_path"),
            score=c.get("score"),
            token_usage=c.get("token_usage") or {},
            latency_seconds=c.get("latency_seconds", 0.0),
            run_id=c.get("run_id"),
            version_id=c.get("version_id"),
            context_packet_hash=c.get("context_packet_hash"),
            failure_stage=c.get("failure_stage"),
            failure_reason=c.get("failure_reason"),
        )
        for c in payload.get("checkpoints", [])
    ]
    return RunStatus(
        expected_chapters=payload.get("expected_chapters", 0),
        completed_chapters=payload.get("completed_chapters", len(checkpoints)),
        checkpoints=checkpoints,
        failure_stage=payload.get("failure_stage"),
        failure_reason=payload.get("failure_reason"),
    )


def checkpoint_from_meta(
    chapter_number: int, meta: dict[str, Any], *, content_path: str | None = None
) -> ChapterCheckpoint:
    """从 adapter generate() 返回的 meta 构造一章 checkpoint（成功路径）。"""
    return ChapterCheckpoint(
        chapter_number=chapter_number,
        status=CHAPTER_RESUMED if meta.get("resumed") else CHAPTER_COMPLETED,
        content_path=content_path,
        score=meta.get("overall") if "overall" in meta else None,
        token_usage=meta.get("token_usage") or {},
        latency_seconds=float(meta.get("elapsed_seconds", 0.0) or 0.0),
        run_id=meta.get("writing_run_id"),
        version_id=meta.get("version_id"),
        context_packet_hash=meta.get("context_packet_hash"),
    )


# ── DurableRun 驱动器 ────────────────────────────────────


async def run_durable(
    *,
    adapter,
    cases: list,
    persist_dir: str,
    checkpoint_path: Path,
    resume: bool = False,
    chapter_timeout: float | None = None,
) -> RunStatus:
    """逐章跑 cases，每章原子落 checkpoint；崩溃/超时可从 checkpoint 恢复。

    - resume 语义分层（方案禁止项 6）：**V2 durable state 是真相源**——始终调用
      adapter.generate，由 adapter 的 resume 按 V2 DB 跳过已 approved 章；
      eval 侧 checkpoint 只补进程内消亡的 token/hash/score（supplementary），
      不充当跳过依据。
    - 失败不丢已完章：失败章记 CHAPTER_FAILED/TIMEOUT/INVALID 后立即落盘并中止，
      已完章 checkpoint 仍在（§1.2 验收）。
    - 不在此处判分（score 留 None 或从 prior 补）；judge 集成留给上层。

    要求 adapter 暴露 `_chapter_number(case)`（NovelAgentAdapter 满足）。
    """
    expected = len(cases)
    prior = load_run_checkpoint(Path(checkpoint_path)) if resume else None
    prior_by_num = {c.chapter_number: c for c in (prior.checkpoints if prior else [])}
    status = RunStatus(expected_chapters=expected, completed_chapters=0)

    for case in cases:
        cn = adapter._chapter_number(case)
        ck: ChapterCheckpoint
        try:
            coro = adapter.generate(case)
            gen = await (
                asyncio.wait_for(coro, timeout=chapter_timeout)
                if chapter_timeout else coro
            )
        except TimeoutError:
            ck = ChapterCheckpoint(
                chapter_number=cn, status=CHAPTER_TIMEOUT,
                failure_stage="timeout",
                failure_reason=f"timeout_seconds={chapter_timeout}",
            )
            status.checkpoints.append(ck)
            status.failure_stage, status.failure_reason = ck.failure_stage, ck.failure_reason
            status.completed_chapters = _count_completed(status)
            save_run_checkpoint(Path(checkpoint_path), status)
            break
        except Exception as exc:  # noqa: BLE001 - 失败章不中止整条评测的可复现性
            msg = str(exc).lower()
            status_val = CHAPTER_INVALID if any(
                k in msg for k in ("empty", "invalid", "censor", "parse")
            ) else CHAPTER_FAILED
            ck = ChapterCheckpoint(
                chapter_number=cn, status=status_val,
                failure_stage="generation", failure_reason=str(exc)[:500],
            )
            status.checkpoints.append(ck)
            status.failure_stage, status.failure_reason = ck.failure_stage, ck.failure_reason
            status.completed_chapters = _count_completed(status)
            save_run_checkpoint(Path(checkpoint_path), status)
            break
        # 成功（含 adapter resume 命中）
        ck = checkpoint_from_meta(cn, gen.meta)
        if gen.meta.get("resumed") and cn in prior_by_num:
            # eval checkpoint 补全进程内消亡的 token/hash/score（V2 DB 只留 content/run）
            prev = prior_by_num[cn]
            ck.token_usage = prev.token_usage or ck.token_usage
            ck.context_packet_hash = prev.context_packet_hash or ck.context_packet_hash
            ck.score = prev.score if ck.score is None else ck.score
            ck.latency_seconds = prev.latency_seconds or ck.latency_seconds
            ck.run_id = prev.run_id or ck.run_id
        status.checkpoints.append(ck)
        status.completed_chapters = _count_completed(status)
        save_run_checkpoint(Path(checkpoint_path), status)

    return status


def _count_completed(status: RunStatus) -> int:
    return sum(
        1 for c in status.checkpoints
        if c.status in {CHAPTER_COMPLETED, CHAPTER_RESUMED}
    )
