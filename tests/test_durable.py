# tests/test_durable.py
"""Phase 1.2/1.3 durable 契约测试（纯逻辑，不依赖 LLM / graph）。

- classify_outcome：completed / partial / failed / timeout / invalid 五类
- save/load_run_checkpoint：原子 round-trip + 损坏文件不抛
- 失败不丢已完章：部分章已落盘，后续失败时已完章仍可从 checkpoint 恢复
- checkpoint_from_meta：从 adapter meta 构造
"""

from novel_agent_eval.durable import (
    CHAPTER_COMPLETED,
    CHAPTER_FAILED,
    CHAPTER_INVALID,
    CHAPTER_RESUMED,
    CHAPTER_TIMEOUT,
    OUTCOME_COMPLETED,
    OUTCOME_FAILED,
    OUTCOME_INVALID,
    OUTCOME_PARTIAL,
    OUTCOME_TIMEOUT,
    ChapterCheckpoint,
    RunStatus,
    checkpoint_from_meta,
    classify_outcome,
    load_run_checkpoint,
    save_run_checkpoint,
)


def _ck(n, status, **kw) -> ChapterCheckpoint:
    return ChapterCheckpoint(chapter_number=n, status=status, **kw)


# ── §1.3 classify_outcome 五类 ──


def test_classify_all_completed():
    cks = [_ck(i, CHAPTER_COMPLETED) for i in range(1, 6)]
    assert classify_outcome(cks, expected=5) == OUTCOME_COMPLETED


def test_classify_partial_when_some_succeeded():
    cks = [
        _ck(1, CHAPTER_COMPLETED),
        _ck(2, CHAPTER_COMPLETED),
        _ck(3, CHAPTER_FAILED, failure_stage="generation"),
    ]
    assert classify_outcome(cks, expected=3) == OUTCOME_PARTIAL


def test_classify_failed_when_none_succeeded():
    cks = [_ck(1, CHAPTER_FAILED, failure_stage="generation")]
    assert classify_outcome(cks, expected=3) == OUTCOME_FAILED


def test_classify_timeout_when_none_succeeded_and_timeout_present():
    cks = [
        _ck(1, CHAPTER_TIMEOUT, failure_stage="timeout"),
        _ck(2, CHAPTER_FAILED, failure_stage="generation"),
    ]
    assert classify_outcome(cks, expected=2) == OUTCOME_TIMEOUT


def test_classify_invalid_when_none_succeeded_and_invalid_present():
    cks = [_ck(1, CHAPTER_INVALID, failure_stage="generation")]
    assert classify_outcome(cks, expected=2) == OUTCOME_INVALID


def test_classify_resumed_counts_as_completed():
    cks = [
        _ck(1, CHAPTER_RESUMED),
        _ck(2, CHAPTER_COMPLETED),
    ]
    assert classify_outcome(cks, expected=2) == OUTCOME_COMPLETED


def test_classify_zero_expected_is_completed():
    assert classify_outcome([], expected=0) == OUTCOME_COMPLETED


# ── §1.2 save / load round-trip ──


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "run.json"
    status = RunStatus(
        expected_chapters=5,
        completed_chapters=3,
        failure_stage="generation",
        failure_reason="API 429",
        checkpoints=[
            _ck(1, CHAPTER_COMPLETED, score=82.0, run_id="r1", context_packet_hash="h1", token_usage={"total_tokens": 100}),
            _ck(2, CHAPTER_RESUMED),
            _ck(3, CHAPTER_COMPLETED, score=78.5),
            _ck(4, CHAPTER_TIMEOUT, failure_stage="timeout"),
        ],
    )
    save_run_checkpoint(path, status)
    loaded = load_run_checkpoint(path)
    assert loaded is not None
    assert loaded.expected_chapters == 5
    assert loaded.completed_chapters == 3
    assert loaded.completion_rate == 0.6
    assert loaded.outcome == OUTCOME_PARTIAL
    assert len(loaded.checkpoints) == 4
    assert loaded.checkpoints[0].score == 82.0
    assert loaded.checkpoints[0].token_usage == {"total_tokens": 100}
    assert loaded.checkpoints[3].status == CHAPTER_TIMEOUT


def test_load_missing_returns_none(tmp_path):
    assert load_run_checkpoint(tmp_path / "nope.json") is None


def test_load_corrupt_returns_none(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_run_checkpoint(path) is None


def test_save_atomic_failure_does_not_lose_prior(tmp_path):
    """模拟：3 章已落盘，第 4 章生成中崩溃 → 已完 3 章仍可从 checkpoint 恢复。"""
    path = tmp_path / "run.json"
    # 前 3 章成功落盘
    status = RunStatus(
        expected_chapters=5,
        completed_chapters=3,
        checkpoints=[_ck(i, CHAPTER_COMPLETED, score=80.0) for i in range(1, 4)],
    )
    save_run_checkpoint(path, status)
    # 模拟第 4 章 crash：不更新 checkpoint（等于崩溃前未落盘）
    # 进程恢复后 load → 前 3 章仍在
    loaded = load_run_checkpoint(path)
    assert loaded is not None
    assert loaded.completed_chapters == 3
    assert all(c.status == CHAPTER_COMPLETED for c in loaded.checkpoints)


# ── checkpoint_from_meta ──


def test_checkpoint_from_meta_completed():
    meta = {
        "writing_run_id": "run-1",
        "context_packet_hash": "hash-abc",
        "token_usage": {"total_tokens": 5000},
        "elapsed_seconds": 12.3,
        "version_id": "v1",
    }
    ck = checkpoint_from_meta(3, meta, content_path="/tmp/ch3.txt")
    assert ck.chapter_number == 3
    assert ck.status == CHAPTER_COMPLETED
    assert ck.run_id == "run-1"
    assert ck.context_packet_hash == "hash-abc"
    assert ck.token_usage == {"total_tokens": 5000}
    assert ck.latency_seconds == 12.3
    assert ck.content_path == "/tmp/ch3.txt"


def test_checkpoint_from_meta_resumed():
    ck = checkpoint_from_meta(5, {"resumed": True, "writing_run_id": "r"})
    assert ck.status == CHAPTER_RESUMED
    assert ck.latency_seconds == 0.0


def test_run_status_outcome_property():
    status = RunStatus(
        expected_chapters=4,
        completed_chapters=4,
        checkpoints=[_ck(i, CHAPTER_COMPLETED) for i in range(1, 5)],
    )
    assert status.outcome == OUTCOME_COMPLETED
    assert status.completion_rate == 1.0
