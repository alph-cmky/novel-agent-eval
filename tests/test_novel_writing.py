# tests/test_novel_writing.py
"""NovelWritingAgent 适配器 mock 测试：不调真实 NwA，全 monkeypatch subprocess。"""
import asyncio
from pathlib import Path

import pytest

from novel_agent_eval.agents.base import ModelConfig
from novel_agent_eval.agents.novel_writing import (
    NovelWritingAgentAdapter,
    build_nwa_brief,
)
from novel_agent_eval.dataset.schema import EvalCase


def _make_case() -> EvalCase:
    return EvalCase(
        name="test_nwa_01",
        stage="opening",
        story_outline="主角穿越到玄幻大陆，立志成为剑仙。",
        previous_context="第一章：主角在异世界醒来，发现体内有神秘力量。",
        target_chapter_outline="第二章：主角拜入青云剑派，随师父学剑。",
        word_target=3000,
    )


def _make_model() -> ModelConfig:
    return ModelConfig(
        base_url="https://api.stepfun.test/v1",
        api_key="k",
        model="step-3.7-flash",
    )


def _repo_with_entry(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    (repo_dir / "examples").mkdir(parents=True)
    (repo_dir / "examples" / "run_with_llm.py").write_text("", encoding="utf-8")
    return repo_dir


class _FakeProc:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


async def _fake_subprocess(*args, **kwargs):
    return _FakeProc()


async def _fake_failing_subprocess(*args, **kwargs):
    return _FakeProc(returncode=1, stderr=b"boom error")


# ── build_nwa_brief ──────────────────────────────────────


def test_build_nwa_brief_includes_all_sections():
    case = _make_case()
    b = build_nwa_brief(case)

    assert case.story_outline in b
    assert case.previous_context in b
    assert case.target_chapter_outline in b
    assert "目标约 3000 字" in b


def test_build_nwa_brief_skips_blank_sections():
    case = _make_case()
    case.previous_context = ""
    b = build_nwa_brief(case)
    assert "前文上下文" not in b
    assert "本章大纲" in b
    assert "目标约 3000 字" in b


# ── _check_repo ──────────────────────────────────────────


def test_check_repo_raises_when_entry_missing(tmp_path):
    adapter = NovelWritingAgentAdapter(repo_dir=tmp_path)
    with pytest.raises(RuntimeError, match="NovelWritingAgent 仓库缺失"):
        adapter._check_repo()


def test_check_repo_ok_when_entry_present(tmp_path):
    adapter = NovelWritingAgentAdapter(repo_dir=_repo_with_entry(tmp_path))
    adapter._check_repo()  # 不抛即通过


# ── _render_config / _write_config / _restore_config ─────


def test_render_config_contains_all_fields():
    adapter = NovelWritingAgentAdapter(repo_dir=Path("/tmp/x"), model=_make_model())
    cfg = adapter._render_config()
    assert "api_key: k" in cfg
    assert "api_base: https://api.stepfun.test/v1" in cfg
    assert "model: step-3.7-flash" in cfg
    assert "provider: custom" in cfg


def test_write_config_returns_original_and_restore(tmp_path):
    repo_dir = _repo_with_entry(tmp_path)
    cfg = repo_dir / "config" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("original-config", encoding="utf-8")

    adapter = NovelWritingAgentAdapter(repo_dir=repo_dir, model=_make_model())
    original = adapter._write_config()
    assert original == "original-config"
    assert "api_key: k" in cfg.read_text(encoding="utf-8")

    adapter._restore_config(original)
    assert cfg.read_text(encoding="utf-8") == "original-config"


def test_write_config_none_model_is_noop(tmp_path):
    repo_dir = _repo_with_entry(tmp_path)
    adapter = NovelWritingAgentAdapter(repo_dir=repo_dir)  # model=None
    assert adapter._write_config() is None
    assert not (repo_dir / "config" / "config.yaml").exists()


def test_restore_config_deletes_when_no_original(tmp_path):
    repo_dir = _repo_with_entry(tmp_path)
    adapter = NovelWritingAgentAdapter(repo_dir=repo_dir, model=_make_model())
    assert adapter._write_config() is None  # 原无 config.yaml
    assert (repo_dir / "config" / "config.yaml").exists()

    adapter._restore_config(None)
    assert not (repo_dir / "config" / "config.yaml").exists()


# ── _read_output ─────────────────────────────────────────


def test_read_output_prefers_revision(tmp_path):
    base = tmp_path / "workspace" / "novel_projects" / "p1" / "outputs" / "chapters"
    base.mkdir(parents=True)
    (base / "chapter_001_draft.md").write_text("draft", encoding="utf-8")
    (base / "chapter_001_revision.md").write_text("revision", encoding="utf-8")

    out = NovelWritingAgentAdapter._read_output(tmp_path, tmp_path)
    assert out == "revision"


def test_read_output_falls_back_to_draft(tmp_path):
    base = tmp_path / "workspace" / "novel_projects" / "p1" / "outputs" / "chapters"
    base.mkdir(parents=True)
    (base / "chapter_001_draft.md").write_text("draft", encoding="utf-8")

    out = NovelWritingAgentAdapter._read_output(tmp_path, tmp_path)
    assert out == "draft"


def test_read_output_raises_when_empty(tmp_path):
    with pytest.raises(RuntimeError, match="未生成章节产物"):
        NovelWritingAgentAdapter._read_output(tmp_path, tmp_path)


# ── generate 全链路（mock subprocess + _read_output） ────


def test_generate_returns_content_and_meta(monkeypatch, tmp_path):
    repo_dir = _repo_with_entry(tmp_path)
    monkeypatch.setattr(
        "novel_agent_eval.agents.novel_writing.asyncio.create_subprocess_exec",
        _fake_subprocess,
    )
    monkeypatch.setattr(
        NovelWritingAgentAdapter, "_read_output",
        staticmethod(lambda r, w: "固定正文"),
    )
    adapter = NovelWritingAgentAdapter(repo_dir=repo_dir, model=_make_model())
    gen = asyncio.run(adapter.generate(_make_case()))

    assert gen.content == "固定正文"
    assert gen.meta["adapter"] == "novel_writing_agent"
    assert gen.meta["model"] == "step-3.7-flash"
    assert gen.meta["elapsed_seconds"] >= 0
    assert gen.meta["tokens"] is None
    # 注入 model 后临时写 config.yaml，跑完恢复（原无 config → 删除）
    assert not (repo_dir / "config" / "config.yaml").exists()


def test_generate_raises_on_nonzero_exit(monkeypatch, tmp_path):
    repo_dir = _repo_with_entry(tmp_path)
    monkeypatch.setattr(
        "novel_agent_eval.agents.novel_writing.asyncio.create_subprocess_exec",
        _fake_failing_subprocess,
    )
    adapter = NovelWritingAgentAdapter(repo_dir=repo_dir)
    with pytest.raises(RuntimeError, match="退出码 1"):
        asyncio.run(adapter.generate(_make_case()))
