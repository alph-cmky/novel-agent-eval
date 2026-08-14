# tests/test_inkos.py
"""InkOS 适配器 mock 测试：不调真实 InkOS CLI / npx，全 monkeypatch subprocess。"""
import asyncio

import pytest

from novel_agent_eval.agents.base import ModelConfig
from novel_agent_eval.agents.inkos import InkOSAdapter, build_inkos_direction
from novel_agent_eval.dataset.schema import EvalCase


def _make_case() -> EvalCase:
    return EvalCase(
        name="test_inkos_01",
        stage="opening",
        story_outline="主角穿越到玄幻大陆，立志成为剑仙。",
        previous_context="第一章：主角在异世界醒来，发现体内有神秘力量。",
        target_chapter_outline="第二章：主角拜入青云剑派，随师父学剑。",
        word_target=3000,
    )


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


# ── build_inkos_direction ────────────────────────────────


def test_build_inkos_direction_includes_all_sections():
    case = _make_case()
    d = build_inkos_direction(case)

    assert case.story_outline in d
    assert case.previous_context in d
    assert case.target_chapter_outline in d
    assert "目标约 3000 字" in d


def test_build_inkos_direction_skips_blank_sections():
    case = _make_case()
    case.story_outline = "  "
    d = build_inkos_direction(case)
    assert "全书大纲" not in d
    assert "本章大纲" in d
    assert "目标约 3000 字" in d


# ── _command ─────────────────────────────────────────────


def test_command_uses_npx_when_inkos_missing(monkeypatch):
    monkeypatch.setattr("novel_agent_eval.agents.inkos.shutil.which", lambda _: None)
    adapter = InkOSAdapter()
    cmd = adapter._command("方向", 3000)
    assert cmd == [
        "npx", "-y", "@actalk/inkos",
        "short", "run", "--direction", "方向", "--chapters", "1", "--chars", "3000",
    ]


def test_command_uses_inkos_exe_when_found(monkeypatch):
    monkeypatch.setattr(
        "novel_agent_eval.agents.inkos.shutil.which", lambda _: "/usr/bin/inkos"
    )
    adapter = InkOSAdapter()
    cmd = adapter._command("方向", 3000)
    assert cmd[0] == "/usr/bin/inkos"
    assert cmd[1:3] == ["short", "run"]


# ── _env ─────────────────────────────────────────────────


def test_env_not_injected_when_model_none():
    adapter = InkOSAdapter()
    env = adapter._env()
    assert "INKOS_LLM_PROVIDER" not in env
    assert "INKOS_LLM_BASE_URL" not in env
    assert "INKOS_LLM_API_KEY" not in env


def test_env_injected_when_model_set():
    adapter = InkOSAdapter(model=ModelConfig(
        base_url="https://api.stepfun.test/v1",
        api_key="k123",
        model="step-3.7-flash",
        temperature=0.5,
    ))
    env = adapter._env()
    assert env["INKOS_LLM_PROVIDER"] == "custom"
    assert env["INKOS_LLM_BASE_URL"] == "https://api.stepfun.test/v1"
    assert env["INKOS_LLM_API_KEY"] == "k123"
    assert env["INKOS_LLM_MODEL"] == "step-3.7-flash"
    assert env["INKOS_LLM_TEMPERATURE"] == "0.5"


# ── _read_output ─────────────────────────────────────────


def test_read_output_reads_first_match(tmp_path):
    (tmp_path / "shorts/foo/final").mkdir(parents=True)
    (tmp_path / "shorts/foo/final/full.md").write_text("正文XYZ", encoding="utf-8")
    assert InkOSAdapter._read_output(tmp_path) == "正文XYZ"


def test_read_output_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError, match="InkOS 未生成"):
        InkOSAdapter._read_output(tmp_path)


# ── generate 全链路（mock subprocess + _read_output） ────


def test_generate_returns_content_and_meta(monkeypatch):
    monkeypatch.setattr(
        "novel_agent_eval.agents.inkos.asyncio.create_subprocess_exec",
        _fake_subprocess,
    )
    monkeypatch.setattr(InkOSAdapter, "_read_output", staticmethod(lambda wd: "固定正文"))
    adapter = InkOSAdapter(model=ModelConfig(
        base_url="https://api.stepfun.test/v1", api_key="k", model="step-3.7-flash"
    ))
    gen = asyncio.run(adapter.generate(_make_case()))

    assert gen.content == "固定正文"
    assert gen.meta["adapter"] == "inkos"
    assert gen.meta["model"] == "step-3.7-flash"
    assert gen.meta["elapsed_seconds"] >= 0
    assert gen.meta["tokens"] is None


def test_generate_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        "novel_agent_eval.agents.inkos.asyncio.create_subprocess_exec",
        _fake_failing_subprocess,
    )
    adapter = InkOSAdapter()
    with pytest.raises(RuntimeError, match="退出码 1"):
        asyncio.run(adapter.generate(_make_case()))
