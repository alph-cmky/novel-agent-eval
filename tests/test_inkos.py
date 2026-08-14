# tests/test_inkos.py
"""InkOS 适配器 mock 测试：不调真实 InkOS CLI / npx，全 monkeypatch subprocess。

实测校准（Task 13 冒烟）：InkOS 单章流程 = init → book create --target-chapters 1
→ draft；产物 books/<书名>/chapters/0001_*.md（short run 要求 12-18 章，非单章入口）。
"""
import asyncio
from pathlib import Path

import pytest

from novel_agent_eval.agents.base import ModelConfig
from novel_agent_eval.agents.inkos import InkOSAdapter, build_inkos_direction
from novel_agent_eval.dataset.schema import EvalCase


def _make_case() -> EvalCase:
    return EvalCase(
        name="test_inkos_01",
        stage="opening",
        genre="玄幻",
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


async def _fake_run_ok(self, cmd, cwd, env):
    return "ok"


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


# ── _inkos_genre ─────────────────────────────────────────


def test_inkos_genre_maps_chinese_to_slug():
    assert InkOSAdapter._inkos_genre("玄幻") == "xuanhuan"
    assert InkOSAdapter._inkos_genre("都市") == "urban"
    assert InkOSAdapter._inkos_genre("科幻") == "sci-fi"
    assert InkOSAdapter._inkos_genre("未知题材") == "xuanhuan"


# ── _exe ─────────────────────────────────────────────────


def test_exe_raises_when_inkos_missing(monkeypatch):
    monkeypatch.setattr("novel_agent_eval.agents.inkos.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="未安装 InkOS"):
        InkOSAdapter()._exe()


def test_exe_returns_path_when_found(monkeypatch):
    monkeypatch.setattr(
        "novel_agent_eval.agents.inkos.shutil.which", lambda _: "/usr/local/bin/inkos"
    )
    assert InkOSAdapter()._exe() == "/usr/local/bin/inkos"


# ── _run ─────────────────────────────────────────────────


def test_run_ok_returns_stdout(monkeypatch):
    async def _fake(*args, **kwargs):
        return _FakeProc(returncode=0, stdout=b"stdout-log")

    monkeypatch.setattr(
        "novel_agent_eval.agents.inkos.asyncio.create_subprocess_exec", _fake
    )
    out = asyncio.run(InkOSAdapter()._run(["inkos"], Path("/tmp"), {}))
    assert "stdout-log" in out


def test_run_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        "novel_agent_eval.agents.inkos.asyncio.create_subprocess_exec",
        _fake_failing_subprocess,
    )
    with pytest.raises(RuntimeError, match="退出码 1"):
        asyncio.run(InkOSAdapter()._run(["inkos"], Path("/tmp"), {}))


def test_run_kills_proc_on_timeout(monkeypatch):
    class _HangingProc:
        def __init__(self):
            self.killed = False
            self.waited = False

        async def communicate(self):
            await asyncio.sleep(60)
            return b"", b""

        def kill(self):
            self.killed = True

        async def wait(self):
            self.waited = True
            return -9

    proc = _HangingProc()

    async def _fake(*args, **kwargs):
        return proc

    monkeypatch.setattr(
        "novel_agent_eval.agents.inkos.asyncio.create_subprocess_exec", _fake
    )
    with pytest.raises(TimeoutError):
        asyncio.run(InkOSAdapter(timeout=0.01)._run(["inkos"], Path("/tmp"), {}))
    # 超时后子进程必须被 kill + wait 回收，不能留孤儿进程
    assert proc.killed
    assert proc.waited


# ── _brief_file ──────────────────────────────────────────


def test_brief_file_writes_direction(tmp_path):
    path = InkOSAdapter._brief_file(tmp_path, "创作方向文案")
    assert path.read_text(encoding="utf-8") == "创作方向文案"


# ── _read_output ─────────────────────────────────────────


def test_read_output_reads_first_chapter(tmp_path):
    (tmp_path / "books/冒烟测试/chapters").mkdir(parents=True)
    (tmp_path / "books/冒烟测试/chapters/0001_第一章.md").write_text("正文XYZ", encoding="utf-8")
    assert InkOSAdapter._read_output(tmp_path) == "正文XYZ"


def test_read_output_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError, match="InkOS 未生成"):
        InkOSAdapter._read_output(tmp_path)


def test_read_output_raises_when_content_empty(tmp_path):
    (tmp_path / "books/x/chapters").mkdir(parents=True)
    (tmp_path / "books/x/chapters/0001_第一章.md").write_text("   \n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="产物为空"):
        InkOSAdapter._read_output(tmp_path)


# ── generate 全链路（mock _run + _read_output） ──────────


def test_generate_returns_content_and_meta(monkeypatch):
    monkeypatch.setattr(InkOSAdapter, "_exe", lambda self: "/usr/local/bin/inkos")
    monkeypatch.setattr(InkOSAdapter, "_run", _fake_run_ok)
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


def test_generate_raises_when_inkos_not_installed(monkeypatch):
    def _no_inkos(self):
        raise RuntimeError("PATH 上无 inkos，未安装 InkOS。请先全局安装：npm i -g @actalk/inkos --ignore-scripts")

    monkeypatch.setattr(InkOSAdapter, "_exe", _no_inkos)
    with pytest.raises(RuntimeError, match="未安装 InkOS"):
        asyncio.run(InkOSAdapter().generate(_make_case()))
