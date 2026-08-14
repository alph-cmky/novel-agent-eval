# novel_agent_eval/agents/inkos.py
"""InkOS 对手适配器：subprocess 调 InkOS CLI（TypeScript npm 包）写章节。

InkOS 不是 Python 库/HTTP API，故通过 subprocess 跑 `inkos short run`，读产物
shorts/<故事名>/final/full.md 得到正文。模型注入走 env INKOS_LLM_*（OpenAI 兼容）。
未装 InkOS（PATH 无 inkos 且 npx 不可用）时 raise，不静默返回空。
"""
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter, ModelConfig
from novel_agent_eval.dataset.schema import EvalCase


def build_inkos_direction(case: EvalCase) -> str:
    """EvalCase → inkos short run --direction 的创作方向文案（纯函数，可单测）。"""
    parts = []
    if case.story_outline.strip():
        parts.append(f"全书大纲：{case.story_outline}")
    if case.previous_context.strip():
        parts.append(f"前文上下文：{case.previous_context}")
    if case.target_chapter_outline.strip():
        parts.append(f"本章大纲：{case.target_chapter_outline}")
    parts.append(f"目标约 {case.word_target} 字，直接输出本章正文。")
    return "\n".join(parts)


class InkOSAdapter(AgentAdapter):
    name = "inkos"

    def __init__(self, model: ModelConfig | None = None, timeout: float = 600.0):
        self._model = model
        self._timeout = timeout

    def _command(self, direction: str, chars: int) -> list[str]:
        """short run 命令：优先 PATH 上的 inkos，回退 npx -y @actalk/inkos。"""
        args = ["short", "run", "--direction", direction, "--chapters", "1", "--chars", str(chars)]
        exe = shutil.which("inkos")
        if exe:
            return [exe, *args]
        return ["npx", "-y", "@actalk/inkos", *args]

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._model is not None:
            env["INKOS_LLM_PROVIDER"] = "custom"
            env["INKOS_LLM_BASE_URL"] = self._model.base_url
            env["INKOS_LLM_API_KEY"] = self._model.api_key
            env["INKOS_LLM_MODEL"] = self._model.model
            env["INKOS_LLM_TEMPERATURE"] = str(self._model.temperature)
        return env

    @staticmethod
    def _read_output(workdir: Path) -> str:
        """读 shorts/<故事名>/final/full.md（首个匹配，未生成则 raise）。"""
        matches = sorted(workdir.glob("shorts/*/final/full.md"))
        if not matches:
            raise RuntimeError("InkOS 未生成 shorts/*/final/full.md 产物")
        return matches[0].read_text(encoding="utf-8").strip()

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        direction = build_inkos_direction(case)
        workdir = Path(tempfile.mkdtemp(prefix="inkos_eval_"))
        try:
            cmd = self._command(direction, case.word_target)
            env = self._env()
            start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(workdir), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            elapsed = time.monotonic() - start
            if proc.returncode != 0:
                raise RuntimeError(f"InkOS 退出码 {proc.returncode}：{stderr.decode('utf-8', 'replace')[-1000:]}")
            content = self._read_output(workdir)
            return GeneratedChapter(content=content, meta={
                "adapter": self.name,
                "model": self._model.model if self._model else os.environ.get("INKOS_LLM_MODEL"),
                "elapsed_seconds": round(elapsed, 3),
                "tokens": None,
            })
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
