# novel_agent_eval/agents/inkos.py
"""InkOS 对手适配器：subprocess 调 InkOS CLI（TypeScript npm 包）写章节。

InkOS 不是 Python 库/HTTP API，故通过 subprocess 跑 CLI 写单章，读产物
books/<书名>/chapters/0001_*.md 得到正文。模型注入走 env INKOS_LLM_*（OpenAI 兼容）。

实测校准（Task 13 冒烟）：
- `inkos short run --chapters 1` 不可用：short run 要求 chapters 在 12-18 之间，
  不是单章入口。单章正确流程是 `init` → `book create --target-chapters 1` →
  `draft`（跳过 audit/revise，比 `write next` 轻）。
- 产物路径不是 shorts/*/final/full.md，而是 books/<书名>/chapters/0001_*.md。
- npx -y @actalk/inkos 首次运行会挂起（某依赖 postinstall 需从被墙 host 拉资源），
  故未装 inkos 时直接 raise（提示 npm i -g @actalk/inkos --ignore-scripts），不回退 npx。
- book create 需要 --brief 文件路径（非字符串），故把创作方向写入 workdir/brief.md。
"""
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter, ModelConfig
from novel_agent_eval.dataset.schema import EvalCase

# EvalCase.genre（中文）→ InkOS genre slug（`inkos genre list` 实测值）；未知名 → xuanhuan
_INKOS_GENRE_MAP = {
    "玄幻": "xuanhuan",
    "都市": "urban",
    "科幻": "sci-fi",
    "仙侠": "xianxia",
    "武侠": "xianxia",
}


def build_inkos_direction(case: EvalCase) -> str:
    """EvalCase → 创作方向文案：写进 brief 文件（book create）与 draft --context（纯函数，可单测）。"""
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
    def _inkos_genre(genre: str) -> str:
        """EvalCase.genre（中文）→ InkOS genre slug；未知回落 xuanhuan。"""
        return _INKOS_GENRE_MAP.get(genre, "xuanhuan")

    def _exe(self) -> str:
        """inkos CLI 可执行文件；未装则 raise（npx 拉包因 postinstall 挂起，不回退）。"""
        exe = shutil.which("inkos")
        if exe is None:
            raise RuntimeError(
                "PATH 上无 inkos，未安装 InkOS。请先全局安装：npm i -g @actalk/inkos --ignore-scripts"
            )
        return exe

    async def _run(self, cmd: list[str], cwd: Path, env: dict[str, str]) -> str:
        """跑单个 inkos 子命令；非 0 退出则 raise（stdout+stderr 尾部），返回 stdout。"""
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError:
            # wait_for 只 cancel communicate()，不杀子进程；kill 后 wait 回收，防孤儿进程
            proc.kill()
            await proc.wait()
            raise
        out = stdout.decode("utf-8", "replace")
        err = stderr.decode("utf-8", "replace")
        if proc.returncode != 0:
            raise RuntimeError(f"InkOS 退出码 {proc.returncode}：{(out + err)[-1000:]}")
        return out

    @staticmethod
    def _brief_file(workdir: Path, direction: str) -> Path:
        """把创作方向写入 brief 文件（book create --brief 需要文件路径）。"""
        path = workdir / "brief.md"
        path.write_text(direction, encoding="utf-8")
        return path

    @staticmethod
    def _read_output(workdir: Path) -> str:
        """读 books/<书名>/chapters/0001_*.md（首个匹配；未生成或内容为空则 raise）。"""
        matches = sorted(workdir.glob("books/*/chapters/0001_*.md"))
        if not matches:
            raise RuntimeError("InkOS 未生成 books/*/chapters/0001_*.md 产物")
        content = matches[0].read_text(encoding="utf-8").strip()
        if not content:
            raise RuntimeError(f"InkOS 产物为空：{matches[0]}")
        return content

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        direction = build_inkos_direction(case)
        workdir = Path(tempfile.mkdtemp(prefix="inkos_eval_"))
        try:
            env = self._env()
            exe = self._exe()
            start = time.monotonic()
            # 1) init 项目：inkos init <name> 在当前目录下创建 <name>/ 子目录
            await self._run([exe, "init", case.name, "--lang", "zh"], workdir, env)
            proj = workdir / case.name
            # 2) book create：AI 生成基础设定，target-chapters 1（short run 不支持单章）
            await self._run([
                exe, "book", "create",
                "--title", case.name,
                "--genre", self._inkos_genre(case.genre),
                "--target-chapters", "1",
                "--chapter-words", str(case.word_target),
                "--brief", str(self._brief_file(workdir, direction)),
            ], proj, env)
            # 3) draft 写草稿：跳过 audit/revise，产出 chapters/0001_*.md
            await self._run([
                exe, "draft",
                "--words", str(case.word_target),
                "--context", direction,
            ], proj, env)
            elapsed = time.monotonic() - start
            content = self._read_output(proj)
            return GeneratedChapter(content=content, meta={
                "adapter": self.name,
                "model": self._model.model if self._model else os.environ.get("INKOS_LLM_MODEL"),
                "elapsed_seconds": round(elapsed, 3),
                "tokens": None,
            })
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
