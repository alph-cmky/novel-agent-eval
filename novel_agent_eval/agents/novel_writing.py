# novel_agent_eval/agents/novel_writing.py
"""NovelWritingAgent 对手适配器：subprocess 调 python -m examples.run_with_llm 写章节。

NovelWritingAgent 是 Python 包，但接口是 CLI 入口（examples.run_with_llm），模型走
config/config.yaml。适配器需传入 repo_dir（已 checkout 的仓库路径，本任务不负责安装）。
模型非空时临时写 config.yaml，跑完恢复；未传 model 则用仓库既有配置不动。

实测校准（Task 13 冒烟）：
- run_with_llm 用相对路径读 config/config.yaml 与 workspace/，都相对 cwd 解析
  （examples/run_with_llm.py:27 / :56）。因此模型注入的 config.yaml 必须写在
  cwd（临时工作目录）里，而非 repo_dir；产物也读 cwd/workspace/...。
- provider 枚举只有 anthropic/openai（novel_writing_agent/schema/schema.py），
  无 custom。OpenAI 兼容第三方端点用 provider: openai，api_base 原样使用
  （llm/llm_wrapper.py 对非 MiniMax 域名不追加后缀）。
"""
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter, ModelConfig
from novel_agent_eval.dataset.schema import EvalCase


def build_nwa_brief(case: EvalCase) -> str:
    """EvalCase → run_with_llm --brief 创作简报（纯函数，可单测）。"""
    parts = []
    if case.story_outline.strip():
        parts.append(f"全书大纲：{case.story_outline}")
    if case.previous_context.strip():
        parts.append(f"前文上下文：{case.previous_context}")
    if case.target_chapter_outline.strip():
        parts.append(f"本章大纲：{case.target_chapter_outline}")
    parts.append(f"目标约 {case.word_target} 字。")
    return "\n".join(parts)


class NovelWritingAgentAdapter(AgentAdapter):
    name = "novel_writing_agent"

    def __init__(self, repo_dir: Path, model: ModelConfig | None = None, timeout: float = 600.0):
        self._repo_dir = Path(repo_dir)
        self._model = model
        self._timeout = timeout

    def _check_repo(self) -> None:
        if not (self._repo_dir / "examples" / "run_with_llm.py").exists():
            raise RuntimeError(f"NovelWritingAgent 仓库缺失或入口不存在：{self._repo_dir}")

    def _render_config(self) -> str:
        assert self._model is not None
        return (
            f"api_key: {self._model.api_key}\n"
            f"api_base: {self._model.base_url}\n"
            f"model: {self._model.model}\n"
            f"provider: openai\n"
        )

    @staticmethod
    def _config_path(workdir: Path) -> Path:
        """模型注入时写入的 config.yaml 路径：必须在 cwd（临时工作目录）。

        run_with_llm 用相对路径 Path("config/config.yaml") 读配置，故写到
        workdir/config/config.yaml 才能在 cwd=workdir 时被读到。
        """
        return workdir / "config" / "config.yaml"

    def _write_config(self, workdir: Path) -> str | None:
        """model 非空时向 workdir/config/config.yaml 写注入配置，返回原内容。

        原内容只可能是 None：workdir 是每次 generate 新建的临时目录，不存在
        既有 config.yaml。返回类型保留 str | None 以覆盖写入失败/恢复的通用语义。
        """
        if self._model is None:
            return None
        cfg = self._config_path(workdir)
        original = cfg.read_text(encoding="utf-8") if cfg.exists() else None
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(self._render_config(), encoding="utf-8")
        return original

    def _restore_config(self, original: str | None, workdir: Path) -> None:
        if self._model is None:
            return
        cfg = self._config_path(workdir)
        if original is None:
            cfg.unlink(missing_ok=True)
        else:
            cfg.write_text(original, encoding="utf-8")

    @staticmethod
    def _read_output(repo_dir: Path, workdir: Path) -> str:
        """读 workspace/novel_projects/<project_id>/outputs/chapters/chapter_001_*.md。

        取 revision 优先，回退 draft；project_id 为运行自动生成，取最新目录。
        """
        base = workdir / "workspace" / "novel_projects"
        candidates = list(base.glob("*/outputs/chapters/chapter_001_*.md"))
        if not candidates:
            raise RuntimeError(f"NovelWritingAgent 未生成章节产物：{base}")
        # revision 优先于 draft
        rev = [p for p in candidates if p.name == "chapter_001_revision.md"]
        return (rev[0] if rev else candidates[0]).read_text(encoding="utf-8").strip()

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        self._check_repo()
        # NwA 产物固定写 <cwd>/workspace/novel_projects/，故在临时目录内运行，
        # 隔离不同 case 的产物，避免串读。但 examples.run_with_llm 依赖仓库内包，
        # 需用 PYTHONPATH 指向 repo_dir，cwd 设为临时目录。
        workdir = Path(tempfile.mkdtemp(prefix="nwa_eval_"))
        original = self._write_config(workdir)
        try:
            cmd = [
                "python", "-m", "examples.run_with_llm",
                "--title", case.name,
                "--brief", build_nwa_brief(case),
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(self._repo_dir) + os.pathsep + env.get("PYTHONPATH", "")
            start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(workdir), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            elapsed = time.monotonic() - start
            if proc.returncode != 0:
                raise RuntimeError(f"NovelWritingAgent 退出码 {proc.returncode}：{stderr.decode('utf-8', 'replace')[-1000:]}")
            content = self._read_output(self._repo_dir, workdir)
            return GeneratedChapter(content=content, meta={
                "adapter": self.name,
                "model": self._model.model if self._model else None,
                "elapsed_seconds": round(elapsed, 3),
                "tokens": None,
            })
        finally:
            self._restore_config(original, workdir)
            shutil.rmtree(workdir, ignore_errors=True)
