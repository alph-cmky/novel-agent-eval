# scripts/smoke_opponents.py
"""Task 13 真实冒烟：InkOS + NovelWritingAgent 各跑一次（一次性脚本，不进 pytest/CI）。

用法：
  STEPFUN_API_KEY=... STEPFUN_BASE_URL=... uv run python scripts/smoke_opponents.py

模型 env 是 STEPFUN_*（brief 指定 Judge/生成用 StepFun）。若 StepFun key 无效可
换任意 OpenAI 兼容端点，例如：STEPFUN_BASE_URL=https://api.deepseek.com/v1 \
  STEPFUN_API_KEY=$DEEPSEEK_API_KEY STEPFUN_MODEL=deepseek-chat。

- InkOS：init → book create --target-chapters 1 → draft（short run 不支持单章）；
  需要 PATH 上有 inkos（npm i -g @actalk/inkos --ignore-scripts）。
- NwA：repo 缺则 curl codeload tarball 解压到 /tmp/nwa，venv 缺则创建并 pip 装依赖
  （PyPI 走清华镜像）。跑完把 /tmp/nwa/venv/bin 前置到 PATH 供适配器用。
  真实流水线（canon+draft+review+revision）单次远超 600s，timeout 需给大。
"""
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

from novel_agent_eval.agents.base import ModelConfig
from novel_agent_eval.agents.inkos import InkOSAdapter
from novel_agent_eval.agents.novel_writing import NovelWritingAgentAdapter
from novel_agent_eval.dataset.schema import EvalCase

NWA_REPO_DIR = Path("/tmp/nwa/NovelWritingAgent-main")
NWA_TARBALL = "/tmp/nwa/nwa.tar.gz"
NWA_VENV = Path("/tmp/nwa/venv")
NWA_TARBALL_URL = "https://codeload.github.com/xindaaW/NovelWritingAgent/tar.gz/refs/heads/main"


def _stepfun_model() -> ModelConfig:
    return ModelConfig(
        base_url=os.environ.get("STEPFUN_BASE_URL", "https://api.stepfun.ai/v1"),
        api_key=os.environ.get("STEPFUN_API_KEY", ""),
        model=os.environ.get("STEPFUN_MODEL", "step-3.7-flash"),
    )


def _case() -> EvalCase:
    return EvalCase(
        name="smoke_opponents",
        stage="opening",
        genre="玄幻",
        story_outline="少年穿越到修仙世界，立志问鼎剑道之巅。",
        previous_context="",
        target_chapter_outline="主角拜入青云剑派，得师父传剑，初窥剑气。",
        word_target=800,
    )


def _setup_nwa() -> None:
    """下载解压 NwA 仓库并装 venv（幂等，已存在则跳过）。"""
    if not (NWA_REPO_DIR / "examples" / "run_with_llm.py").exists():
        print(f"[nwa] 下载 {NWA_TARBALL_URL}")
        subprocess.run(
            ["curl", "-L", "--connect-timeout", "30", "-m", "300", NWA_TARBALL_URL, "-o", NWA_TARBALL],
            check=True,
        )
        subprocess.run(["tar", "xzf", NWA_TARBALL, "-C", "/tmp/nwa"], check=True)
        print(f"[nwa] 解压到 {NWA_REPO_DIR}")
    if not (NWA_VENV / "bin" / "python").exists():
        print("[nwa] 创建 venv")
        subprocess.run([sys.executable, "-m", "venv", str(NWA_VENV)], check=True)
        print("[nwa] pip 安装依赖（清华镜像）")
        subprocess.run(
            [
                str(NWA_VENV / "bin" / "pip"), "install", "--quiet",
                "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple",
                "-r", str(NWA_REPO_DIR / "requirements.txt"),
            ],
            check=True,
        )
    print(f"[nwa] 就绪：{NWA_REPO_DIR} / {NWA_VENV}")


async def _smoke_inkos() -> dict:
    model = _stepfun_model()
    if not model.api_key:
        return {"ok": False, "why": "STEPFUN_API_KEY 未设置"}
    # 冒烟环境已全局装 inkos（npm i -g @actalk/inkos --ignore-scripts），
    # 适配器 `shutil.which("inkos")` 命中后直接走 inkos 命令，不走 npx。
    if shutil.which("inkos") is None:
        return {"ok": False, "why": "inkos 未安装（PATH 无 inkos，npx 因 postinstall 挂住不可用）"}
    adapter = InkOSAdapter(model=model, timeout=1800.0)
    try:
        gen = await adapter.generate(_case())
        ok = len(gen.content) >= 400  # 目标 800 字的一半
        print(f"[inkos] content 长度={len(gen.content)} meta={gen.meta}")
        print(f"[inkos] 前120字：{gen.content[:120]}")
        return {"ok": ok, "why": None, "len": len(gen.content)}
    except Exception as e:  # noqa: BLE001 — 冒烟脚本如实上报
        print(f"[inkos] 失败：{type(e).__name__}: {e}")
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}


async def _smoke_nwa() -> dict:
    model = _stepfun_model()
    if not model.api_key:
        return {"ok": False, "why": "STEPFUN_API_KEY 未设置"}
    if not (NWA_REPO_DIR / "examples" / "run_with_llm.py").exists():
        return {"ok": False, "why": f"NwA 仓库缺失：{NWA_REPO_DIR}"}
    # 让适配器的 `python` 解析到 NwA venv（装好了依赖）
    os.environ["PATH"] = f"{NWA_VENV / 'bin'}:{os.environ.get('PATH', '')}"
    # 真实流水线单次 >600s（实测 10 分钟未走完），冒烟给足 1800s
    adapter = NovelWritingAgentAdapter(repo_dir=NWA_REPO_DIR, model=model, timeout=1800.0)
    try:
        gen = await adapter.generate(_case())
        ok = bool(gen.content.strip())
        print(f"[nwa] content 长度={len(gen.content)} meta={gen.meta}")
        print(f"[nwa] 前120字：{gen.content[:120]}")
        return {"ok": ok, "why": None, "len": len(gen.content)}
    except Exception as e:  # noqa: BLE001
        print(f"[nwa] 失败：{type(e).__name__}: {e}")
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}


async def main() -> None:
    setup = os.environ.get("SMOKE_SETUP", "1") != "0"
    if setup:
        try:
            _setup_nwa()
        except Exception as e:  # noqa: BLE001
            print(f"[nwa] 环境准备失败：{type(e).__name__}: {e}")

    results = {}
    for name, coro in [("inkos", _smoke_inkos()), ("nwa", _smoke_nwa())]:
        results[name] = await coro

    print("\n=== 冒烟汇总 ===")
    for name, r in results.items():
        status = "OK" if r["ok"] else "FAIL"
        why = f"（{r['why']}）" if r.get("why") else ""
        print(f"  {name}: {status}{why}")
    if not all(r["ok"] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
