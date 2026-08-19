# scripts/run_horizontal_eval.py
"""横评：novel-agent vs vanilla_llm 用 StepFun 跑单章质量对比（一次性脚本，不进 pytest/CI）。

用法：
  STEPFUN_API_KEY=... REPEAT=1 uv run python scripts/run_horizontal_eval.py

- novel-agent 走主仓库 ModelRouter（读 QUALITY/BUDGET_MODEL + OPENAI_*）
- vanilla_llm 读 BASELINE_*
- Judge 读 STEPFUN_*（默认 step-3.7-flash）
三者统一指向 StepFun 的 step-3.7-flash，且都注入 reasoning_effort=low。
"""
import asyncio
import json
import os
import sys
from pathlib import Path

BASE_URL = "https://api.stepfun.com/step_plan/v1"

if not os.environ.get("STEPFUN_API_KEY"):
    print("STEPFUN_API_KEY 未设置", file=sys.stderr)
    sys.exit(1)

# ── env：统一走 StepFun（正确 base_url），各 adapter 读各自的 env ──
os.environ["STEPFUN_BASE_URL"] = BASE_URL
os.environ["STEPFUN_JUDGE_MODEL"] = "step-3.7-flash"
# 主仓库 ModelRouter（novel-agent）
os.environ["OPENAI_API_KEY"] = os.environ["STEPFUN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = BASE_URL
os.environ["QUALITY_MODEL"] = "step-3.7-flash"
os.environ["BUDGET_MODEL"] = "step-3.7-flash"
# step-3.7-flash 是 reasoning 模型：显式声明 is_reasoning，主仓库 build_chat_model
# 才会注入 reasoning_effort=low + 禁用 stream_chunk_timeout，否则推理挤空正文/长
# thinking 触发 StreamChunkTimeoutError。
os.environ["QUALITY_IS_REASONING"] = "true"
os.environ["BUDGET_IS_REASONING"] = "true"
# vanilla 基线
os.environ["BASELINE_API_KEY"] = os.environ["STEPFUN_API_KEY"]
os.environ["BASELINE_BASE_URL"] = BASE_URL
os.environ["BASELINE_MODEL"] = "step-3.7-flash"

import argparse

from novel_agent_eval.agents.base import ModelConfig
from novel_agent_eval.agents.inkos import InkOSAdapter
from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.agents.novel_writing import NovelWritingAgentAdapter
from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter
from novel_agent_eval.constory import ConStoryCheckerAdapter
from novel_agent_eval.dataset.loader import load_cases
from novel_agent_eval.judge import Judge
from novel_agent_eval.report import render_json, render_scorecard
from novel_agent_eval.runner import BenchmarkReport, BenchmarkRunner


def _build_agent_list(agent_names: list[str]) -> list:
    agents = []
    stepfun_model = ModelConfig(
        base_url=BASE_URL,
        api_key=os.environ.get("STEPFUN_API_KEY", ""),
        model="step-3.7-flash",
    )
    for name in agent_names:
        name = name.strip().lower()
        if name in ("novel_agent", "novel-agent", "novel"):
            agents.append(NovelAgentAdapter(max_rounds=2))
        elif name in ("vanilla_llm", "vanilla"):
            agents.append(VanillaLLMAdapter())
        elif name in ("inkos",):
            agents.append(InkOSAdapter(model=stepfun_model, timeout=1800.0))
        elif name in ("nwa", "novel_writing_agent"):
            nwa_root = Path(os.environ.get("NWA_ROOT", "/tmp/nwa"))
            nwa_repo = Path(os.environ.get("NWA_REPO_DIR", nwa_root / "NovelWritingAgent-main"))
            nwa_venv = Path(os.environ.get("NWA_VENV_DIR", nwa_root / "venv"))
            os.environ["PATH"] = f"{nwa_venv / 'bin'}:{os.environ.get('PATH', '')}"
            agents.append(NovelWritingAgentAdapter(repo_dir=nwa_repo, model=stepfun_model, timeout=1800.0))
        else:
            print(f"警告：未知的 Agent 名称 '{name}'，已跳过")
    return agents


async def main() -> None:
    parser = argparse.ArgumentParser(description="小说 Agent 横向对比评测工具")
    parser.add_argument("--agents", default=os.environ.get("AGENTS", "novel_agent,vanilla_llm"),
                        help="逗号分隔的被测 Agent 列表，如 novel_agent,vanilla_llm,inkos,nwa")
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", "1")),
                        help="每个用例的独立采样重复次数（默认 1）")
    parser.add_argument("--dataset", default="novel_agent_eval/dataset/self_built",
                        help="测试集路径")
    parser.add_argument("--out", default="/tmp/horizontal_eval.json",
                        help="评测结果 JSON 输出路径")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="启用断点续跑（跳过 /tmp/horizontal_eval.json 中已完成的用例）")
    args = parser.parse_args()

    agent_names = [a.strip() for a in args.agents.split(",") if a.strip()]
    agents = _build_agent_list(agent_names)
    if not agents:
        print("错误：未指定任何有效的被测 Agent", file=sys.stderr)
        sys.exit(1)

    cases = load_cases(args.dataset)
    judge = Judge(n_samples=3)
    consistency_checker = ConStoryCheckerAdapter()
    runner = BenchmarkRunner(judge=judge, repeat=args.repeat, consistency_checker=consistency_checker)

    # 断点续跑缓存读取
    results = []
    completed_keys = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        try:
            cached_data = json.loads(out_path.read_text(encoding="utf-8"))
            for item in cached_data.get("results", []):
                completed_keys.add((item.get("agent"), item.get("case")))
            print(f"[断点续跑] 已加载 {len(completed_keys)} 个已完成的用例记录", flush=True)
        except (OSError, json.JSONDecodeError):
            pass

    failed = []
    for agent in agents:
        for case in cases:
            if (agent.name, case.name) in completed_keys:
                print(f"[{agent.name}] {case.name} 已在缓存中，跳过", flush=True)
                continue

            try:
                res = await runner.run_case(agent, case, args.repeat)
            except Exception as e:  # noqa: BLE001 - one failed case must not abort the suite
                failed.append(f"{agent.name}:{case.name}")
                print(f"[{agent.name}] {case.name} FAILED: {type(e).__name__}: {e}", flush=True)
                continue

            results.append(res)
            dims_brief = ", ".join(f"{k}={v:.0f}" for k, v in res.dims_mean.items())
            print(f"[{res.agent}] {res.case} overall={res.overall_mean:.1f} | {dims_brief}", flush=True)

    report = BenchmarkReport(
        results=results,
        repeat=args.repeat,
        agents=[a.name for a in agents],
        cases=[c.name for c in cases],
    )
    out_path.write_text(render_json(report), encoding="utf-8")
    print(f"\n=== 结果已存 {out_path} ===\n", flush=True)
    print(render_scorecard(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
