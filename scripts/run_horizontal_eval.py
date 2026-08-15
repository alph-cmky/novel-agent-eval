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

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter
from novel_agent_eval.constory import ConStoryCheckerAdapter
from novel_agent_eval.dataset.loader import load_cases
from novel_agent_eval.judge import Judge
from novel_agent_eval.report import render_json, render_scorecard
from novel_agent_eval.runner import BenchmarkReport, BenchmarkRunner


async def main() -> None:
    repeat = int(os.environ.get("REPEAT", "1"))
    cases = load_cases("novel_agent_eval/dataset/self_built")
    agents = [NovelAgentAdapter(evolution_enabled=True), VanillaLLMAdapter()]
    judge = Judge(n_samples=3)  # 中位数采样：抑制 reasoning 模型 consistency 维偶发极端分
    # ConStory 一致性检测器：evidence-grounded 找矛盾，覆盖 Judge 的 consistency 维
    # （Judge 原始分 + ConStory 错误数一并存入 CaseRun.meta 供对比）
    consistency_checker = ConStoryCheckerAdapter()
    runner = BenchmarkRunner(judge=judge, repeat=repeat, consistency_checker=consistency_checker)

    results = []
    failed = []
    for agent in agents:
        for case in cases:
            try:
                res = await runner.run_case(agent, case, repeat)
            except Exception as e:  # noqa: BLE001 — 单 case 偶发崩溃不中断整体横评
                failed.append(f"{agent.name}:{case.name}")
                print(
                    f"[{agent.name}] {case.name} FAILED: {type(e).__name__}: {e}",
                    flush=True,
                )
                continue
            results.append(res)
            dims_brief = ", ".join(
                f"{k}={v:.0f}" for k, v in res.dims_mean.items()
            )
            print(f"[{res.agent}] {res.case} overall={res.overall_mean:.1f} | {dims_brief}", flush=True)

    report = BenchmarkReport(
        results=results,
        repeat=repeat,
        agents=[a.name for a in agents],
        cases=[c.name for c in cases],
    )
    out = Path("/tmp/horizontal_eval.json")
    out.write_text(render_json(report), encoding="utf-8")
    print(f"\n=== 结果已存 {out} ===\n", flush=True)
    print(render_scorecard(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
