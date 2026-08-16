# scripts/run_parallel_all.py
"""一键分片并发横评调度器：将自建用例按阶段、开源集按类型分片并行，全量横评提效 3-5 倍。

用法：
  STEPFUN_API_KEY=... DEEPSEEK_API_KEY=... uv run python scripts/run_parallel_all.py
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

os.environ["STEPFUN_BASE_URL"] = BASE_URL
os.environ["STEPFUN_JUDGE_MODEL"] = "step-3.7-flash"
os.environ["OPENAI_API_KEY"] = os.environ["STEPFUN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = BASE_URL
os.environ["QUALITY_MODEL"] = "step-3.7-flash"
os.environ["BUDGET_MODEL"] = "step-3.7-flash"
os.environ["QUALITY_IS_REASONING"] = "true"
os.environ["BUDGET_IS_REASONING"] = "true"
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


async def _run_worker(agent, cases: list, repeat: int, worker_name: str) -> list:
    judge = Judge(n_samples=3)
    consistency_checker = ConStoryCheckerAdapter()
    runner = BenchmarkRunner(judge=judge, repeat=repeat, consistency_checker=consistency_checker)
    results = []
    print(f"[{worker_name}] 启动分片 Worker，处理 {len(cases)} 个用例...", flush=True)
    for c in cases:
        try:
            res = await runner.run_case(agent, c, repeat)
            results.append(res)
            dims_brief = ", ".join(f"{k}={v:.0f}" for k, v in res.dims_mean.items())
            print(f"[{worker_name}] [{res.agent}] {res.case} overall={res.overall_mean:.1f} | {dims_brief}", flush=True)
        except Exception as e:
            print(f"[{worker_name}] [{agent.name}] {c.name} FAILED: {e}", flush=True)
    return results


async def main() -> None:
    repeat = int(os.environ.get("REPEAT", "1"))
    all_cases = load_cases("novel_agent_eval/dataset/self_built")

    # 1. 按阶段拆分用例分片
    opening_cases = [c for c in all_cases if c.stage == "opening"]
    middle_cases = [c for c in all_cases if c.stage == "middle"]
    long_cases = [c for c in all_cases if c.stage == "long"]

    novel_agent = NovelAgentAdapter(evolution_enabled=True)
    vanilla_agent = VanillaLLMAdapter()

    print(f"=== 启动全量分片高并发横评 (共 {len(all_cases)} 个用例, repeat={repeat}) ===", flush=True)

    # 2. 构建 6 个并行分片 Worker (novel_agent 3组 + vanilla_llm 3组)
    tasks = [
        _run_worker(novel_agent, opening_cases, repeat, "Worker-Novel-Opening"),
        _run_worker(novel_agent, middle_cases, repeat, "Worker-Novel-Middle"),
        _run_worker(novel_agent, long_cases, repeat, "Worker-Novel-Long"),
        _run_worker(vanilla_agent, opening_cases, repeat, "Worker-Vanilla-Opening"),
        _run_worker(vanilla_agent, middle_cases, repeat, "Worker-Vanilla-Middle"),
        _run_worker(vanilla_agent, long_cases, repeat, "Worker-Vanilla-Long"),
    ]

    worker_results = await asyncio.gather(*tasks)

    # 3. 合并所有分片结果
    all_results = []
    for wr in worker_results:
        all_results.extend(wr)

    report = BenchmarkReport(
        results=all_results,
        repeat=repeat,
        agents=["novel_agent", "vanilla_llm"],
        cases=[c.name for c in all_cases],
    )

    out_file = Path("/tmp/parallel_horizontal_eval.json")
    out_file.write_text(render_json(report), encoding="utf-8")
    print(f"\n=== 全量分片并发横评完成！结果已存 {out_file} ===\n", flush=True)
    print(render_scorecard(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
