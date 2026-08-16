# scripts/run_external_benchmarks.py
"""一键并行运行三大开源评测集（LitBench, LongWriter-Bench, StoryBench）横评。

用法：
  STEPFUN_API_KEY=... BENCHMARK=litbench uv run python scripts/run_external_benchmarks.py
  STEPFUN_API_KEY=... BENCHMARK=longwriter uv run python scripts/run_external_benchmarks.py
  STEPFUN_API_KEY=... BENCHMARK=storybench uv run python scripts/run_external_benchmarks.py
  STEPFUN_API_KEY=... BENCHMARK=all uv run python scripts/run_external_benchmarks.py
"""
import argparse
import asyncio
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
from novel_agent_eval.dataset.loader import load_external_benchmark_cases
from novel_agent_eval.judge import Judge
from novel_agent_eval.report import render_json, render_scorecard
from novel_agent_eval.runner import BenchmarkReport, BenchmarkRunner

BENCHMARK_PATH = "novel_agent_eval/dataset/external/external_benchmarks.json"


async def main() -> None:
    parser = argparse.ArgumentParser(description="三大开源评测集并行横评工具")
    parser.add_argument("--benchmark", default=os.environ.get("BENCHMARK", "litbench"),
                        help="评测集名称: litbench, longwriter, storybench 或 all")
    parser.add_argument("--agents", default="novel_agent,vanilla_llm", help="被测 Agent 列表")
    parser.add_argument("--repeat", type=int, default=1, help="重复次数")
    parser.add_argument("--out", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    benchmarks = ["litbench", "longwriter", "storybench"] if args.benchmark.lower() == "all" else [args.benchmark.lower()]
    
    all_cases = []
    for b in benchmarks:
        cases = load_external_benchmark_cases(BENCHMARK_PATH, benchmark_type=b)
        all_cases.extend(cases)

    if not all_cases:
        print(f"错误：未加载到任何用例，请检查 benchmark 参数: {args.benchmark}", file=sys.stderr)
        sys.exit(1)

    from novel_agent_eval.agents.base import ModelConfig
    from novel_agent_eval.agents.inkos import InkOSAdapter
    from novel_agent_eval.agents.novel_writing import NovelWritingAgentAdapter
    from novel_agent_eval.agents.story_diffusion import StoryDiffusionAdapter

    stepfun_model = ModelConfig(
        base_url=BASE_URL,
        api_key=os.environ.get("STEPFUN_API_KEY", ""),
        model="step-3.7-flash",
    )

    agents = []
    for name in [a.strip().lower() for a in args.agents.split(",")]:
        if name in ("novel_agent", "novel"):
            agents.append(NovelAgentAdapter(evolution_enabled=True))
        elif name in ("vanilla_llm", "vanilla"):
            agents.append(VanillaLLMAdapter())
        elif name in ("inkos",):
            agents.append(InkOSAdapter(model=stepfun_model, timeout=1800.0))
        elif name in ("nwa", "novel_writing_agent"):
            nwa_repo = Path("/tmp/nwa/NovelWritingAgent-main")
            nwa_venv = Path("/tmp/nwa/venv")
            os.environ["PATH"] = f"{nwa_venv / 'bin'}:{os.environ.get('PATH', '')}"
            agents.append(NovelWritingAgentAdapter(repo_dir=nwa_repo, model=stepfun_model, timeout=1800.0))
        elif name in ("story_diffusion", "diffusion", "story"):
            agents.append(StoryDiffusionAdapter(model=stepfun_model))

    judge = Judge(n_samples=3)
    runner = BenchmarkRunner(judge=judge, repeat=args.repeat)

    print(f"=== 启动开源评测集横评: {', '.join(benchmarks)} (共 {len(all_cases)} 个用例) ===", flush=True)
    results = []
    for agent in agents:
        for case in all_cases:
            try:
                res = await runner.run_case(agent, case, args.repeat)
                results.append(res)
                dims_brief = ", ".join(f"{k}={v:.0f}" for k, v in res.dims_mean.items())
                print(f"[{res.agent}] {res.case} overall={res.overall_mean:.1f} | {dims_brief}", flush=True)
            except Exception as e:
                print(f"[{agent.name}] {case.name} FAILED: {e}", flush=True)

    report = BenchmarkReport(
        results=results,
        repeat=args.repeat,
        agents=[a.name for a in agents],
        cases=[c.name for c in all_cases],
    )
    
    out_file = args.out or f"/tmp/external_{args.benchmark}_eval.json"
    Path(out_file).write_text(render_json(report), encoding="utf-8")
    print(f"\n=== 结果已存 {out_file} ===\n", flush=True)
    print(render_scorecard(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
