"""Run a reproducible ConStory prompt-pool evaluation.

Example:
  STEPFUN_API_KEY=... uv run python scripts/run_constory_benchmark.py \
    --limit 30 --language zh --repeat 4 --dry-run

Remove --dry-run only after checking the manifest and estimated API budget.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter
from novel_agent_eval.constory import ConStoryCheckerAdapter
from novel_agent_eval.dataset.open_benchmarks import load_constory_cases
from novel_agent_eval.judge import Judge
from novel_agent_eval.report import render_json, render_scorecard
from novel_agent_eval.runner import BenchmarkReport, BenchmarkRunner

BASE_URL = "https://api.stepfun.com/step_plan/v1"


def _agents(names: str):
    result = []
    for name in names.split(","):
        normalized = name.strip().lower()
        if normalized in {"novel_agent", "novel-agent", "novel"}:
            result.append(NovelAgentAdapter())
        elif normalized.startswith("novel_agent:r"):
            rounds = int(normalized.removeprefix("novel_agent:r"))
            result.append(
                NovelAgentAdapter(max_rounds=rounds, label=f"novel_agent_r{rounds}")
            )
        elif normalized in {"vanilla", "vanilla_llm"}:
            result.append(VanillaLLMAdapter())
        else:
            raise ValueError(f"unsupported agent: {name}")
    return result


async def _run(args: argparse.Namespace) -> None:
    cases = load_constory_cases(
        limit=args.limit,
        language=args.language,
        task_types=tuple(args.task_type or ("generation", "continuation")),
    )
    for case in cases:
        case.word_target = args.word_target
    if args.dry_run:
        print(f"cases={len(cases)} agents={args.agents.split(',')} repeat={args.repeat}")
        print("dry-run: no model calls made")
        return

    if not os.environ.get("STEPFUN_API_KEY"):
        print("STEPFUN_API_KEY 未设置", file=sys.stderr)
        raise SystemExit(1)
    os.environ.setdefault("STEPFUN_BASE_URL", BASE_URL)
    os.environ.setdefault("STEPFUN_JUDGE_MODEL", "step-3.7-flash")
    os.environ.setdefault("OPENAI_API_KEY", os.environ["STEPFUN_API_KEY"])
    os.environ.setdefault("OPENAI_BASE_URL", BASE_URL)
    os.environ.setdefault("QUALITY_MODEL", "step-3.7-flash")
    os.environ.setdefault("BUDGET_MODEL", "step-3.7-flash")
    os.environ.setdefault("QUALITY_IS_REASONING", "true")
    os.environ.setdefault("BUDGET_IS_REASONING", "true")
    os.environ.setdefault("BASELINE_API_KEY", os.environ["STEPFUN_API_KEY"])
    os.environ.setdefault("BASELINE_BASE_URL", BASE_URL)
    os.environ.setdefault("BASELINE_MODEL", "step-3.7-flash")

    agents = _agents(args.agents)
    print(f"cases={len(cases)} agents={[agent.name for agent in agents]} repeat={args.repeat}")
    runner = BenchmarkRunner(
        judge=Judge(n_samples=args.judge_samples),
        repeat=args.repeat,
        consistency_checker=ConStoryCheckerAdapter(),
    )
    results = []
    failures = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_job(agent, case):
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    runner.run_case(
                        agent,
                        case,
                        parallel_repeats=args.parallel_repeats,
                    ),
                    timeout=args.case_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - persist one-case failure and continue
                failure = {
                    "agent": agent.name,
                    "case": case.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(
                    f"{agent.name} {case.name} FAILED: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return None, failure
            print(f"{agent.name} {case.name} overall={result.overall_mean:.1f}", flush=True)
            return result, None

    jobs = [run_job(agent, case) for agent in agents for case in cases]
    job_results = await asyncio.gather(*jobs)
    for result, failure in job_results:
        if result is not None:
            results.append(result)
        if failure is not None:
            failures.append(failure)

    report = BenchmarkReport(
        results=results,
        repeat=args.repeat,
        agents=[agent.name for agent in agents],
        cases=[case.name for case in cases],
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    output.with_name(f"{output.stem}.failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(render_scorecard(report))
    print(f"结果已写入 {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--word-target", type=int, default=2000)
    parser.add_argument("--language", choices=["en", "zh"], default=None)
    parser.add_argument("--task-type", action="append", default=None)
    parser.add_argument("--agents", default="novel_agent,vanilla_llm")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--judge-samples", type=int, default=3)
    parser.add_argument("--case-timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--parallel-repeats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="traces/eval_results/constory_benchmark.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
