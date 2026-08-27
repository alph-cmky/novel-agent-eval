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
from novel_agent_eval.evaluation_config import EvaluationConfig
from novel_agent_eval.judge import Judge
from novel_agent_eval.manifest import build_eval_manifest
from novel_agent_eval.report import render_json, render_scorecard
from novel_agent_eval.runner import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    CaseRun,
)


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
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "4")),
                        help="跨 Agent/case 的最大并发数")
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

    # Phase 0.3：冻结评测协议，写入 manifest（凭证只经 env，不进 manifest）
    eval_config = EvaluationConfig(
        model="step-3.7-flash",
        judge_model=os.environ.get("STEPFUN_JUDGE_MODEL", "step-3.7-flash"),
        prompt_version="self_built_v1",
        chapter_count=len(cases),
        repeat=args.repeat,
        max_rounds=2,
        deterministic_gate_first=True,
        memory_protocol="structured_narrative_state",
        scene_first=True,
        context_mode="bounded_memory",
        timeout=float(os.environ["CHAPTER_TIMEOUT"]) if os.environ.get("CHAPTER_TIMEOUT") else None,
        resume=args.resume,
    )
    dataset_dir = Path(args.dataset)
    prompt_path = next(iter(dataset_dir.glob("*.json")), dataset_dir)
    manifest = build_eval_manifest(eval_config, prompt_path)
    manifest_path = Path(args.out).with_name(Path(args.out).name + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[manifest] 协议已冻结 → {manifest_path}", flush=True)

    # 断点续跑缓存读取
    results = []
    completed_keys = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        try:
            cached_data = json.loads(out_path.read_text(encoding="utf-8"))
            for item in cached_data.get("results", []):
                completed_keys.add((item.get("agent"), item.get("case")))
            results.extend(_result_from_json(item) for item in cached_data.get("results", []))
            print(f"[断点续跑] 已加载 {len(completed_keys)} 个已完成的用例记录", flush=True)
        except (OSError, json.JSONDecodeError):
            pass

    failed = []
    semaphore = asyncio.Semaphore(max(args.concurrency, 1))
    jobs = [
        (agent, case) for agent in agents for case in cases
        if (agent.name, case.name) not in completed_keys
    ]

    async def run_one(agent, case):
        async with semaphore:
            try:
                res = await runner.run_case(agent, case, args.repeat)
                print(f"[{res.agent}] {res.case} overall={res.overall_mean:.1f}", flush=True)
                return res, None
            except Exception as e:  # noqa: BLE001 - one failed case must not abort the suite
                failure = {
                    "agent": agent.name,
                    "case": case.name,
                    "error_type": type(e).__name__,
                    "error": str(e) or repr(e),
                }
                print(f"[{agent.name}] {case.name} FAILED: {failure['error']}", flush=True)
                return None, failure

    for job in asyncio.as_completed([run_one(agent, case) for agent, case in jobs]):
        result, failure = await job
        if result is not None:
            results.append(result)
        if failure is not None:
            failed.append(failure)
        _write_checkpoint(out_path, results, failed, args.repeat, agents, cases)

    report = BenchmarkReport(
        results=results,
        repeat=args.repeat,
        agents=[a.name for a in agents],
        cases=[c.name for c in cases],
    )
    _write_checkpoint(out_path, results, failed, args.repeat, agents, cases)
    print(f"\n=== 结果已存 {out_path} ===\n", flush=True)
    print(render_scorecard(report), flush=True)


def _write_checkpoint(path, results, failures, repeat, agents, cases):
    payload = json.loads(render_json(BenchmarkReport(
        results=results, repeat=repeat,
        agents=[a.name for a in agents], cases=[c.name for c in cases],
    )))
    payload["failures"] = failures
    temporary = Path(path).with_name(f".{Path(path).name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _result_from_json(item):
    return BenchmarkResult(
        agent=item["agent"],
        case=item["case"],
        stage=item["stage"],
        repeat=item["repeat"],
        dims_mean=item.get("dims_mean", {}),
        dims_std=item.get("dims_std", {}),
        overall_mean=item.get("overall_mean", 0.0),
        overall_std=item.get("overall_std", 0.0),
        runs=[CaseRun(**run) for run in item.get("runs", [])],
    )


if __name__ == "__main__":
    asyncio.run(main())
