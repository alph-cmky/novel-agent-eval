# scripts/run_eqbench_longform.py
"""EQ-Bench Longform 小规模横评：2-3 prompts × novel-agent vs vanilla（一次性脚本，不进 pytest/CI）。

用法：
  DEEPSEEK_API_KEY=... STEPFUN_API_KEY=... N_PROMPTS=2 uv run python scripts/run_eqbench_longform.py

三套 LLM 完全解耦：
  - judge ：DEEPSEEK_*（deepseek-v4-pro，14 维评分，thinking disabled）
  - bridge ：BRIDGE_*（独立 LLM，缺省同 DeepSeek，可覆盖为其它模型跑 planning）
  - 被测   ：novel-agent 走主仓库 ModelRouter（QUALITY/BUDGET_MODEL + OPENAI_*）；
            vanilla 走 BASELINE_*。默认均指向 StepFun step-3.7-flash（延续现有横评）。

流程：每 prompt 先 bridge.plan() 一次（两个 agent 共享同一 plan），再对每个 agent
run_longform 逐章生成 + 14 维评分，聚合 0-100 总分 + degradation（尾-首章分差），
最后打印 degradation 报告表并落盘 /tmp/eqbench_longform.json。

成本预期（N_PROMPTS=2，8 章）：bridge 5 步 planning ×2 + 生成 32 章 + judge 32 次，
novel-agent 进化流水线单章较慢，整体约数十分钟。链路验证建议先 N_PROMPTS=1。
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from statistics import fmean, pstdev

STEPFUN_BASE_URL = "https://api.stepfun.com/step_plan/v1"

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("DEEPSEEK_API_KEY 未设置（judge 用）", file=sys.stderr)
    sys.exit(1)
if not os.environ.get("STEPFUN_API_KEY"):
    print("STEPFUN_API_KEY 未设置（被测 agent 用）", file=sys.stderr)
    sys.exit(1)

# ── judge：DeepSeek（14 维，thinking disabled 在 EQBenchJudge 内部处理） ──
os.environ.setdefault("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-pro")
# bridge 缺省也走 DeepSeek，但 BRIDGE_* 可覆盖为其它「独立 LLM」
os.environ.setdefault("BRIDGE_MODEL", "deepseek-v4-pro")

# ── 被测 agent：novel-agent 走主仓库 ModelRouter，vanilla 走 BASELINE_*（StepFun） ──
os.environ["OPENAI_API_KEY"] = os.environ["STEPFUN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = STEPFUN_BASE_URL
os.environ["QUALITY_MODEL"] = "step-3.7-flash"
os.environ["BUDGET_MODEL"] = "step-3.7-flash"
# step-3.7-flash 是 reasoning 模型：显式声明 is_reasoning，主仓库 build_chat_model
# 才会注入 reasoning_effort=low，否则推理挤空正文/长 thinking 触发 StreamChunkTimeoutError。
os.environ["QUALITY_IS_REASONING"] = "true"
os.environ["BUDGET_IS_REASONING"] = "true"
os.environ["BASELINE_API_KEY"] = os.environ["STEPFUN_API_KEY"]
os.environ["BASELINE_BASE_URL"] = STEPFUN_BASE_URL
os.environ["BASELINE_MODEL"] = "step-3.7-flash"

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter
from novel_agent_eval.eqbench_bridge import EQBenchBridge
from novel_agent_eval.eqbench_judge import EQBenchJudge
from novel_agent_eval.longform import render_longform_table, run_longform
from novel_agent_eval.manifest import build_run_manifest

_PROMPTS_PATH = Path("novel_agent_eval/dataset/eqbench/prompts.json")


def load_prompts(n: int) -> list[dict]:
    data = json.loads(_PROMPTS_PATH.read_text(encoding="utf-8"))
    # prompts.json 为 {"1": {...}, "2": {...}, ...}，按 key 排序取前 n 个
    return [data[k] for k in sorted(data, key=int)[:n]]


def _serialize_result(result) -> dict:
    return {
        "agent": result.agent,
        "prompt_id": result.prompt_id,
        "sample_index": result.sample_index,
        "title": result.title,
        "mean_score": result.mean_score,
        "eqbench_0_100": result.eqbench_0_100,
        "degradation": result.degradation,
        "valid_chapters": result.valid_chapters,
        "completion_rate": result.completion_rate,
        "first_window_score": result.first_window_score,
        "middle_window_score": result.middle_window_score,
        "last_window_score": result.last_window_score,
        "trend_slope": result.trend_slope,
        "chapter_scores": result.chapter_scores,
        "chapters": [
            {
                "chapter_index": chapter.chapter_index,
                "content_hash": chapter.content_hash,
                "content_length": chapter.content_length,
                "eqbench_score": chapter.eqbench_score,
                "meta": chapter.meta,
            }
            for chapter in result.chapters
        ],
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write a checkpoint without leaving a truncated JSON file on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def main() -> None:
    n_prompts = int(os.environ.get("N_PROMPTS", "12"))
    repeat = int(os.environ.get("REPEAT", "2"))
    n_samples = int(os.environ.get("JUDGE_N_SAMPLES", "1"))
    concurrency = int(os.environ.get("CONCURRENCY", "2"))
    story_timeout = float(os.environ.get("STORY_TIMEOUT", "1200"))
    novel_max_rounds = int(os.environ.get("NOVEL_MAX_ROUNDS", "2"))
    novel_skip_orchestrator = os.environ.get("NOVEL_SKIP_ORCHESTRATOR", "1") == "1"
    novel_skip_reviews = os.environ.get("NOVEL_SKIP_REVIEWS", "0") == "1"
    novel_skip_worldbuilding = os.environ.get("NOVEL_SKIP_WORLDBUILDING", "0") == "1"
    novel_review_interval = int(os.environ.get("NOVEL_REVIEW_INTERVAL", "2"))
    novel_skip_enrichment = os.environ.get("NOVEL_SKIP_ENRICHMENT", "1") == "1"
    max_story_outline_chars = int(os.environ.get("MAX_STORY_OUTLINE_CHARS", "12000"))
    target_pid = os.environ.get("PROMPT_INDEX", None)  # 1-indexed: "1" 或 "2"

    all_prompts = load_prompts(n_prompts)
    if target_pid:
        idx = int(target_pid) - 1
        prompts = [(idx + 1, all_prompts[idx])]
    else:
        prompts = list(enumerate(all_prompts, start=1))

    bridge = EQBenchBridge()
    judge = EQBenchJudge(n_samples=n_samples)
    selected_agents = {
        "novel_agent": NovelAgentAdapter(
            max_rounds=novel_max_rounds,
            skip_orchestrator=novel_skip_orchestrator,
            skip_reviews=novel_skip_reviews,
            skip_worldbuilding=novel_skip_worldbuilding,
            review_interval=novel_review_interval,
            skip_evolution_enrichment=novel_skip_enrichment,
        ),
        "vanilla_llm": VanillaLLMAdapter(),
    }
    agent_names = [name.strip() for name in os.environ.get(
        "AGENTS", "novel_agent,vanilla_llm"
    ).split(",") if name.strip()]
    agents = [selected_agents[name] for name in agent_names]
    semaphore = asyncio.Semaphore(concurrency)
    out = Path(os.environ.get("EQBENCH_OUT", "/tmp/eqbench_longform.json"))
    progress_out = out.with_name(f"{out.stem}.progress.json")
    partial_out = out.with_name(f"{out.stem}.partial_results.json")
    failures_out = out.with_name(f"{out.stem}.failures.json")

    config_payload = {
        "n_prompts": n_prompts,
        "repeat": repeat,
        "concurrency": concurrency,
        "judge_n_samples": n_samples,
        "story_timeout": story_timeout,
        "novel_max_rounds": novel_max_rounds,
        "novel_skip_orchestrator": novel_skip_orchestrator,
        "novel_skip_reviews": novel_skip_reviews,
        "novel_skip_worldbuilding": novel_skip_worldbuilding,
        "novel_review_interval": novel_review_interval,
        "novel_skip_enrichment": novel_skip_enrichment,
        "max_story_outline_chars": max_story_outline_chars,
        "agents": agent_names,
    }
    manifest = build_run_manifest(config_payload, _PROMPTS_PATH)

    results = []
    failures = []

    def persist_partial() -> None:
        _write_json_atomic(
            partial_out,
            {
                "config": config_payload,
                "manifest": manifest,
                "results": [_serialize_result(r) for r in results],
            },
        )
        _write_json_atomic(
            failures_out,
            {"config": config_payload, "manifest": manifest, "failures": failures},
        )
        # Keep the legacy combined checkpoint for callers that already consume it.
        _write_json_atomic(
            progress_out,
            {
                "config": config_payload,
                "manifest": manifest,
                "failures": failures,
                "completed_results": [_serialize_result(r) for r in results],
            },
        )

    # Create usable empty checkpoints before the first network request.
    persist_partial()

    for pid, prompt in prompts:
        writing_prompt = prompt["writing_prompt"]
        title = prompt["title"]
        # 同一 prompt 的 plan 只跑一次 bridge，两个 agent 共享（省一半 planning 调用）
        try:
            plan = await bridge.plan(
                writing_prompt, prompt_id=str(pid), title=title, category=prompt["category"]
            )
        except Exception as e:  # noqa: BLE001 — planning 失败不中断其它 prompt
            failures.append(
                {
                    "stage": "planning",
                    "prompt_id": str(pid),
                    "title": title,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            print(f"[plan] {title} FAILED: {type(e).__name__}: {e}", flush=True)
            persist_partial()
            continue
        print(
            f"[plan] {title}: final_plan={len(plan.final_plan)} chars, "
            f"characters={len(plan.character_profiles)} chars",
            flush=True,
        )
        async def run_one(agent, sample_index: int, prompt_id: int, prompt_title: str, run_plan):
            async with semaphore:
                try:
                    res = await asyncio.wait_for(
                        run_longform(
                            agent=agent,
                            judge=judge,
                            plan=run_plan,
                            sample_index=sample_index,
                            max_story_outline_chars=max_story_outline_chars,
                        ),
                        timeout=story_timeout,
                    )
                except Exception as e:  # noqa: BLE001 — 单样本崩溃不中断整体横评
                    failure = {
                        "agent": agent.name,
                        "prompt_id": str(prompt_id),
                        "sample_index": sample_index,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                    print(
                        f"[{agent.name}] {prompt_title} sample={sample_index} FAILED: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    return None, failure
                per_ch = "/".join(
                    f"{s:.0f}" if s is not None else "NA"
                    for s in res.chapter_scores
                )
                if res.eqbench_0_100 is None:
                    print(
                        f"[{res.agent}] {res.title} sample={sample_index} INVALID "
                        f"degradation={res.degradation} | {per_ch}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{res.agent}] {res.title} sample={sample_index} "
                        f"0-100={res.eqbench_0_100:.1f} "
                        f"degradation={res.degradation:+.2f} | {per_ch}",
                        flush=True,
                    )
                return res, None

        jobs = [
            run_one(agent, sample_index, pid, title, plan)
            for agent in agents
            for sample_index in range(repeat)
        ]
        for completed in asyncio.as_completed(jobs):
            result, failure = await completed
            if result is not None:
                results.append(result)
            if failure is not None:
                failures.append(failure)
            persist_partial()

    print("\n=== EQ-Bench Longform degradation 报告 ===\n")
    print(render_longform_table(results))

    by_agent = {}
    for result in results:
        by_agent.setdefault(result.agent, []).append(result)
    summary = {}
    for agent_name, agent_results in by_agent.items():
        valid_results = [r for r in agent_results if r.eqbench_0_100 is not None]
        scores = [r.eqbench_0_100 for r in valid_results]
        degradations = [r.degradation for r in valid_results if r.degradation is not None]
        summary[agent_name] = {
            "samples": len(agent_results),
            "valid_samples": len(valid_results),
            "invalid_samples": len(agent_results) - len(valid_results),
            "mean_score": round(fmean(scores), 3) if scores else None,
            "score_std": round(pstdev(scores), 3) if len(scores) > 1 else None,
            "mean_degradation": round(fmean(degradations), 3) if degradations else None,
            "degradation_std": round(pstdev(degradations), 3)
            if len(degradations) > 1 else 0.0,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    _write_json_atomic(
        out,
        {
            "config": config_payload,
            "manifest": manifest,
            "failures": failures,
            "summary": summary,
            "results": [_serialize_result(r) for r in results],
        },
    )
    print(f"\n结果已存 {out}，部分结果已存 {partial_out}，失败记录已存 {failures_out}")


if __name__ == "__main__":
    asyncio.run(main())
