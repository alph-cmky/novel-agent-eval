# scripts/run_vanilla_50ch.py
"""Phase 5 对照：VanillaLLM 50 章基线（deepseek-v4-flash）。

与 NovelAgent 50 章（baseline50_50ch_s1_r{0,1}_deepseek-v4-flash_gate1）同协议：
同大纲 / 同 Judge（deepseek-v4-flash）/ 同 ConStory / 3000字/章 / 20 章质量指标口径。

VanillaLLM = 无 Agent 框架：单次 prompt 直接生成，无 Canon 记忆 / 无 Editor /
无 Evolution / 无 gate-first——架构对照的意义就是隔离出这些机制的价值。

memory_mode：
  none（默认）  纯大纲生成，无跨章记忆
  full         拼接全部前章全文（朴素长上下文基线）

用法：
  uv run python scripts/run_vanilla_50ch.py --chapters 50
  uv run python scripts/run_vanilla_50ch.py --chapters 20 --memory-mode full
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys

_env_path = pathlib.Path(__file__).resolve().parents[1].parent / "novel-agent" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ[_k] = _v.strip().strip('"').strip("'")
os.environ.setdefault("STEPFUN_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
os.environ.setdefault("STEPFUN_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))
os.environ.setdefault("STEPFUN_JUDGE_MODEL", os.environ.get("BUDGET_MODEL", "deepseek-v4-flash"))

VAULT_EVAL_DATA = pathlib.Path(
    "/Users/gaoyinrun/Documents/Obsidian Vault/novel-agent/eval-data"
)

# 与 run_baseline_20ch.py 保持同一份实验材料（大纲/大纲版本）
from run_baseline_20ch import CHAPTER_OUTLINES, STORY_OUTLINE  # noqa: E402


def _case(ch: int, sample: int, chapters: int):
    from novel_agent_eval.dataset.schema import EvalCase

    if ch <= max(1, round(chapters * 0.25)):
        stage = "opening"
    elif ch > chapters - round(chapters * 0.25):
        stage = "long"
    else:
        stage = "middle"
    return EvalCase(
        name=f"baseline{chapters}_s{sample}_ch{ch:02d}",
        stage=stage,
        story_outline=STORY_OUTLINE,
        previous_context="",
        target_chapter_outline=CHAPTER_OUTLINES[ch],
        word_target=3000,
        project_id=f"vanilla{chapters}_s{sample}",
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", type=int, default=50)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--memory-mode", choices=["none", "full"], default="none")
    parser.add_argument("--no-consistency", action="store_true", help="跳过 ConStory")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if not os.environ.get("BASELINE_API_KEY"):
        # 缺省复用 QUALITY_*（与 NovelAgent 同生成模型/凭证）
        os.environ["BASELINE_API_KEY"] = os.environ.get("QUALITY_API_KEY", "")
        os.environ["BASELINE_BASE_URL"] = os.environ.get("QUALITY_BASE_URL", "")
        os.environ.setdefault("BASELINE_MODEL", os.environ.get("QUALITY_MODEL", "deepseek-v4-flash"))

    from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter
    from novel_agent_eval.baseline import go_no_go_gates, run_baseline
    from novel_agent_eval.constory import ConStoryCheckerAdapter
    from novel_agent_eval.judge import Judge

    run_tag = (
        f"vanilla{args.chapters}_{args.chapters}ch_s{args.sample}"
        f"_{args.memory_mode}_{os.environ.get('BASELINE_MODEL', 'unknown').lower().replace('.', '-')}"
    )
    out_dir = VAULT_EVAL_DATA / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    persist = str(out_dir / "v2_state")
    checkpoint = out_dir / "run.checkpoint.json"

    adapter = VanillaLLMAdapter(
        persist_dir=persist,
        resume=True,
        memory_mode=args.memory_mode,
    )
    judge = Judge(n_samples=1)
    consistency = None if args.no_consistency else ConStoryCheckerAdapter()

    cases = [_case(ch, args.sample, args.chapters) for ch in range(1, args.chapters + 1)]
    result = await run_baseline(
        adapter=adapter, judge=judge, cases=cases,
        persist_dir=persist, checkpoint_path=checkpoint,
        consistency_checker=consistency, resume=True,
        chapter_timeout=args.timeout, sample_index=args.sample,
    )

    report = {
        "run_tag": run_tag,
        "adapter": "vanilla_llm",
        "memory_mode": args.memory_mode,
        "model": os.environ.get("BASELINE_MODEL"),
        "reliability": {
            "completed": result.completed_chapters,
            "expected": result.expected_chapters,
            "full_run_success": result.full_run_success,
            "chapter_completion_rate": result.chapter_completion_rate,
            "timeout_rate": result.timeout_rate,
            "invalid_rate": result.invalid_rate,
            "censorship_rate": result.censorship_rate,
        },
        "quality": {
            "mean": result.quality_mean, "std": result.quality_std,
            "first": result.first_window_score, "middle": result.middle_window_score,
            "last": result.last_window_score,
            "degradation": result.degradation, "trend_slope": result.trend_slope,
            "segments": result.segment_stats(window=5) if args.chapters == 20 else None,
        },
        "consistency": {
            "ced": result.ced,
            "first_density": result.first_error_density,
            "last_density": result.last_error_density,
            "error_growth_slope": result.error_growth_slope,
        },
        "cost": {
            "total_tokens_raw": result.total_tokens,
            "tokens_per_chapter_raw": result.tokens_per_chapter,
            "latency_per_chapter": result.latency_per_chapter,
        },
        "quality_per_cost": result.quality_per_cost,
        "chapters": [
            {
                "chapter": c.chapter_number, "status": c.status,
                "overall": c.overall, "dimensions": c.dimensions,
                "judge_status": c.judge_status,
                "consistency_errors": c.consistency_errors,
                "ground_truth": c.ground_truth,
                "tokens_raw": c.token_usage.get("total_tokens"),
                "latency_seconds": c.latency_seconds,
                "failure_stage": c.failure_stage, "failure_reason": c.failure_reason,
            }
            for c in result.chapters
        ],
        "go_no_go": go_no_go_gates([result]),
    }
    out_path = out_dir / "baseline_result.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {run_tag} ===", flush=True)
    print(f"completed={result.completed_chapters}/{result.expected_chapters} "
          f"quality_mean={result.quality_mean}±{result.quality_std} "
          f"degradation={result.degradation} ced={result.ced}", flush=True)
    print(f"tokens_raw_total={result.total_tokens} "
          f"latency/ch={result.latency_per_chapter}s", flush=True)
    print(f"go/no-go: {report['go_no_go']}", flush=True)
    print(f"saved → {out_path}", flush=True)

    from novel_agent.graph.chapter import aclose_checkpointers
    await aclose_checkpointers()


if __name__ == "__main__":
    asyncio.run(main())
