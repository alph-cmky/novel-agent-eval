# scripts/run_baseline_20ch.py
"""EVAL-BASELINE-20：正式 20 章 diagnostic baseline（耗真实 API，不进 pytest/CI）。

C-1 固定实验条件：EvaluationConfig 冻结协议 → build_eval_manifest 落盘。
断点续跑：resume=True + checkpoint（V2 durable state 为真相源）。
输出：manifest / per-chapter 结果 / 聚合指标（segments、growth slopes、
quality-per-cost、evolution gains）→ vault eval-data/。

用法：
  uv run python scripts/run_baseline_20ch.py --chapters 20 [--sample 1] [--no-consistency]
  uv run python scripts/run_baseline_20ch.py --chapters 3   # 试跑
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from novel_agent_eval.dataset.schema import EvalCase

_env_path = pathlib.Path(__file__).resolve().parents[1].parent / "novel-agent" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))
os.environ["STEPFUN_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["STEPFUN_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "")
os.environ.setdefault("STEPFUN_JUDGE_MODEL", "step-3.7-flash")

VAULT_EVAL_DATA = pathlib.Path(
    "/Users/gaoyinrun/Documents/Obsidian Vault/novel-agent/eval-data"
)

# ── 版本化的实验材料（C-1：prompt 变更必须 bump PROMPT_VERSION）──
PROMPT_VERSION = "baseline20-v1"
STORY_OUTLINE = (
    "《断剑重铸》设定：少年沈舟在剑冢捡到一柄断裂的古剑，剑中封印着上一代剑圣的残魂。"
    "为重铸断剑，他拜入天衡剑宗，从外门弟子起步。全书写他三年内集齐三味铸剑材料、"
    "在宗门大比上一鸣惊人，并揭开古剑与剑宗祖师的前世渊源。"
)

# 20 章逐章大纲（三幕：1-5 崭露头角 / 6-14 历练与暗流 / 15-20 大比与真相）
CHAPTER_OUTLINES: dict[int, str] = {
    1: "沈舟在剑冢捡到断剑，剑中残魂苏醒，警告他三日内须以血契认主，否则剑灵消散。",
    2: "沈舟拜入天衡剑宗外门，因断剑被同门嘲笑；夜里残魂教他第一式「引星」，初窥剑气。",
    3: "外门考核，沈舟以「引星」一招破敌，却被内门长老看出断剑来历，引来盘问。",
    4: "沈舟被迫立下三月后与内门弟子比武的生死状；残魂透露断剑需三味材料重铸。",
    5: "沈舟入后山秘境采药，初遇女主角苏晚晴，二人联手脱困，结下善缘。",
    6: "比武将至，对手暗中购来克剑的软筋散；沈舟将计就计，当众反杀局。",
    7: "生死比武，沈舟险胜；断剑异动引护宗长老注意，长老欲收缴断剑。",
    8: "沈舟以血契秘辛保住断剑，被罚守藏经阁；阁中发现祖师手札残页。",
    9: "手札残页记载第一味材料「星髓铁」下落：城外坠星涧。沈舟夜间出山。",
    10: "坠星涧遇矿匪与妖兽双重夹击，苏晚晴率巡查队驰援，二人再携手。",
    11: "取得星髓铁；回山途中撞见黑袍人盗掘祖师遗物，黑袍人身上有剑宗失传剑意。",
    12: "黑袍人夜袭外门，掳走沈舟同窗；残魂认出黑袍人是当年叛出师门的祖师首徒。",
    13: "沈舟追击黑袍人，初闻「祖师之死另有隐情」；断剑残魂开始隐瞒部分记忆。",
    14: "第二味材料「地心火莲」在火域秘境；沈舟闭关突破，残魂传授完整三式剑诀。",
    15: "火域取莲，遭各派争夺；沈舟首露锋芒，名动一方，也彻底暴露在幕后势力眼中。",
    16: "宗门大比开幕；沈舟连败强敌，苏晚晴身份揭晓——宗主之女。",
    17: "大比决赛，沈舟对上内门第一天才；断剑之力失控，残魂强行接管。",
    18: "残魂失控引祖师封印共鸣，黑袍人现身夺剑；沈舟以本心唤回残魂，合力退敌。",
    19: "真相揭开：祖师为封印魔渊自断剑心，首徒盗剑实为复活师尊；三味材料只差最后一味。",
    20: "第三味材料线索指向魔渊深处；沈舟接下祖师传承，立誓重铸断剑、了结千年恩怨。",
}


def _case(ch: int, sample: int) -> "EvalCase":
    from novel_agent_eval.dataset.schema import EvalCase

    return EvalCase(
        name=f"baseline20_s{sample}_ch{ch:02d}",
        stage="opening" if ch <= 5 else ("middle" if ch <= 14 else "long"),
        story_outline=STORY_OUTLINE,
        previous_context="",  # 留空：前文一律由 V2 DB Canon 经 ContextCompiler 提供
        target_chapter_outline=CHAPTER_OUTLINES[ch],
        word_target=3000,
        project_id=f"baseline20_s{sample}",
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", type=int, default=20)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--no-consistency", action="store_true", help="跳过 ConStory（省 API）")
    parser.add_argument("--max-rounds", type=int, default=0, help="evolution 预算（0=无重写）")
    parser.add_argument("--timeout", type=float, default=900.0, help="单章超时秒")
    args = parser.parse_args()

    if not os.environ.get("STEPFUN_API_KEY"):
        print("STEPFUN_API_KEY 未加载", file=sys.stderr)
        sys.exit(1)

    from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
    from novel_agent_eval.baseline import (
        BaselineRunResult,
        go_no_go_gates,
        run_baseline,
    )
    from novel_agent_eval.constory import ConStoryCheckerAdapter
    from novel_agent_eval.evaluation_config import EvaluationConfig
    from novel_agent_eval.judge import Judge
    from novel_agent_eval.manifest import build_eval_manifest

    cfg = EvaluationConfig(
        model=os.environ.get("QUALITY_MODEL", "step-3.7-flash"),
        judge_model=os.environ.get("STEPFUN_JUDGE_MODEL", "step-3.7-flash"),
        prompt_version=PROMPT_VERSION,
        chapter_count=args.chapters,
        repeat=1,
        max_rounds=args.max_rounds,
        scene_first=False,  # B-1 parity 默认
        deterministic_gate_first=False,
        timeout=args.timeout,
        resume=True,
    )

    run_tag = f"baseline20_{args.chapters}ch_s{args.sample}_r{args.max_rounds}"
    out_dir = VAULT_EVAL_DATA / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # C-1：协议冻结 manifest（commits / dataset hash / config）
    prompt_file = out_dir / "prompts.json"
    prompt_file.write_text(
        json.dumps({"story_outline": STORY_OUTLINE, "chapter_outlines": CHAPTER_OUTLINES},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = build_eval_manifest(cfg, prompt_file)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[baseline20] manifest → {out_dir / 'manifest.json'}", flush=True)

    # 稳定 persist_dir：跨调用 resume 依赖 V2 DB（Project by name）存活，
    # 必须落在输出目录而非临时目录——否则每次重启都会整段重跑。
    persist = str(out_dir / "v2_state")
    checkpoint = out_dir / "run.checkpoint.json"

    adapter = NovelAgentAdapter(
        max_rounds=args.max_rounds, persist_dir=persist,
        resume=True, label=f"na_s{args.sample}",
    )
    judge = Judge(n_samples=1)
    consistency = None if args.no_consistency else ConStoryCheckerAdapter()

    cases = [_case(ch, args.sample) for ch in range(1, args.chapters + 1)]
    result: BaselineRunResult = await run_baseline(
        adapter=adapter, judge=judge, cases=cases,
        persist_dir=persist, checkpoint_path=checkpoint,
        consistency_checker=consistency, resume=True, chapter_timeout=args.timeout,
        sample_index=args.sample,
    )

    # ── 聚合输出 ──
    report = {
        "run_tag": run_tag,
        "config": cfg.to_manifest_config(),
        "manifest": manifest,
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
            "segments": result.segment_stats(window=5),
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
            "cost_growth_slope_raw": result.cost_growth_slope,
            "note": "tokens 为 provider 原始报告（stepfun suspect，见 Token Truth Report），仅供相对比较",
        },
        "context": {
            "context_growth_slope_chars": result.context_growth_slope,
            "per_chapter": [
                {"chapter": c.chapter_number, "context_sizes": c.context_sizes,
                 "canon_counts": c.canon_counts}
                for c in result.chapters
            ],
        },
        "evolution": {
            "gain_per_revision": result.evolution_gain_per_revision(),
            "paths": [
                {"chapter": c.chapter_number, "path": c.evolution_path}
                for c in result.chapters if c.evolution_path
            ],
        },
        "quality_per_cost": result.quality_per_cost,
        "chapters": [
            {
                "chapter": c.chapter_number, "status": c.status,
                "overall": c.overall, "dimensions": c.dimensions,
                "consistency_errors": c.consistency_errors,
                "consistency_failed_categories": c.consistency_failed_categories,
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
    (out_dir / "run.log").write_text(
        json.dumps(report["reliability"], ensure_ascii=False), encoding="utf-8"
    )

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
