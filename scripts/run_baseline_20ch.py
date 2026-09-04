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
        os.environ[_k] = _v.strip().strip('"').strip("'")
os.environ.setdefault("STEPFUN_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
os.environ.setdefault("STEPFUN_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))
# Judge/ConStory 与生成模型同 provider：默认跟随 BUDGET_MODEL（E10 也允许 env 显式覆盖）
os.environ.setdefault(
    "STEPFUN_JUDGE_MODEL", os.environ.get("BUDGET_MODEL", "step-3.7-flash")
)

VAULT_EVAL_DATA = pathlib.Path(
    "/Users/gaoyinrun/Documents/Obsidian Vault/novel-agent/eval-data"
)

# ── 版本化的实验材料（C-1：prompt 变更必须 bump PROMPT_VERSION）──
PROMPT_VERSION = (
    "baseline50-v1"  # v1: 50章长篇验证（8dim editor + gate78 + gate-first 默认）
)
STORY_OUTLINE = (
    "《断剑重铸》设定：少年沈舟在剑冢捡到一柄断裂的古剑，剑中封印着上一代剑圣的半魂残识。"
    "为重铸断剑，他拜入天衡剑宗，从外门弟子起步，集齐星髓铁、地心火莲、玄冥铁三味铸剑材料。"
    "过程中揭开古剑与剑宗祖师的前世渊源：祖师为封印魔渊自断剑心，以魔渊核心铸成人形「活印」苏晚晴为锁。"
    "沈舟深入魔渊寻玄冥铁，遭魔君残识阻挠，与黑袍人首徒解开心结，最终重铸断剑、将魔渊永镇剑中，了结千年恩怨。"
)

# 逐章大纲：三卷（1-20 崭露与历练 / 21-38 魔渊深入与真相 / 39-50 重铸与决战）
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
    21: "沈舟深入魔渊外围，残魂感应到第三味材料「玄冥铁」的气息；魔渊瘴气侵蚀经脉，他以剑气护体强撑。",
    22: "魔渊入口遇苏晚晴率巡查队拦阻；她奉宗主之命押沈舟回宗，二人争执间遭遇魔物突袭。",
    23: "联手退魔物后，苏晚晴暗中转交一枚可避瘴气的「避魔珠」，暗示宗主对魔渊另有安排。",
    24: "沈舟潜入魔渊第一层，见无数封魔石碑；残魂触碑生异变，忆起当年祖师封印之战片段。",
    25: "碑林深处藏有祖师手刻剑痕，残魂指点沈舟参悟，习得专门克制魔气的「镇魔剑意」。",
    26: "魔渊守将发现入侵者，率魔兵围剿；沈舟凭镇魔剑意杀出重围，向第二层坠落。",
    27: "第二层是废弃的铸剑古墟，遍地断剑残骸；沈舟惊觉此地正是当年祖师铸剑之处。",
    28: "古墟核心藏半部铸剑秘录，记载「玄冥铁」需以持剑人精血浇铸方可驯服；沈舟心神震动。",
    29: "黑袍人现身古墟，揭露当年真相：祖师铸剑为镇魔，却在最后一步遭魔君残识反噬、剑心俱裂。",
    30: "黑袍人坦言盗剑非为复活师尊，而是防止断剑落入魔君之手；二人暂时休战，约定合力寻玄冥铁。",
    31: "魔渊第三层现玄冥铁矿脉，矿脉受魔君残识守护，采铁必先破其护矿魔阵。",
    32: "破阵之法需祖师镇魔剑意配合血契；黑袍人护法，沈舟以残魂为引，开始强行采炼玄冥铁。",
    33: "采炼至半，魔君残识借矿脉反扑，黑袍人重伤；残魂被迫现身硬撼魔君残识，境界不稳。",
    34: "危急关头苏晚晴赶到，她掌心浮现魔纹，竟能驱散魔君残识——她的血脉与魔渊本源相连。",
    35: "苏晚晴救下众人，却因动用魔纹遭反噬昏厥；沈舟从秘录残页推断：她是祖师当年以魔渊核心封印铸成的人形「活印」。",
    36: "带苏晚晴回宗求医，宗主却避而不见；藏经阁老阁主道出：苏晚晴实为祖师以自身剑心血肉孕养的封印载体。",
    37: "老阁主授沈舟「唤心诀」，可唤醒苏晚晴沉睡的自我意识，让她摆脱「活印」的宿命。",
    38: "沈舟日日以唤心诀助苏晚晴，二人情愫渐深；魔渊封印却因玄冥铁被采而松动，魔气外泄。",
    39: "宗主被迫现身，坦言当年祖师以命铸印、以苏晚晴为印锁，如今印锁将崩，要么牺牲苏晚晴重封，要么彻底封印魔渊。",
    40: "沈舟抉择：他拒绝牺牲苏晚晴，决定以重铸后的断剑为全新封印核心，将魔渊永镇剑中。",
    41: "重铸玄冥铁需天火，沈舟携玄冥铁再赴火域秘境；残魂传授最后的心法「剑我合一」。",
    42: "火域深处，沈舟以血契精血浇铸玄冥铁，三味材料初聚；断剑剑胚成型，残魂却因耗损过度开始消散。",
    43: "残魂弥留之际道出：自己正是祖师当年分出的半魂，守护断剑千年只为等今日之选；沈舟嚎啕大恸。",
    44: "黑袍人携祖师遗骨前来，以秘法稳住残魂不散；真相大白：首徒一生执念是还清当年未能护住师尊的愧。",
    45: "魔君残识趁断剑未成，引魔渊大军倾巢而出；天衡剑宗与各派联军死守魔渊谷口。",
    46: "大战惨烈，宗主以命封堵第一波魔潮；苏晚晴血脉彻底觉醒，化身「活印」现形与魔君残识对峙。",
    47: "沈舟抱断剑剑胚冲入战阵，唤心诀唤醒苏晚晴自我，她自愿将印锁之力渡入断剑，助其完成最后一步。",
    48: "断剑重铸功成，沈舟以「剑我合一」驾驭新剑，魔君残识被封入剑中；千年恩怨在此一剑了结。",
    49: "大战终歇，魔渊封印重塑；黑袍人卸下执念，将祖师遗骨葬回剑冢，转身云游四方。",
    50: "沈舟与苏晚晴携手重铸剑宗，断剑镇魔渊永世；剑冢残碑上，沈舟亲手刻下祖师与首徒之名。",
}


def _case(ch: int, sample: int, chapters: int) -> "EvalCase":
    from novel_agent_eval.dataset.schema import EvalCase

    # stage 三档：opening 前 25% / long 后 25% / 其余 middle
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
        previous_context="",  # 留空：前文一律由 V2 DB Canon 经 ContextCompiler 提供
        target_chapter_outline=CHAPTER_OUTLINES[ch],
        word_target=3000,
        project_id=f"baseline{chapters}_s{sample}",
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", type=int, default=20)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument(
        "--no-consistency", action="store_true", help="跳过 ConStory（省 API）"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=0, help="evolution 预算（0=无重写）"
    )
    parser.add_argument(
        "--v0-gate",
        type=float,
        default=78.0,
        help="v0 门控阈值（>=此分跳过重写，<0 禁用）",
    )
    parser.add_argument(
        "--scene-first", action="store_true", help="Phase4-A: scene_first=true"
    )
    parser.add_argument(
        "--gate-first",
        dest="gate_first",
        action="store_true",
        default=True,
        help="Phase4-B: deterministic_gate_first=true（默认开启，Phase4 证实无质量损失且省 59%% token）",
    )
    parser.add_argument(
        "--no-gate-first",
        dest="gate_first",
        action="store_false",
        help="关闭 deterministic_gate_first（对照用）",
    )
    parser.add_argument(
        "--synthetic-context",
        action="store_true",
        help="Phase4-C: 仅 previous_context，无结构化叙事状态",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="单章超时秒")
    args = parser.parse_args()

    if not os.environ.get("STEPFUN_API_KEY"):
        print("STEPFUN_API_KEY 未加载", file=sys.stderr)
        sys.exit(1)

    from novel_agent.observability.tracing import require_tracing_config

    tracing_err = require_tracing_config(strict=False)
    if tracing_err:
        print(f"[tracing] {tracing_err}；本跑次使用 NullHandle", flush=True)

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
        scene_first=args.scene_first,
        deterministic_gate_first=args.gate_first,
        timeout=args.timeout,
        resume=True,
    )

    # run_tag 带生成模型标签 + Phase4 消融标记：不同 run 不互相覆盖
    gen_model = os.environ.get("QUALITY_MODEL", "unknown")
    model_tag = "".join(c if c.isalnum() else "-" for c in gen_model.lower())
    phase_tag = ""
    if args.scene_first:
        phase_tag += "_scene1"
    if args.gate_first:
        phase_tag += "_gate1"
    if args.synthetic_context:
        phase_tag += "_synctx"
    run_tag = f"baseline{args.chapters}_{args.chapters}ch_s{args.sample}_r{args.max_rounds}_{model_tag}{phase_tag}"
    out_dir = VAULT_EVAL_DATA / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # C-1：协议冻结 manifest（commits / dataset hash / config）
    prompt_file = out_dir / "prompts.json"
    prompt_file.write_text(
        json.dumps(
            {"story_outline": STORY_OUTLINE, "chapter_outlines": CHAPTER_OUTLINES},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = build_eval_manifest(cfg, prompt_file)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[baseline{args.chapters}] manifest → {out_dir / 'manifest.json'}", flush=True
    )

    # 稳定 persist_dir：跨调用 resume 依赖 V2 DB（Project by name）存活，
    # 必须落在输出目录而非临时目录——否则每次重启都会整段重跑。
    persist = str(out_dir / "v2_state")
    checkpoint = out_dir / "run.checkpoint.json"

    adapter = NovelAgentAdapter(
        max_rounds=args.max_rounds,
        persist_dir=persist,
        v0_gate_score=args.v0_gate if args.max_rounds > 0 else None,
        scene_first=args.scene_first,
        deterministic_gate_first=args.gate_first,
        synthetic_context=args.synthetic_context,
        resume=True,
        label=f"na_s{args.sample}",
    )
    judge = Judge(n_samples=1)
    consistency = None if args.no_consistency else ConStoryCheckerAdapter()

    cases = [
        _case(ch, args.sample, args.chapters) for ch in range(1, args.chapters + 1)
    ]
    result: BaselineRunResult = await run_baseline(
        adapter=adapter,
        judge=judge,
        cases=cases,
        persist_dir=persist,
        checkpoint_path=checkpoint,
        consistency_checker=consistency,
        resume=True,
        chapter_timeout=args.timeout,
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
            "mean": result.quality_mean,
            "std": result.quality_std,
            "first": result.first_window_score,
            "middle": result.middle_window_score,
            "last": result.last_window_score,
            "degradation": result.degradation,
            "trend_slope": result.trend_slope,
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
                {
                    "chapter": c.chapter_number,
                    "context_sizes": c.context_sizes,
                    "canon_counts": c.canon_counts,
                }
                for c in result.chapters
            ],
        },
        "evolution": {
            "gain_per_revision": result.evolution_gain_per_revision(),
            "paths": [
                {"chapter": c.chapter_number, "path": c.evolution_path}
                for c in result.chapters
                if c.evolution_path
            ],
        },
        "quality_per_cost": result.quality_per_cost,
        "chapters": [
            {
                "chapter": c.chapter_number,
                "status": c.status,
                "overall": c.overall,
                "dimensions": c.dimensions,
                "judge_status": c.judge_status,
                "consistency_errors": c.consistency_errors,
                "consistency_failed_categories": c.consistency_failed_categories,
                "ground_truth": c.ground_truth,
                "tokens_raw": c.token_usage.get("total_tokens"),
                "latency_seconds": c.latency_seconds,
                "failure_stage": c.failure_stage,
                "failure_reason": c.failure_reason,
            }
            for c in result.chapters
        ],
        "go_no_go": go_no_go_gates([result]),
    }
    out_path = out_dir / "baseline_result.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "run.log").write_text(
        json.dumps(report["reliability"], ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== {run_tag} ===", flush=True)
    print(
        f"completed={result.completed_chapters}/{result.expected_chapters} "
        f"quality_mean={result.quality_mean}±{result.quality_std} "
        f"degradation={result.degradation} ced={result.ced}",
        flush=True,
    )
    print(
        f"tokens_raw_total={result.total_tokens} "
        f"latency/ch={result.latency_per_chapter}s",
        flush=True,
    )
    print(f"go/no-go: {report['go_no_go']}", flush=True)
    print(f"saved → {out_path}", flush=True)
    from novel_agent_eval.human_review import export_run_chapters

    n_md = export_run_chapters(out_dir)
    print(f"chapters md → {out_dir / 'chapters'} ({n_md})", flush=True)

    from novel_agent.graph.chapter import aclose_checkpointers
    from novel_agent.observability.tracing import flush_tracing

    flush_tracing()
    await aclose_checkpointers()


if __name__ == "__main__":
    asyncio.run(main())
