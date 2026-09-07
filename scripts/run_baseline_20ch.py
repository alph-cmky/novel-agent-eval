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
PROMPT_VERSION = "baseline50-v4"  # v4: 《神仙也得走流程》抽象搞笑设定，替代严肃古风
STORY_OUTLINE = (
    "《神仙也得走流程》：苏迟考公三战上岸，岗位是「天庭驻人间办事处综合岗」，"
    "编制在天庭，办公地点在城中村临街铺面，被邻居当算命馆。办事处就她一个人，"
    "负责神仙下凡渡劫后的善后——神仙拍拍屁股回天庭，留下的因果、欠的债、"
    "炸的山、淹的城、谈的恋爱，全得她收尾。直属领导文曲星是天庭综合办公室副主任，"
    "被KPI折磨到发际线后退三百年，靠微信派活，消息永远在下班后发。"
    "工具箱每月限额：还魂丹三颗、忘忧符一张、因果剪刀一把（剪断因果但留毛边）、"
    "天庭介绍信（公章在天庭，得寄顺丰到付）。全书女本位、无感情线。"
    "搞笑来源是神仙当甲方、苏迟当乙方、天庭流程荒诞、工具永远不够用。"
    "叙事用具体事件、对话和闯祸后果往前推，少写设定说明和内心独白。"
)

# 逐章大纲：四卷（1-5 入职 / 6-10 神仙甲方 / 11-15 旧账 / 16-20 退休神仙）
CHAPTER_OUTLINES: dict[int, str] = {
    1: "苏迟考公上岸，报到地点是城中村铺面，前任留了张纸条「别接龙王单，别惹孟婆，别信月老」，人已经跑了。",
    2: "第一桩活：某龙王渡劫把一条街下水道炸了，居民以为是地震，苏迟得在天庭理赔和市政维修之间两头骗。",
    3: "文曲星微信发来《神仙渡劫善后操作手册》三百页，苏迟翻到第二页睡着，醒来发现还魂丹被老鼠偷了一颗。",
    4: "月老下凡渡情劫，走前给三个凡人牵了红线全是死结。苏迟拿因果剪刀剪，剪完三人变成同时恨上彼此，比死结还乱。",
    5: "苏迟给天庭写第一份周报，三百字写四小时，文曲星批回来「格式不对重写」，重写批回来「字体不对重写」。",
    6: "某散仙渡劫时在人间开了奶茶店，走后留三百张会员卡没核销，顾客天天堵门。苏迟得让凡人以为卡过期了。",
    7: "雷公渡劫劈歪了，把一户人家WiFi路由器劈成舍利子，能接收天庭信号。苏迟去换，但这户已靠天庭信号炒股赚了三百万。",
    8: "苏迟发现办事处隔壁快递站站长是退休土地公，两人开始互相借东西，她借还魂丹土地公借金元宝，账本越记越乱。",
    9: "某花神渡劫种的花海走后全变异，长出迷你版花神，会说话会骂人，集体罢工要求八小时工作制。苏迟得去谈判。",
    10: "文曲星来人间视察，苏迟紧张准备一周，结果文曲星只是来蹭WiFi——天庭网速太慢，他要在人间下完季度报告。",
    11: "三百年前旧案翻出来：某大仙渡劫借了人间一户三两银子说日后必还，三百年没还。苏迟得找到最近的血缘后人去还钱。",
    12: "孟婆汤被人间奶茶店老板偷了配方做成忘忧奶茶，喝了真能忘事。苏迟去维权，但天庭没有知识产权法。",
    13: "苏迟的因果剪刀被土地公借走，土地公用它剪掉了自己欠苏迟的所有人情。苏迟去要，土地公说扯平了。",
    14: "某剑仙渡劫时收的徒弟没师傅教，自己练成半套剑法只会劈不会收。苏迟去帮注销修仙资格，徒弟说已筑基不肯退。",
    15: "年终考核KPI是善后完成率，苏迟算了一下十二桩完成七桩三桩走流程两桩被投诉。文曲星说及格线百分之八十。",
    16: "苏迟发现城中村有一群退休神仙，渡完劫不想回天庭，在人间开小卖部修车铺早点摊，组了退休神仙协会会长是灶神。",
    17: "退休神仙协会和天庭起冲突：天庭要他们回去开渡劫经验交流会，退休神仙说都退休了还开什么会。苏迟两头挨骂。",
    18: "灶神请苏迟吃饭，饭桌上说你那个办事处其实是空编，天庭三百年没往这岗派过人，你是第一个。苏迟筷子掉了。",
    19: "苏迟查档案发现岗位三百年前设的，设岗原因是某大仙渡劫把人间搞太烂天庭被投诉。十七任平均任期十七年离职原因疯。",
    20: "苏迟把第十八任离职报告翻出来，上面就一行字「流程走完了我也走」。她给文曲星发微信「这活我接了但还魂丹得加量」文曲星秒回已读。",
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
        genre="女主无CP",
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
    pv_tag = PROMPT_VERSION.rsplit("-", 1)[-1]
    run_tag = (
        f"baseline{args.chapters}_{args.chapters}ch_s{args.sample}"
        f"_r{args.max_rounds}_{model_tag}{phase_tag}_{pv_tag}"
    )
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
        persist_dir=persist,
        resume=True,
        synthetic_context=args.synthetic_context,
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

    from novel_agent.observability.tracing import flush_tracing

    flush_tracing()


if __name__ == "__main__":
    asyncio.run(main())
