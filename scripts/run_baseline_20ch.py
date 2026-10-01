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
PROMPT_VERSION = "baseline50-v6"  # v6: 同一情节，改成两个人各要一件事，留一件没说完
STORY_OUTLINE = (
    "《神仙也得走流程》：苏迟考公三战上岸，岗位是「天庭驻人间办事处综合岗」，"
    "编制在天庭，办公地点在城中村临街铺面，门口招牌被前任改成「苏半仙算命起名风水堪舆」，"
    "她撕了三次撕不掉，索性在下面加一行小字「不算命，只善后，神仙欠的债我替它还」。"
    "办事处就她一个人，负责神仙下凡渡劫后的善后——神仙拍拍屁股回天庭，"
    "留下的因果、欠的债、炸的山、淹的城、谈的恋爱，全得她收尾。"
    "直属领导文曲星是天庭综合办公室副主任，被KPI折磨到发际线后退三百年，"
    "靠微信派活，消息永远在下班后发，已读不回是常态。"
    "工具箱每月限额：还魂丹三颗、忘忧符一张、因果剪刀一把（剪断因果但留毛边，天庭不保修）、"
    "天庭介绍信（公章在天庭，得寄顺丰到付，邮费苏迟垫，报销得走流程，流程走完邮费票已过期）。"
    "苏迟一身反骨，面对天庭流程和神仙甲方彻底发疯：能糊弄就糊弄，能反咬就反咬，"
    "能拖到下个考核周期绝不本周交。全书女本位、无感情线、无单箭头修罗场。"
    "搞笑来源是神仙当甲方提离谱需求、苏迟当乙方发疯应对、天庭流程荒诞到缺大德、"
    "工具永远不够用还得自己贴邮费。叙事用具体事件、对话和闯祸后果往前推，"
    "少写设定说明和内心独白，金句要嘴贱不要抒情。"
)

# 逐章大纲：四卷（1-5 入职发疯 / 6-10 神仙甲方缺大德 / 11-15 旧账反咬 / 16-20 退休神仙摆烂）
CHAPTER_OUTLINES: dict[int, str] = {
    1: "邻居认定铺面是算命馆，要苏迟给她看一段姻缘。苏迟要贴前任留下的免责声明「别接龙王单，别惹孟婆，别信月老，别问第十八任去哪了」。邻居追问第十八任是谁，苏迟把纸条拍在门上，这句话没答。",
    2: "市政的人要一份地震报告好追责到人。苏迟要把炸下水道写成龙王失职，让天庭出钱。她给市政报「地质塌陷」，给天庭报「龙王失职」。钱从谁口袋出，她当着市政的面没说。",
    3: "文曲星要苏迟明早交完三百页《神仙渡劫善后操作手册》的心得。苏迟要他先补被老鼠偷走的那颗还魂丹。他只催心得。她回「心得：手册可防身，老鼠识字」，丹药的事他已读不回。",
    4: "三个凡人互相指责苏迟把红线剪成了仇。月老要她承认这是艺术。苏迟要他把「这叫三角恋，是艺术」写成书面当免责。月老不肯落字，只在微信里丢下这一句。谁先恨上谁，三人都没说清。",
    5: "文曲星要周报按格式、字体、行距重写三遍。苏迟要他指出到底哪一条算错，否则按他发过的模板原样发回。他批「你怎么抄我模板」。他自己的模板为什么也不算数，没解释。",
    6: "会员堵门要退卡，市监要苏迟交出经营者。苏迟要市监承认店主已经飞升、卡随缘作废，并掏出天庭介绍信。市监看不懂但被唬住了。店主人在哪，双方都没问完。",
    7: "户主要留着被劈成舍利子的路由器继续炒股。苏迟要换回普通路由器。文曲星说天庭信号属机密，让她去收；她要文曲星自己来。户主赚的三百万算谁的，没人认。文曲星已读不回。",
    8: "土地公要借还魂丹，说这叫资源整合。苏迟要他把金元宝的账写清楚，否则就是监守自盗。两人越借越乱。谁先欠谁，账本上没写。",
    9: "迷你花神要苏迟填入职表，给八小时工作制和五险一金。苏迟要它们先停工离开花海，反手开离职证明，理由「花期已过」。编制谁批，没谈拢。",
    10: "文曲星只要蹭人间 WiFi 下完季度报告，不许苏迟打扰。苏迟要把准备好的汇报塞给他签字。他边下文件边签，签完才发现是加薪申请，已读不回。签字时知不知道那是加薪，他没说。",
    11: "外卖小哥要苏迟别把三两银子的三百年利息算到他头上，说一天就赚回来，让她留着。苏迟要他签字收下，否则天庭流程得她垫。他不肯收。这笔钱算赠与还是坏账，两人没定。",
    12: "奶茶店老板要苏迟别查忘忧配方，生意正爆。苏迟要他承认偷了孟婆的方子。条例只罚泄密者，不罚偷方的人。老板回「你们连营业执照都没有」。方子是谁漏出去的，他装不知道。",
    13: "土地公要苏迟承认人情已经用因果剪刀剪平。苏迟要他把剪刀还回来，说他这是洗因果。土地公说神仙不洗钱，神仙洗因果。剪刀毛边剪没剪干净，他岔开了。",
    14: "徒弟要苏迟承认自己没编制，凭什么注销他的修仙资格。苏迟要他在刚劈开的墙前面签字注销，把天庭介绍信拍在桌上。他看不懂公章，劈完最后一道墙才肯签。筑基到哪一步，签字前没承认。",
    15: "文曲星要苏迟重做，差三个点也是差，及格线百分之八十。苏迟要把两个走流程的结案、投诉改成已调解，凑到八十点一交差。哪两桩其实没做完，她没报。文曲星已读不回。",
    16: "灶神要苏迟别管退休神仙协会，说他们开小卖部、修车铺、早点摊，退休了不归办事处。苏迟要协会登记，否则这些摊子还得她善后。财神的账为什么对不上，灶神没讲。",
    17: "文曲星要苏迟把退休神仙劝回去开渡劫经验交流会。灶神要她传话：都退休了，去也是睡觉。苏迟两头传。文曲星说你再劝，她说劝不动、要不您亲自来。会到底开不开，三人都没定。文曲星已读不回。",
    18: "灶神请苏迟吃饭，要她吃完别问编制。苏迟要问空编和第十八任。灶神说天庭三百年没往这岗派过人，前十七任是临时工，档案在天庭，编制在天庭，人在人间，钱在虚空，她考公考进来算工伤。她再问第十八任，他用夹菜岔开。人在哪，饭桌上没说。",
    19: "灶神要苏迟把档案合上。苏迟要他把十七任离职原因和第十八任那页读出来：设岗是为了背锅，离职原因一栏全是「疯」，第十八任任期半年，报告只有「流程走完了我也走」，附一张还魂丹空瓶照片。他读了「疯」，读到空瓶就停。第十八任去哪了，还是没说。",
    20: "文曲星要这章只走已阅，什么都不批。苏迟把第十八任的离职报告贴上照壁，旁边贴「第十九任苏迟，在职，还魂丹已缺量」，并要他加还魂丹、报邮费、降 KPI，否则她也写那行字。文曲星秒回已读，三秒后又回「已阅，所提事项均不批准」。照壁上那两张纸他看没看见，没提。",
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
        "--max-rounds", type=int, default=0, help="legacy 标签字段（S1 不驱动编排；仅写入 manifest/run_tag）"
    )
    parser.add_argument(
        "--v0-gate",
        type=float,
        default=78.0,
        help="v0 门控阈值（>=此分跳过重写，<0 禁用）",
    )
    parser.add_argument(
        "--scene-first", action="store_true", help="已排除：S1 拒绝 scene-first（传入将导致 adapter 报错）"
    )
    parser.add_argument(
        "--gate-first",
        dest="gate_first",
        action="store_true",
        default=True,
        help="legacy 标签：S1 主仓库已内建 Hard Gate 条件化审查；此 flag 只影响 run_tag/manifest",
    )
    parser.add_argument(
        "--no-gate-first",
        dest="gate_first",
        action="store_false",
        help="legacy 标签对照（不关闭主仓库 S1 gate 行为）",
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
    phase_tag = "_discourse"
    if args.scene_first:
        phase_tag += "_scene1"
    if args.gate_first:
        phase_tag += "_gate1"
    if args.synthetic_context:
        phase_tag += "_synctx"
    pv_tag = PROMPT_VERSION.rsplit("-", 1)[-1]
    suffix = os.environ.get("EVAL_TAG_SUFFIX", "")
    run_tag = (
        f"baseline{args.chapters}_{args.chapters}ch_s{args.sample}"
        f"_r{args.max_rounds}_{model_tag}{phase_tag}_{pv_tag}{suffix}"
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
        max_rounds=args.max_rounds,
        deterministic_gate_first=args.gate_first,
        label=f"na_discourse_s{args.sample}",
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
