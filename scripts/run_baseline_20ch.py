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
PROMPT_VERSION = "baseline50-v5"  # v5: 加密搞笑梗，发疯文学+玩梗+缺大德，参考长矛老师风格
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
    1: "苏迟考公上岸，报到地点是城中村铺面，门牌写「天庭驻人间办事处」被邻居当算命馆。前任留了张纸条「别接龙王单，别惹孟婆，别信月老，别问第十八任去哪了」，人已经跑了。她把纸条贴在门口当免责声明。",
    2: "第一桩活：某龙王渡劫把一条街下水道炸了，居民以为是地震，市政要追责。苏迟给市政报「地质塌陷」，给天庭报「龙王失职」，两边各拿一份报告，钱从谁口袋出她不管，反正不从她口袋出。",
    3: "文曲星下班后微信发来《神仙渡劫善后操作手册》三百页，要求「明早前读完写心得」。苏迟翻到第二页睡着，醒来发现还魂丹被老鼠偷了一颗。她给文曲星回「心得：手册可防身，老鼠识字」。",
    4: "月老下凡渡情劫，走前给三个凡人牵了红线全是死结。苏迟拿因果剪刀剪，剪完三人变成同时恨上彼此，比死结还乱。月老在天庭回微信「这叫三角恋，是艺术」，苏迟把聊天记录截图存档当免责证据。",
    5: "苏迟给天庭写第一份周报，三百字写四小时。文曲星批回来「格式不对重写」，重写批回来「字体不对重写」，再写批回来「行距不对重写」。她把周报改成文曲星发过的格式原样发回，批回来「你怎么抄我模板」。",
    6: "某散仙渡劫时在人间开了奶茶店，走后留三百张会员卡没核销，顾客天天堵门要退卡。苏迟贴告示「本店已渡劫飞升，会员卡随缘核销，缘尽作废」，被投诉到市监。她给市监看天庭介绍信，市监看不懂但被唬住了。",
    7: "雷公渡劫劈歪了，把一户人家WiFi路由器劈成舍利子，能接收天庭信号。苏迟去换路由器，但这户已靠天庭信号炒股赚了三百万，不肯换。她给文曲星请示，文曲星回「天庭信号属机密」，她回「那您来收」文曲星已读不回。",
    8: "苏迟发现办事处隔壁快递站站长是退休土地公，两人开始互相借东西。她借还魂丹土地公借金元宝，账本越记越乱。土地公说「咱俩这叫资源整合」，苏迟说「咱俩这叫监守自盗，但没人查」。",
    9: "某花神渡劫种的花海走后全变异，长出迷你版花神，会说话会骂人，集体罢工要求八小时工作制和五险一金。苏迟去谈判，迷你花神们让她填入职表，她反手给它们开离职证明，理由「花期已过」。",
    10: "文曲星来人间视察，苏迟紧张准备一周，擦窗拖地背汇报词。结果文曲星只是来蹭WiFi——天庭网速太慢，他要在人间下完季度报告。苏迟把准备好的汇报词塞他手里，他边下文件边签字，签完才发现签的是苏迟的加薪申请，已读不回。",
    11: "三百年前旧案翻出来：某大仙渡劫借了人间一户三两银子说日后必还，三百年没还，利息滚到能把那户买下来。苏迟得找到最近的血缘后人去还钱。后人是个外卖小哥，说「三两银子？我一天就赚回来，您留着吧」苏迟说「不行，天庭流程，您不收我得垫」。",
    12: "孟婆汤被人间奶茶店老板偷了配方做成忘忧奶茶，喝了真能忘事，生意爆好。苏迟去维权，但天庭没有知识产权法，只有「天庭秘方外泄处罚条例」罚的是泄密者不是偷方。苏迟给老板发律师函，老板回「你们连营业执照都没有」。",
    13: "苏迟的因果剪刀被土地公借走，土地公用它剪掉了自己欠苏迟的所有人情，把毛边也剪干净了。苏迟去要，土地公说「扯平了，你以前借我金元宝我也剪了」。苏迟说「您这是洗钱」土地公说「神仙不洗钱，神仙洗因果」。",
    14: "某剑仙渡劫时收的徒弟没师傅教，自己练成半套剑法只会劈不会收，天天劈邻居家的墙。苏迟去帮注销修仙资格，徒弟说已筑基不肯退，还反咬「你算老几，你有编制吗」。苏迟把天庭介绍信拍桌上，徒弟看不懂公章但被唬住，劈了最后一道墙才肯签字。",
    15: "年终考核KPI是善后完成率，苏迟算了一下十二桩完成七桩三桩走流程两桩被投诉一桩反咬她。文曲星说及格线百分之八十。苏迟说「那我差三个点」文曲星说「差三个点也是差，重做」。苏迟把两个走流程的结了案，把投诉的改成「已调解」，KPI刚好八十点一，文曲星已读不回。",
    16: "苏迟发现城中村有一群退休神仙，渡完劫不想回天庭，在人间开小卖部修车铺早点摊，组了退休神仙协会会长是灶神，副会长是门神（负责看门），财务是财神（但账永远对不上）。苏迟去登记，灶神说「我们退休了，不归你管」苏迟说「那你们也别归我善后」。",
    17: "退休神仙协会和天庭起冲突：天庭要他们回去开渡劫经验交流会，退休神仙说都退休了还开什么会，去也是睡觉。天庭派苏迟去通知，苏迟去了一趟回来说「他们说不去」，文曲星说「你再劝」，苏迟说「劝不动，要不您亲自来」，文曲星已读不回。",
    18: "灶神请苏迟吃饭，饭桌上说你那个办事处其实是空编，天庭三百年没往这岗派过人，你是第一个。苏迟筷子掉了。灶神说「前十七任都是临时工，档案在天庭，编制在天庭，人在人间，钱在虚空」苏迟问「那我呢」灶神说「你也是，但你考公考进来的，算工伤」。",
    19: "苏迟查档案发现岗位三百年前设的，设岗原因是某大仙渡劫把人间搞太烂天庭被投诉，设个岗背锅。十七任平均任期十七年，离职原因一栏全写「疯」。她翻到第十八任，任期半年，离职报告就一行字「流程走完了我也走」，附一张还魂丹空瓶照片。",
    20: "苏迟把第十八任离职报告复印一份贴在办事处照壁上，旁边贴自己写的「第十九任苏迟，在职，还魂丹已缺量」。她给文曲星发微信「这活我接了，但还魂丹得加量，邮费得报销，KPI得降，不然我也写那行字」文曲星秒回已读，三秒后又回「已阅，所提事项均不批准」。",
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
    phase_tag = "_s1"
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
        max_rounds=args.max_rounds,
        deterministic_gate_first=args.gate_first,
        label=f"na_s1_s{args.sample}",
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
