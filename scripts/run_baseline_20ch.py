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
    "baseline50-v3"  # v3: 《无祀录》章纲改场面化，压公文腔
)
STORY_OUTLINE = (
    "《无祀录》：北冕靠「记名入祀」收灵，除了名就像从人间蒸发。太常寺除籍司专干这脏活，"
    "除籍官三戒——不得结契、不得收徒、不得立私人香火；新官第一天必须办一桩「干净案子」证明无私。"
    "全书女本位、无感情线、无单箭头修罗场。好看的是女官怎么把衙门捅漏、女匠怎么把活人炼成灯，"
    "不是谁能得到女主。叙事用场面、对话和闯祸后果往前推，少写公文节录和程序说明。"
    "裴照晚上任就被砸上方绣「擅造私祀」案，限当日结案。她不结，当众把改期批文贴上照壁，"
    "从此成了太常寺最会闯祸的主事。她查到边地无香营：白日编户，夜里用女匠生辰炼假祀灯供边墙。"
    "营里有人跪求别把自己写回去，有人要名，有人已死。她不代人勾选，只查谁改了日期。"
    "生母裴蘅出现在旧除名录上，她不认亲，把这条并进案子。回京后卫岑三次要她自除其名以证无私，"
    "她三次拒绝，拆开除籍专权，只把「要名」的人写回户籍。有些人永不被记回，她自己扛。"
)

# 逐章大纲：三卷（1-20 上任闯祸 / 21-38 无香营善后 / 39-50 改例不立香火）
CHAPTER_OUTLINES: dict[int, str] = {
    1: "上任第一日，老吏把方绣案连同「今日结案、以证无私」的朱批一起塞过来；裴照晚看完只问：人呢，炉呢，供状呢。",
    2: "她对着光看副文，「廿三」那一横是后描的；当众拒签，把残页塞进袖里，厅里一时没人敢出声。",
    3: "卫岑请喝茶，话说到「不自证清白，下一个除名的就是你」；她把茶盏放下：「我办的是案，不是我的名声。」",
    4: "工坊空了，案卷写「畏罪逃亡」；阿麦从灶灰里爬出来，说夜里来的是礼部的车，车帘上绣着沈晏家的鹤。",
    5: "阮衡劝她按旧例销案，省得第一周就丢官；她把阿麦的口供另抄一份，藏进自己的铁匣，不进官文。",
    6: "卫岑封了工坊，她夜里翻墙，只偷走一块没注销的工牌；牌背刮过，底下还露着半个「蘅」字。",
    7: "沈晏派人送来迁祠嘉奖，说结了这案就能离开晦气衙门；她当众把锦盒退回去，只要近三年女匠除名册。",
    8: "名册被拒。阮衡夜里用油纸包来一份「已焚」底账，七十三个人名，有几个被灯油浸透。",
    9: "沈晏要她当堂交袖中残页「以证无私」；她把残页贴上照壁，司门口围了一圈看热闹的女史。",
    10: "司里罚她去抄过时祀典。废卷里「无香营」三字被刀刮了又写，她用指甲抠出最后一层。",
    11: "阿麦被以「妄言」抓走。她在发配文书上改了一笔：罪名改「待核」，人就卡在半道，发不走。",
    12: "卫岑给两条路：自请除名，或立刻销案。她选第三条，递了请调边地的帖子，司里一阵骂她疯。",
    13: "出城前阮衡塞给她半块蒸饼，饼里是半枚破印：「这印只在无香营收屯户时盖过。」",
    14: "驿站被沈晏的人翻箱，搜出的只有空白名册和方绣工牌；她问领头的：要不要连袖子也翻给你们看。",
    15: "边县不认太常寺调令，要她先「报效」几个除名换通行。她把鱼符拍在公案上：除名不是过路费。",
    16: "县档房里她把京中除名录和边上口粮编号对了半夜：七十三人在京中已死，在边上还在领粮。",
    17: "夜巡兵当她闲人轰走。女户头用土话骂兵，把她领进营门；灯影里转出来的头领正是方绣。",
    18: "方绣不谢她，先问：你来救人，还是来补档。裴照晚说：我来查谁改了日期，救不救得了是后话。",
    19: "营里白日编户，夜里炼灯。灯油碗沿写着生辰，她认出阿麦的干支，油还是温的。",
    20: "她撕走一页灯油簿，没有立刻上折；先问方绣：灭这些灯，边墙会不会塌三个月。",
    21: "边将把假祀说成军密，按着她的手要盖印。她盖了四个字：已见，未准。",
    22: "一个女户跪着求她别把自己写回京册：「除了名，婆家和债主才找不着。」裴照晚第一回把笔搁下。",
    23: "她让方绣把七十三人分成三列：要名、拒名、已死。自己不代勾，只在纸头写「本人愿」。",
    24: "沈晏信使送来灭口名单，第一行是阿麦。她连夜把阿麦改挂「在核」，名先从刀口上挪开。",
    25: "信使要搜营。方绣拆了一座假祀炉挡路，尘土漫天；裴照晚当场宣布此处为「待勘祀所」，兵不能进。",
    26: "边将暗示十个无主户换整营。她说人数不是差额，把刀鞘抵回他桌沿。",
    27: "阮衡口信到：卫岑要拿「心怀旧人」办她私心。她回了四个字：私心是册。",
    28: "旧录边页上她看见生母裴蘅，理由「妇人干政」，日期跟一批女祝齐齐整整。她看了两遍，没有哭。",
    29: "方绣承认裴蘅教过她们铸灯。裴照晚说：这条并进改日期案，不并进家谱。",
    30: "灭口刀落下，阿麦肩头见血。裴照晚反手写除名，执刀者从军籍上蒸发，营里有人倒抽一口冷气。",
    31: "有人喊她滥用除籍。她把文书抄三份：一份拍给方绣，一份塞进县档，一份塞进自己袖里。",
    32: "召回令到。方绣问还回不回京。她说：回京改规则，不是回去洗自己。",
    33: "离营只带走「要名」那一列。拒名者留下口粮编号；已死者立无祀牌，牌上不写族姓。",
    34: "路上沈晏设宴讲和，要灯油簿换一条生路。她把簿上他的花押揭下来，放进他酒杯旁边。",
    35: "沈晏翻脸扣人。阮衡咬开那半枚破印调县驿，硬把她塞进出边的车，自己挨了一鞭。",
    36: "京城门下，职名已被暂扣。卫岑隔着门缝说先自除其名再说话。她站在门外把差额七十三读完。",
    37: "太常寺女史们第一次拥上来抄她的名册。这一章的热闹不在她受辱，在她们公开复写。",
    38: "沈晏以泄露军密请旨，边墙祀力恰好晃了三日。朝堂把账算到她头上，她在殿外吃了一碗凉面。",
    39: "她不辩边墙，只呈三列名册和灯油簿，请把除籍改成双人副署。殿上静了一息，有人笑她天真。",
    40: "卫岑说副署等于削弱朝廷。她答：专权才会把活人炼成灯。这句话后来被人传出宫。",
    41: "方绣入京作证，不认恩主，只认「肯停笔的官」。堂下有人要她下跪，她不跪。",
    42: "拒名者的族人拦轿要名。她当场驳回：名不是奖赏，也不是你们的面子。",
    43: "阿麦在公堂指认礼部那辆绣鹤车。沈晏旧档被翻开，改期的那一横在光里发亮。",
    44: "沈晏夺职。卫岑仍要她自除其名。她第三次拒绝，司里有人偷偷鼓掌，被卫岑瞪了回去。",
    45: "新例通过：除名须双署，副本外放女史库。她的名字留在职官录，哪座祠都不进。",
    46: "无香营改屯为籍，假祀炉封掉，边墙改用官祀。她不邀功，只在档上写两个字：已核。",
    47: "没被写回的人夜里来寻仇，刺伤阮衡。她不追杀，让方绣把仇因写进副本，墨很浓。",
    48: "卫岑问恨不恨生母被除名。她说恨的是日期能改、活人不能见。卫岑许久没说话。",
    49: "有人提亲，有人要收她为弟子，有人要立生祠。她把帖子原路退回，重申除籍官三戒。",
    50: "岁末她独自在司里点灯，把裴蘅和「要名」诸人写入外放副本。扉页一行：不证清白，只证有过此人。",
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
