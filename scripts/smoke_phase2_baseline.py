# scripts/smoke_phase2_baseline.py
"""Phase 2 baseline harness 真实验证（耗 API，不进 pytest/CI）。

配对主仓库分支 feat/discourse-canon-review（../novel-agent-discourse）。

3 章 durable + 真 Judge + 真 ConStory，验证 run_baseline 端到端：
run_durable → V2 DB 读 content → Judge/ConStory 评分 → BaselineRunResult + Go/No-Go。
非正式 baseline（仅 1×1×3），正式需 6 prompts × 3 repeats × 20 chapters。
"""
import asyncio
import os
import pathlib
import sys
import tempfile

_env_path = pathlib.Path(__file__).resolve().parents[1].parent / "novel-agent" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))
# Judge / ConStory 读 STEPFUN_*，.env 是 OPENAI_*（同为 StepFun）→ 强制对齐
#（用赋值非 setdefault，避免 shell 里 stale STEPFUN_API_KEY 不被覆盖）
os.environ["STEPFUN_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["STEPFUN_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "")
os.environ.setdefault("STEPFUN_JUDGE_MODEL", "step-3.7-flash")
os.environ.setdefault("QUALITY_IS_REASONING", "true")
os.environ.setdefault("BUDGET_IS_REASONING", "true")

from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
from novel_agent_eval.baseline import go_no_go_gates, run_baseline
from novel_agent_eval.constory import ConStoryCheckerAdapter
from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.judge import Judge

STORY = (
    "《神仙也得走流程》：苏迟考公三战上岸，岗位是「天庭驻人间办事处综合岗」，"
    "编制在天庭，办公地点在城中村临街铺面，门口招牌被前任改成算命馆。"
    "办事处就她一个人，负责神仙下凡渡劫后的善后。全书女本位、无感情线。"
    "苏迟一身反骨，面对天庭流程和神仙甲方彻底发疯。搞笑来源是神仙当甲方、"
    "苏迟当乙方发疯应对、天庭流程荒诞缺大德。叙事用具体事件和对话往前推，"
    "金句要嘴贱不要抒情。"
)


def _case(ch: int) -> EvalCase:
    outlines = {
        1: "苏迟考公上岸，报到地点是城中村铺面，门牌写「天庭驻人间办事处」被邻居当算命馆。前任留了张纸条「别接龙王单，别惹孟婆，别信月老，别问第十八任去哪了」，人已经跑了。她把纸条贴在门口当免责声明。",
        2: "第一桩活：某龙王渡劫把一条街下水道炸了，居民以为是地震，市政要追责。苏迟给市政报「地质塌陷」，给天庭报「龙王失职」，两边各拿一份报告，钱从谁口袋出她不管，反正不从她口袋出。",
        3: "文曲星下班后微信发来《神仙渡劫善后操作手册》三百页，要求「明早前读完写心得」。苏迟翻到第二页睡着，醒来发现还魂丹被老鼠偷了一颗。她给文曲星回「心得：手册可防身，老鼠识字」。",
    }
    return EvalCase(
        name=f"bl_smoke_ch{ch:02d}",
        stage="opening",
        genre="女主无CP",
        story_outline=STORY,
        previous_context="",  # 留空，靠 V2 DB Canon 经 ContextCompiler 提供前文
        target_chapter_outline=outlines[ch],
        word_target=600,
        project_id="bl_smoke",
    )


async def main():
    if not os.environ.get("STEPFUN_API_KEY"):
        print("STEPFUN_API_KEY 未加载", file=sys.stderr)
        sys.exit(1)
    persist = tempfile.mkdtemp(prefix="novel_phase2_")
    ckpt = pathlib.Path(persist) / "run.json"
    print(f"[smoke] persist={persist} model={os.environ.get('QUALITY_MODEL')}", flush=True)

    adapter = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="na")
    judge = Judge(n_samples=1)
    consistency = ConStoryCheckerAdapter()

    result = await run_baseline(
        adapter=adapter, judge=judge, cases=[_case(i) for i in range(1, 4)],
        persist_dir=persist, checkpoint_path=ckpt, consistency_checker=consistency,
    )

    print("\n=== BaselineRunResult ===", flush=True)
    print(f"agent={result.agent} expected={result.expected_chapters} "
          f"completed={result.completed_chapters} success={result.full_run_success}", flush=True)
    print("\n-- per chapter --", flush=True)
    for cs in result.chapters:
        print(f"ch{cs.chapter_number} status={cs.status} overall={cs.overall} "
              f"cons_errors={cs.consistency_errors} tokens={cs.token_usage.get('total_tokens')} "
              f"latency={cs.latency_seconds:.1f}s", flush=True)

    print("\n-- Reliability --", flush=True)
    print(f"completion={result.chapter_completion_rate} timeout={result.timeout_rate} "
          f"invalid={result.invalid_rate} censorship={result.censorship_rate}", flush=True)
    print("\n-- Quality --", flush=True)
    print(f"mean={result.quality_mean} std={result.quality_std} "
          f"first={result.first_window_score} last={result.last_window_score} "
          f"degradation={result.degradation} trend={result.trend_slope}", flush=True)
    print("\n-- Consistency --", flush=True)
    print(f"CED={result.ced} first_density={result.first_error_density} "
          f"last_density={result.last_error_density} growth={result.error_growth_slope}", flush=True)
    print("\n-- Cost --", flush=True)
    print(f"total_tokens={result.total_tokens} tokens/chapter={result.tokens_per_chapter} "
          f"latency/chapter={result.latency_per_chapter}", flush=True)

    gates = go_no_go_gates([result])
    print("\n-- Go/No-Go（单 run 示意，非正式门槛）--", flush=True)
    print(gates, flush=True)

    assert result.completed_chapters == 3, "3 章应全部完成"
    assert all(cs.overall is not None for cs in result.chapters), "每章应有 overall 分"
    assert result.quality_mean is not None
    assert result.ced is not None
    print("\n✅ Phase 2 baseline harness smoke 通过：durable + Judge + ConStory + 指标聚合端到端")

    from novel_agent.graph.chapter import aclose_checkpointers
    await aclose_checkpointers()


if __name__ == "__main__":
    asyncio.run(main())
