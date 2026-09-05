# scripts/smoke_phase2_baseline.py
"""Phase 2 baseline harness 真实验证（耗 API，不进 pytest/CI）。

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
    "《无祀录》：北冕靠「记名入祀」收灵。除籍司专干把人从人间抹掉的脏活。"
    "全书女本位、无感情线。叙事用场面和对话往前推，少写公文。"
    "裴照晚上任就被砸上方绣案，限当日结案以证无私；她不结，当众拒签。"
)


def _case(ch: int) -> EvalCase:
    outlines = {
        1: "上任第一日，老吏把方绣案连同「今日结案、以证无私」的朱批一起塞过来；裴照晚看完只问：人呢，炉呢，供状呢。",
        2: "她对着光看副文，「廿三」那一横是后描的；当众拒签，把残页塞进袖里，厅里一时没人敢出声。",
        3: "卫岑请喝茶，话说到「不自证清白，下一个除名的就是你」；她把茶盏放下：「我办的是案，不是我的名声。」",
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
