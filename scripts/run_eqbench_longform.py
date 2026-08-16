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

_PROMPTS_PATH = Path("novel_agent_eval/dataset/eqbench/prompts.json")


def load_prompts(n: int) -> list[dict]:
    data = json.loads(_PROMPTS_PATH.read_text(encoding="utf-8"))
    # prompts.json 为 {"1": {...}, "2": {...}, ...}，按 key 排序取前 n 个
    return [data[k] for k in sorted(data, key=int)[:n]]


async def main() -> None:
    n_prompts = int(os.environ.get("N_PROMPTS", "2"))
    n_samples = int(os.environ.get("JUDGE_N_SAMPLES", "1"))
    target_pid = os.environ.get("PROMPT_INDEX", None)  # 1-indexed: "1" 或 "2"

    all_prompts = load_prompts(n_prompts)
    if target_pid:
        idx = int(target_pid) - 1
        prompts = [(idx + 1, all_prompts[idx])]
    else:
        prompts = list(enumerate(all_prompts, start=1))

    bridge = EQBenchBridge()
    judge = EQBenchJudge(n_samples=n_samples)
    agents = [NovelAgentAdapter(evolution_enabled=True), VanillaLLMAdapter()]

    results = []
    for pid, prompt in prompts:
        writing_prompt = prompt["writing_prompt"]
        title = prompt["title"]
        # 同一 prompt 的 plan 只跑一次 bridge，两个 agent 共享（省一半 planning 调用）
        plan = await bridge.plan(
            writing_prompt, prompt_id=str(pid), title=title, category=prompt["category"]
        )
        print(
            f"[plan] {title}: final_plan={len(plan.final_plan)} chars, "
            f"characters={len(plan.character_profiles)} chars",
            flush=True,
        )
        for agent in agents:
            try:
                res = await run_longform(agent=agent, judge=judge, plan=plan)
            except Exception as e:  # noqa: BLE001 — 单 prompt 偶发崩溃不中断整体横评
                print(f"[{agent.name}] {title} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            results.append(res)
            per_ch = "/".join(f"{s:.0f}" for s in res.chapter_scores)
            print(
                f"[{res.agent}] {res.title} 0-100={res.eqbench_0_100:.1f} "
                f"degradation={res.degradation:+.2f} | {per_ch}",
                flush=True,
            )

    print("\n=== EQ-Bench Longform degradation 报告 ===\n")
    print(render_longform_table(results))

    out = Path("/tmp/eqbench_longform.json")
    out.write_text(
        json.dumps(
            [
                {
                    "agent": r.agent,
                    "prompt_id": r.prompt_id,
                    "title": r.title,
                    "mean_score": r.mean_score,
                    "eqbench_0_100": r.eqbench_0_100,
                    "degradation": r.degradation,
                    "chapter_scores": r.chapter_scores,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已存 {out}")


if __name__ == "__main__":
    asyncio.run(main())
