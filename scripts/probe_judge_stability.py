# scripts/probe_judge_stability.py
"""探针：同一固定 draft 对 Judge 连打 N 次，量化 consistency 维度的 overthinking 方差。

目的：判断 Judge（StepFun reasoning 模型）的极端分是「偶发」（中位数采样能治）
还是「持续」（中位数治不了，需换 Judge 模型）。

用法：
  STEPFUN_API_KEY=... .venv/bin/python scripts/probe_judge_stability.py [case_name] [n_samples]
"""
import asyncio
import os
import sys

BASE_URL = "https://api.stepfun.com/step_plan/v1"

os.environ["STEPFUN_BASE_URL"] = BASE_URL
os.environ["STEPFUN_JUDGE_MODEL"] = "step-3.7-flash"
os.environ["BASELINE_API_KEY"] = os.environ.get("STEPFUN_API_KEY", "")
os.environ["BASELINE_BASE_URL"] = BASE_URL
os.environ["BASELINE_MODEL"] = "step-3.7-flash"

from novel_agent_eval.agents.vanilla_llm import VanillaLLMAdapter
from novel_agent_eval.dataset.loader import load_cases
from novel_agent_eval.judge import Judge


async def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "long_01_魔潮将至"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    cases = load_cases("novel_agent_eval/dataset/self_built")
    by_name = {c.name: c for c in cases}
    case = by_name[target]

    # 1) 生成一个固定 draft（后续 Judge 都用它，排除生成噪声）
    gen = await VanillaLLMAdapter().generate(case)
    draft = gen.content
    print(f"case={target}  draft_len={len(draft)} chars", flush=True)

    # 2) 同一 draft 连打 n 次
    judge = Judge()
    all_dims = []
    for i in range(n):
        js = await judge.score(draft, case)
        all_dims.append(js.dimensions)
        print(f"sample {i}: {js.dimensions}", flush=True)

    print(f"\n=== 各维度方差（{n} 次采样）===", flush=True)
    for d in all_dims[0]:
        vals = [s[d] for s in all_dims]
        srt = sorted(vals)
        print(
            f"{d:15s} min={min(vals):3d} max={max(vals):3d} "
            f"range={max(vals) - min(vals):3d} median={srt[len(srt) // 2]:3d}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
