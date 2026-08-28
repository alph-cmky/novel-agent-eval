# scripts/probe_token_truth.py
"""EVAL-P0 / A-5 最小实验：确定 step-3.7-flash usage_metadata 膨胀规律（耗 API，不进 pytest/CI）。

实验矩阵：
  Exp1  直连 OpenAI 兼容 client：2 prompt 规模 × 2 reasoning 配置 × 3 repeats
  Exp2  生产 WriterAgent（call_model 路径）+ per-request 账本（script 内 monkeypatch）
  Exp2b LangChain ChatOpenAI 直连（隔离 LangChain 因素，对齐 build_chat_model 配置）

  Exp3 完整 chapter 引用 EVAL-1 已有数据，不重跑。

输出：per-request 账本 → 存 vault eval-data/，stdout 打 ratio 摘要。
"""
import asyncio
import json
import os
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_env_path = pathlib.Path("/Users/gaoyinrun/Desktop/qy/novel-agent/.env")
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

from openai import AsyncOpenAI

from novel_agent_eval.token_truth import estimate_tokens

MODEL = os.environ.get("BUDGET_MODEL", "step-3.7-flash")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
PROVIDER = "stepfun"


def cjk_prompt(n_chars: int) -> str:
    base = "少年握紧手中长剑，望向远山之外翻涌的云海，心中默念师门教诲。"
    unit = len(base)
    return (base * (n_chars // unit + 1))[:n_chars]


REQUESTS: list[dict] = []


def record(role: str, call: str, attempt: int, prompt_text: str, output_text: str, usage: dict) -> None:
    est_in = estimate_tokens(prompt_text)
    rep_in = usage.get("input_tokens", 0)
    REQUESTS.append({
        "request_id": str(uuid.uuid4())[:8],
        "model": MODEL,
        "role": role,
        "call": call,
        "attempt": attempt,
        "prompt_chars": len(prompt_text),
        "prompt_estimate_tokens": est_in,
        "output_chars": len(output_text),
        "output_estimate_tokens": estimate_tokens(output_text),
        "reported_input": rep_in,
        "reported_output": usage.get("output_tokens", 0),
        "reported_cached": (usage.get("input_token_details") or {}).get("cache_read", 0),
        "reported_reasoning": (usage.get("output_token_details") or {}).get("reasoning", 0),
        "input_ratio_vs_estimate": round(rep_in / max(est_in, 1), 2),
        "raw_usage_metadata": usage,
    })


def _usage_from_openai_resp(resp) -> dict:
    u = resp.usage
    return {
        "input_tokens": getattr(u, "prompt_tokens", 0),
        "output_tokens": getattr(u, "completion_tokens", 0),
        "input_token_details": {
            "cache_read": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
        },
        "output_token_details": {
            "reasoning": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0,
        },
    }


async def exp1_single_call(client: AsyncOpenAI) -> None:
    for label, kwargs, n_chars in [
        ("plain_small", {"max_tokens": 512}, 500),
        ("plain_large", {"max_tokens": 512}, 5000),
        ("reasoning_small", {"max_tokens": 8192, "reasoning_effort": "low"}, 500),
        ("reasoning_large", {"max_tokens": 8192, "reasoning_effort": "low"}, 5000),
    ]:
        prompt = f"请用一句话概括以下材料的主题：\n{cjk_prompt(n_chars)}"
        for attempt in range(1, 4):
            t0 = time.monotonic()
            resp = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, **kwargs,
            )
            usage = _usage_from_openai_resp(resp)
            content = resp.choices[0].message.content or ""
            record(f"exp1_{label}", "direct_chat", attempt, prompt, content, usage)
            print(f"[exp1 {label} #{attempt}] reported_in={usage['input_tokens']} est_in={estimate_tokens(prompt)} "
                  f"reported_out={usage['output_tokens']} ({time.monotonic()-t0:.1f}s)", flush=True)


async def exp2b_langchain_direct() -> None:
    """LangChain ChatOpenAI 直连（对齐 build_chat_model 配置）——隔离 LangChain 因素。"""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    for label, kwargs, n_chars in [
        ("lc_plain_large", {"max_tokens": 512}, 5000),
        ("lc_reasoning_large", {"max_tokens": 8192, "reasoning_effort": "low", "stream_chunk_timeout": None}, 5000),
    ]:
        prompt = f"请用一句话概括以下材料的主题：\n{cjk_prompt(n_chars)}"
        model = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0.0, **kwargs)
        for attempt in range(1, 4):
            resp = await model.ainvoke([HumanMessage(content=prompt)])
            usage = dict(getattr(resp, "usage_metadata", {}) or {})
            record(f"exp2b_{label}", "langchain_ainvoke", attempt, prompt, str(resp.content or ""), usage)
            print(f"[exp2b {label} #{attempt}] reported_in={usage.get('input_tokens')} "
                  f"est_in={estimate_tokens(prompt)} reported_out={usage.get('output_tokens')}", flush=True)


async def exp2_writer_call() -> None:
    """生产 WriterAgent：per-request 账本观察 call_model 每次 reported input vs prompt 大小。"""
    import novel_agent.agents.base as base_mod
    from novel_agent.agents.base import AgentConfig
    from novel_agent.agents.writer import WriterAgent
    from novel_agent.tools.base import BaseTool, ToolResult
    from pydantic import BaseModel, Field

    class SearchSchema(BaseModel):
        query: str = Field(description="Search query")

    class DummyTool(BaseTool):
        name = "search_context"
        description = "dummy search"
        input_schema = SearchSchema

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, data={"result": "前文：主角在山门外捡到断剑。"})

    orig_call_model = base_mod.BaseAgent.call_model

    async def traced_call_model(self, messages, tools=None, action="model_call"):
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        resp = await orig_call_model(self, messages, tools=tools, action=action)
        usage = dict(getattr(resp, "usage_metadata", {}) or {})
        record("exp2_writer", action, self.model_calls, prompt_text, str(resp.content or ""), usage)
        print(f"[exp2 #{self.model_calls}] reported_in={usage.get('input_tokens')} "
              f"prompt_chars={len(prompt_text)} est_in={estimate_tokens(prompt_text)} "
              f"reported_out={usage.get('output_tokens')} out_chars={len(resp.content or '')}", flush=True)
        return resp

    base_mod.BaseAgent.call_model = traced_call_model

    for run in range(1, 3):
        agent = WriterAgent(config=AgentConfig(
            model=MODEL, api_key=API_KEY, base_url=BASE_URL,
            max_tokens=4096, temperature=0.85, is_reasoning=True,
        ))
        agent.register_tool(DummyTool())
        content, _ = await agent.write(
            chapter_number=run, outline="主角在剑冢捡到断剑，残魂苏醒警告三日内血契认主。",
            context_packet={"recent_summary": "前文提要：" + cjk_prompt(800)},
            target_chapter_words=300,
        )
        print(f"[exp2 run{run}] writer_tokens in={agent.input_tokens} out={agent.output_tokens} "
              f"calls={agent.model_calls} content_chars={len(content)}", flush=True)

    base_mod.BaseAgent.call_model = orig_call_model


async def exp3_full_chapter() -> None:
    """完整单章（生产 graph 经 adapter）：per-request 账本定位膨胀发生在调用数还是单调用报告。

    额外捕获 graph 内首次 orchestrator 调用的完整 messages + 请求参数，
    存盘供 exp4 在 graph 外重放对比（同内容同参数 → 隔离 graph 上下文因素）。
    """
    import tempfile

    import novel_agent.agents.base as base_mod

    from novel_agent_eval.agents.novel_agent import NovelAgentAdapter
    from novel_agent_eval.dataset.schema import EvalCase

    orig_call_model = base_mod.BaseAgent.call_model
    orig_build = base_mod.build_chat_model
    captured: dict = {}

    async def traced_call_model(self, messages, tools=None, action="model_call"):
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        if not captured:
            captured["messages"] = [dict(m) for m in messages]
            captured["config"] = {
                "model": self.config.model, "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature, "is_reasoning": self.config.is_reasoning,
            }
        resp = await orig_call_model(self, messages, tools=tools, action=action)
        usage = dict(getattr(resp, "usage_metadata", {}) or {})
        record(f"exp3_{self.name}", action, self.model_calls, prompt_text, str(resp.content or ""), usage)
        print(f"[exp3 {self.name} #{self.model_calls}] reported_in={usage.get('input_tokens')} "
              f"prompt_chars={len(prompt_text)} est_in={estimate_tokens(prompt_text)} "
              f"reported_out={usage.get('output_tokens')} out_chars={len(resp.content or '')}", flush=True)
        return resp

    def traced_build(config):
        model = orig_build(config)
        if not captured:
            captured["chat_openai_kwargs"] = {
                k: v for k, v in getattr(model, "model_kwargs", {}).items()
            }
            captured["lc_max_tokens"] = getattr(model, "max_tokens", None)
        return model

    base_mod.BaseAgent.call_model = traced_call_model
    base_mod.build_chat_model = traced_build

    persist = tempfile.mkdtemp(prefix="novel_probe_exp3_")
    adapter = NovelAgentAdapter(max_rounds=0, persist_dir=persist, label="probe")
    case = EvalCase(
        name="token_probe_ch01",
        stage="opening",
        story_outline="《断剑重铸》：少年沈舟在剑冢捡到断剑，剑中封印上一代剑圣残魂。",
        previous_context="",
        target_chapter_outline="沈舟捡到断剑，残魂苏醒，警告三日内血契认主。",
        word_target=600,
        project_id="token_probe",
    )
    gen = await adapter.generate(case)
    print(f"[exp3] chapter done: content_chars={len(gen.content)} "
          f"token_usage={gen.meta.get('token_usage')}", flush=True)

    # ── exp4：graph 外重放同一 messages ──
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    msgs = captured["messages"]
    cfg = captured["config"]
    lc_msgs = [SystemMessage(content=msgs[0]["content"]) if m["role"] == "system" else HumanMessage(content=m["content"]) for m in msgs]
    print(f"\n[exp4] replaying orchestrator messages standalone: cfg={cfg}", flush=True)
    for attempt in range(1, 4):
        model = ChatOpenAI(
            model=cfg["model"], api_key=API_KEY, base_url=BASE_URL,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            reasoning_effort="low", stream_chunk_timeout=None,
        )
        resp = await model.ainvoke(lc_msgs)
        usage = dict(getattr(resp, "usage_metadata", {}) or {})
        record("exp4_replay_orchestrator", "langchain_ainvoke_standalone", attempt, msgs[-1]["content"], str(resp.content or ""), usage)
        print(f"[exp4 #{attempt}] reported_in={usage.get('input_tokens')} "
              f"est_in={estimate_tokens(msgs[-1]['content'])} reported_out={usage.get('output_tokens')}", flush=True)

    base_mod.BaseAgent.call_model = orig_call_model
    base_mod.build_chat_model = orig_build

    probe_payload = {
        "captured_config": cfg,
        "lc_max_tokens": captured.get("lc_max_tokens"),
        "chat_openai_kwargs": captured.get("chat_openai_kwargs"),
    }
    out_path = pathlib.Path("/Users/gaoyinrun/Documents/Obsidian Vault/novel-agent/eval-data/token_truth_probe_20260828.exp4_context.json")
    out_path.write_text(json.dumps(probe_payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    if not API_KEY:
        print("OPENAI_API_KEY 未加载", file=sys.stderr)
        sys.exit(1)
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    if only in ("all", "exp1"):
        await exp1_single_call(client)
    if only in ("all", "exp2b"):
        await exp2b_langchain_direct()
    if only in ("all", "exp2"):
        await exp2_writer_call()
    if only in ("all", "exp3"):
        await exp3_full_chapter()

    print("\n=== Per-request ledger ===")
    print("| role/call | att | prompt_ch | est_in | reported_in | ratio | reported_out | reasoning |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in REQUESTS:
        print(f"| {r['role']} | {r['attempt']} | {r['prompt_chars']} | {r['prompt_estimate_tokens']} "
              f"| {r['reported_input']} | {r['input_ratio_vs_estimate']}x | {r['reported_output']} "
              f"| {r['reported_reasoning']} |")

    out_path = pathlib.Path("/Users/gaoyinrun/Documents/Obsidian Vault/novel-agent/eval-data/token_truth_probe_20260828.json")
    existing = json.loads(out_path.read_text()) if out_path.exists() else {"requests": []}
    existing["requests"].extend(REQUESTS)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
