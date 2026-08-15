# tests/test_eqbench_bridge.py
"""EQBenchBridge mock 测试：不消耗真实 API。"""
import asyncio

from novel_agent_eval.eqbench_bridge import EQBenchBridge, LongformPlan, plan_to_cases


class _RecordingCompletions:
    """按调用顺序返回固定内容，并记录每次的 messages。"""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = []  # 每次调用的 messages 参数

    async def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        content = self._contents[len(self.calls) - 1]
        message = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


def _client(contents):
    comps = _RecordingCompletions(contents)
    client = type("Client", (), {"chat": type("Chat", (), {"completions": comps})()})()
    return client, comps


def _plan(writing_prompt="Write a short story.") -> LongformPlan:
    return LongformPlan(
        prompt_id="1",
        title="Test Story",
        category="Fantasy",
        writing_prompt=writing_prompt,
        n_chapters=8,
        final_plan="# Intention\nA tale.\n\n# Chapter Planning\nch1..ch8",
        character_profiles="# Hero\nBrave.",
        step_outputs={str(i): f"step{i}" for i in range(1, 6)},
    )


def test_plan_runs_5_steps_and_extracts_final_plan_characters():
    client, comps = _client([f"step{i}" for i in range(1, 6)])
    plan = asyncio.run(
        EQBenchBridge(client=client, n_chapters=8).plan(
            "Write a short story.", prompt_id="1", title="T", category="F"
        )
    )
    assert len(comps.calls) == 5
    assert plan.final_plan == "step4"
    assert plan.character_profiles == "step5"


def test_build_messages_accumulates_history():
    bridge = EQBenchBridge(client=_client(["x"])[0], n_chapters=8)
    outputs = {"1": "brainstorm_out", "2": "plan_out"}
    messages = bridge._build_messages(3, "WP", outputs)
    # 历史：user(prompt1) + assistant(step1) + user(prompt2) + assistant(step2) + user(prompt3)
    assert len(messages) == 5
    assert messages[0]["role"] == "user"
    assert "WP" in messages[0]["content"]            # prompt1 注入 writing_prompt
    assert messages[1] == {"role": "assistant", "content": "brainstorm_out"}
    assert messages[2]["role"] == "user"
    assert "8" in messages[2]["content"]             # prompt2 注入 n_chapters
    assert messages[3] == {"role": "assistant", "content": "plan_out"}
    assert messages[4]["role"] == "user"             # 当前步 prompt3（无占位符）


def test_plan_to_cases_produces_n_chapters():
    cases = plan_to_cases(_plan())
    assert len(cases) == 8
    assert cases[0].name == "eqbench_p1_ch01"
    assert cases[7].name == "eqbench_p1_ch08"
    assert cases[0].previous_context == ""
    # story_outline 含 writing_prompt + character_profiles + final_plan
    for keyword in ("Write a short story.", "# Hero", "Brave.", "# Intention", "A tale."):
        assert keyword in cases[0].story_outline
    # target_chapter_outline 给「按计划写第 i 章」指令
    assert "chapter 1 of 8" in cases[0].target_chapter_outline
    assert "chapter 8 of 8" in cases[7].target_chapter_outline
    assert cases[0].stage == "opening"
    assert cases[0].word_target == 1000


def test_plan_to_cases_word_target_override():
    cases = plan_to_cases(_plan(), word_target=3000)
    assert cases[0].word_target == 3000
