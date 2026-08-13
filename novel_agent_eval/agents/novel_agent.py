# novel_agent_eval/agents/novel_agent.py
"""novel-agent 完整流水线适配器：跑主仓库单章 StateGraph 生成章节。

调用范式对齐主仓库 novel_agent/api/routes.py / sse.py 的真实写法：
  1. build_chapter_graph_async(persist_dir, evolution_enabled) 编译 graph
  2. astream_events(initial_state, config, version="v2") 跑流水线
  3. 跑完 aget_state(config)：next 非空 → 在 human_review interrupt 处暂停
  4. 评测场景自动 approve：Command(resume={"action": "approve", ...}) 恢复并走完

EvalCase → initial_state 字段映射见 _map_initial_state。
persist_dir 缺省用每次调用的临时目录——注意不能传 ""：虽然 checkpoint 会退化
为内存 MemorySaver，但主仓库 writer_node 仍会以 {persist_dir}/chroma_data 创建
ChromaDB PersistentClient，传 "" 会在 cwd 落盘 chroma_data/ 污染仓库。
"""
import tempfile
import time
import zlib

from langgraph.types import Command

from novel_agent.graph.chapter import build_chapter_graph_async

from novel_agent_eval.dataset.schema import EvalCase
from novel_agent_eval.agents.base import AgentAdapter, GeneratedChapter

# case.stage → 主仓库 story_length 取值（Orchestrator 只用于篇幅语义标注，
# 单章内不改变逻辑；映射与 brief 一致：opening→short / middle→medium / long→long）
_STAGE_TO_STORY_LENGTH = {"opening": "short", "middle": "medium", "long": "long"}


class NovelAgentAdapter:
    """把主仓库完整流水线收敛成 generate(case) 接口。

    消融开关 evolution_enabled 作为构造参数，供 Task 8 run_ablation 切换
    进化 / 线性（legacy）两条路径。
    """

    name = "novel_agent"

    def __init__(self, evolution_enabled: bool = True, persist_dir: str | None = None):
        self.evolution_enabled = evolution_enabled
        # None → 每次 generate 用临时目录；也可显式指定（持久化调试用）
        self._persist_dir = persist_dir

    @staticmethod
    def _chapter_number(case: EvalCase) -> int:
        """用 case 名确定性哈希得到章号（跨进程稳定，1..10000）。"""
        return zlib.crc32(case.name.encode("utf-8")) % 10_000 + 1

    @staticmethod
    def _compose_chapter_outline(case: EvalCase) -> str:
        """章节大纲字段：主仓库 Orchestrator 只从 chapter_outline 读大纲（全书大纲
        在真项目里存 DB，评测无 ProjectManager），故把 story_outline 前置到本章大纲。"""
        sections = []
        if (case.story_outline or "").strip():
            sections.append(f"## 全书大纲\n{case.story_outline}")
        if (case.target_chapter_outline or "").strip():
            sections.append(f"## 本章大纲\n{case.target_chapter_outline}")
        return "\n\n".join(sections)

    def _map_initial_state(self, case: EvalCase, persist_dir: str) -> dict:
        """EvalCase → 主仓库 initial_state（对齐 routes.py 的字段清单）。

        纯逻辑，不依赖 LLM，可单测。
        """
        # project_id 置空：主仓库 node 用 `if project_id:` 短路，跳过 ProjectManager
        # DB 读取与 ChromaDB 检索工具注册（writer/continuity 仅 project_id 非空才挂
        # search 工具，避免触发 embedding 模型下载）；评测场景无真实项目库，置空最干净。
        return {
            "project_id": "",
            "chapter_number": self._chapter_number(case),
            "chapter_outline": self._compose_chapter_outline(case),
            "story_length": _STAGE_TO_STORY_LENGTH.get(case.stage, "long"),
            "target_chapter_words": case.word_target,
            "narrative_mode": case.narrative_mode,
            "narrative_perspective": case.narrative_perspective or "",
            "character_context": "",
            "world_context": "",
            "recent_summary": case.previous_context,
            "existing_world_entities": [],
            "persist_dir": persist_dir,
            "retry_count": 0,
        }

    @staticmethod
    def _extract_meta(
        values: dict, elapsed: float, evolution_enabled: bool
    ) -> dict:
        """从最终 state 提取评测信号（供 Task 9 internal_signals）。"""
        history = values.get("evolution_history") or []
        composites = [
            h.get("composite")
            for h in history
            if isinstance(h, dict) and h.get("composite") is not None
        ]
        return {
            "adapter": "novel_agent",
            "evolution_enabled": evolution_enabled,
            # composite_score 取进化历史里最高的 composite（无 history 时为 None）
            "composite_score": max(composites) if composites else None,
            "evolution_rounds": len(history),
            "evolution_termination": values.get("evolution_termination", ""),
            "evolution_best_version": values.get("evolution_best_version"),
            "editor_overall": (values.get("editor_report") or {}).get("overall_score"),
            "continuity_overall": (values.get("continuity_report") or {}).get("overall_score"),
            "human_approved": values.get("human_approved"),
            "elapsed_seconds": round(elapsed, 3),
            # 主仓库 NovelState 无 token 计数字段（writer latest_trace 不落 state）
            "tokens": None,
        }

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        persist_dir = self._persist_dir
        cleanup = None
        if persist_dir is None:
            persist_dir = tempfile.mkdtemp(prefix="novel_eval_")
            cleanup = lambda: _rmtree(persist_dir)

        graph = None
        final_state = None
        try:
            graph = await build_chapter_graph_async(
                persist_dir=persist_dir,
                evolution_enabled=self.evolution_enabled,
            )
            initial_state = self._map_initial_state(case, persist_dir)
            config = {
                "configurable": {
                    "thread_id": f"eval:{case.name}:{self._chapter_number(case)}",
                }
            }

            start = time.monotonic()

            async for _ in graph.astream_events(initial_state, config, version="v2"):
                pass

            final_state = await graph.aget_state(config)
            if final_state and final_state.next:
                # 流水线在 human_review interrupt 处暂停 → 评测场景自动 approve
                async for _ in graph.astream_events(
                    Command(resume={"action": "approve", "comments": ""}),
                    config,
                    version="v2",
                ):
                    pass
                final_state = await graph.aget_state(config)

            elapsed = time.monotonic() - start
        finally:
            # 关掉 AsyncSqliteSaver 的 aiosqlite 连接：其 _connection_worker_thread
            # 是非守护线程，不关会在解释器退出时永久阻塞（进程挂死）。
            # MemorySaver 无 .conn，getattr 兜底。
            checkpointer = getattr(graph, "checkpointer", None)
            conn = getattr(checkpointer, "conn", None)
            if conn is not None:
                try:
                    await conn.close()
                except Exception:
                    pass
            if cleanup:
                cleanup()

        values = final_state.values if final_state else {}
        content = (values.get("draft_content") or "").strip()
        return GeneratedChapter(
            content=content,
            meta=self._extract_meta(values, elapsed, self.evolution_enabled),
        )


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
