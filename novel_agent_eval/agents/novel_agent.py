# novel_agent_eval/agents/novel_agent.py
"""novel-agent 适配器：跑主仓库话语合同编排生成章节。

配对主仓库分支：``feat/discourse-canon-review``（目录 ``novel-agent-discourse``）。

调用范式（S1）：
1. ``run_agent_loop(initial_state)``：
   Orchestrator → Writer tool loop → Hard Gate
   → PASS：跳过 Editor + Continuity
   → FAIL：Editor（rewrite 有界重入）→ Continuity
   → Worldbuilding → 返回（评测侧自动 approve / commit）
2. loop 内无人审 interrupt；空稿抛 ``EmptyDraftError``。

EvalCase → initial_state 字段映射见 ``_map_initial_state``。
persist_dir 缺省用每次调用的临时目录——不能传 ""，否则 ChromaDB 会在 cwd 落盘。
"""

import hashlib
import json
import os
import re
import tempfile
import time
import zlib
from pathlib import Path

from novel_agent.api.run_service import ChapterRunService
from novel_agent.graph.agent_loop import EmptyDraftError, run_agent_loop
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.services.context import ContextCompiler
from novel_agent.storage.manager import ProjectManager

from novel_agent_eval.agents.base import GeneratedChapter
from novel_agent_eval.dataset.schema import EvalCase

# case.stage → 主仓库 story_length 取值（Orchestrator 只用于篇幅语义标注，
# 单章内不改变逻辑；映射与 brief 一致：opening→short / middle→medium / long→long）
_STAGE_TO_STORY_LENGTH = {"opening": "short", "middle": "medium", "long": "long"}


class NovelAgentAdapter:
    """把主仓库 S1 流水线收敛成 generate(case) 接口。

    S1 下 Hard Gate 条件化审查写在主仓库 ``agent_loop`` 内，适配器不再注入
    ``deterministic_gate_first`` / ``scene_first`` 到 state。

    仍接受 ``max_rounds`` / ``scene_first`` / ``deterministic_gate_first`` 关键字，
    仅为兼容旧脚本与消融工厂；S1 路径上它们不改变编排（scene-first 禁止启用）。
    """

    name = "novel_agent"
    orchestration = "discourse"

    def __init__(
        self,
        persist_dir: str | None = None,
        label: str | None = None,
        skip_orchestrator: bool = False,
        resume: bool = False,
        project_id: str = "",
        synthetic_context: bool = False,
        max_rounds: int | None = None,
        scene_first: bool = False,
        deterministic_gate_first: bool = True,
    ):
        if scene_first:
            raise ValueError(
                "S1 adapter rejects scene_first=True (scene-first is excluded by eval evidence)"
            )
        self.skip_orchestrator = skip_orchestrator
        # resume=True：跨进程恢复——按 name 复用已有 Project，已 approved 章节不重生。
        # V2 durable state（Project/ChapterVersion/Canon）即真相源，不依赖 eval 侧 checkpoint。
        self.resume = resume
        self.project_id = project_id
        # B-3：默认 False → 上下文以 V2 DB（真实 Production ContextCompiler）为准；
        # 仅专门测试「有限 harness context」的 benchmark 才置 True（显式标记）。
        self.synthetic_context = synthetic_context
        # Legacy ablation knobs (ignored by S1 agent_loop control flow).
        self.max_rounds = max_rounds
        self.scene_first = False
        self.deterministic_gate_first = deterministic_gate_first
        if label:
            self.name = label
        # None → 每次 generate 用临时目录；也可显式指定（持久化调试用）
        self._persist_dir = persist_dir
        self._sessions: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _chapter_number(case: EvalCase) -> int:
        """解析长篇 case 的真实章号，否则用稳定哈希生成。"""
        chapter_match = re.search(r"(?:^|_)ch(\d+)(?:_|$)", case.name)
        if chapter_match:
            return int(chapter_match.group(1))
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

    @staticmethod
    def _truncate_previous_context(context: str, max_chars: int = 1500) -> str:
        """多章连续连载时的智能前文滑动窗口截断。

        若前文超过 max_chars（如第 7/8 章累积数万字），只保留首段背景提示 + 最近一章末尾 1200 字，
        防止超长前文挤爆 Prompt 导致注意力稀释与后程字数崩塌。
        """
        if not context or len(context) <= max_chars:
            return context
        # 保留前 300 字作为宏观背景，加上末尾 1200 字作为紧邻剧情钩子
        head = context[:300].strip()
        tail = context[-1200:].strip()
        return f"{head}\n\n[...中间章节前文已由世界观记忆库接管...]\n\n{tail}"

    def _map_initial_state(self, case: EvalCase, persist_dir: str) -> dict:
        """EvalCase → 主仓库 initial_state（V2 契约：单一上下文载体 context_packet）。

        纯逻辑，不依赖 LLM，可单测。

        V2 所有 node（orchestrator/writer/editor/continuity）只读
        ``state["context_packet"]``，不再读取顶层 character_context /
        world_context / recent_summary / existing_world_entities（这些是
        V1 遗留字段，NovelState 已无定义）。故：
        - 删除上述 legacy 顶层字段，避免向 V2 传递失效 context；
        - previous_context 折叠进 ``context_packet["recent_summary"]``，
          经 ContextCompiler 一次性塑造后由各 node 的 for_* 投影消费；
        - retry_count / writer_prompt_profile 同属已废弃 V1 字段，删除。
        """
        chapter_number = self._chapter_number(case)
        state = {
            "project_id": case.project_id or self.project_id,
            "chapter_number": chapter_number,
            "chapter_outline": self._compose_chapter_outline(case),
            "story_length": _STAGE_TO_STORY_LENGTH.get(case.stage, "long"),
            "target_chapter_words": case.word_target,
            "narrative_mode": case.narrative_mode,
            "narrative_perspective": case.narrative_perspective or "",
            "context_packet": self._eval_context_packet(case, chapter_number),
            "persist_dir": case.persist_dir or persist_dir,
        }
        if self.skip_orchestrator:
            state["skip_orchestrator"] = True
        return state

    def _eval_context_packet(self, case: EvalCase, chapter_number: int) -> dict:
        """由 EvalCase 构造 V2 ContextPacket 的初始投影。

        previous_context 属 benchmark 输入（DB 为空时是唯一前文来源）。
        synthetic_context=True 时才做 head+tail 截断——那是专门测试
        「有限 harness context」的模式；默认路径不做 eval 侧 memory 模拟，
        长篇前文一律由 V2 DB 经 ContextCompiler 提供。
        """
        prev = case.previous_context or ""
        if self.synthetic_context:
            prev = self._truncate_previous_context(prev)
        return {
            "project_id": case.project_id or self.project_id,
            "chapter_number": chapter_number,
            "character_context": "",
            "world_context": "",
            "recent_summary": prev,
            "unresolved_foreshadowings": [],
            "timeline_events": [],
            "timeline_findings": [],
        }

    @staticmethod
    def _merge_context_packet(
        eval_packet: dict, db_packet: dict, *, prefer: str = "production"
    ) -> dict:
        """合并评测 previous_context 投影与 ContextCompiler 编译出的 Canon 投影。

        B-3 parity：默认 prefer="production" —— DB（真实 Production memory）优先；
        eval previous_context 只在 DB 为空时补位（单章 benchmark 输入）。
        prefer="synthetic" 供显式标记的 harness-context benchmark 反转优先级。
        """
        if not db_packet:
            return eval_packet
        if not eval_packet:
            return db_packet
        merged = {**eval_packet, **db_packet}
        eval_summary = eval_packet.get("recent_summary", "") or ""
        db_summary = db_packet.get("recent_summary", "") or ""
        if prefer == "synthetic":
            merged["recent_summary"] = eval_summary or db_summary
        else:
            merged["recent_summary"] = db_summary or eval_summary
        return merged

    @staticmethod
    def _find_project_by_name(manager, name: str) -> str | None:
        """跨进程复用：按 name 在 persist_dir 的 DB 里找已有 project，返回 id；无则 None。"""
        for project in manager.list_projects():
            if project.get("name") == name:
                return project["id"]
        return None

    def _maybe_resume_chapter(
        self, manager, project_id: str, chapter_number: int, case: EvalCase
    ) -> GeneratedChapter | None:
        """resume=True 且该章已 approved：返回缓存 content，跳过重生。

        跨进程恢复的核心：第 N 章在进程 A 已 commit，进程 B 不应重生。
        V2 durable state 即真相源——content 取自 chapters 表，writing_run_id
        取自 writing_runs。进程 A 的 token_usage / context_packet_hash 随进程
        消失，无法恢复（标记为 resumed / None），需 eval 侧逐章 checkpoint 补全。
        """
        chapter = manager.get_chapter(project_id, chapter_number)
        if not chapter or chapter.get("status") != "approved":
            return None
        runs = manager.list_writing_runs(project_id, chapter_number)
        run = runs[0] if runs else None
        content = (chapter.get("draft_content") or "").strip()
        meta = {
            "adapter": self.name,
            "resumed": True,
            "project_id": project_id,
            "writing_run_id": run["id"] if run else None,
            "chapter_number": chapter_number,
            "elapsed_seconds": 0.0,
            "tokens": 0,
            "token_usage": NovelAgentAdapter._extract_token_usage({}),
            "evolution_rounds": 0,
            "evolution_termination": "resumed",
            "context_packet_hash": None,
            "workflow_version": "v2",
        }
        return GeneratedChapter(content=content, meta=meta)

    def index_chapter(
        self, *, project_id: str, persist_dir: str, chapter_number: int, content: str
    ) -> None:
        """Persist generated chapter text for subsequent longform retrieval."""
        for (session_dir, external_id), session in self._sessions.items():
            if (
                session_dir == str(Path(persist_dir).resolve())
                and external_id == project_id
            ):
                project_id = session["project_id"]
                break
        store = ChapterStore(Path(persist_dir) / "chroma_data")
        store.index_chapter(project_id, chapter_number, content)

    def close_session(self, project_id: str, persist_dir: str) -> None:
        """Release one longform session after its artifacts have been persisted."""
        key = (str(Path(persist_dir).resolve()), project_id)
        self._sessions.pop(key, None)

    @staticmethod
    def _extract_token_usage(values: dict) -> dict:
        """从 NovelState 提取 token 消耗（去重 + validity 标注）。

        provider 语义：cached ⊂ input，reasoning ⊂ output。
        total = sum(input + output) per role；cached/reasoning 是 telemetry，
        不重复计入 total，单独报告供诊断。

        数值为 provider usage_metadata 原始报告（raw），按 token_truth 规则
        注入 validity / normalization 元数据。step-3.7-flash 经 A-5 实验证实
        在 graph 执行上下文内报告 600-2900× 膨胀 → suspect，不进 trusted 聚合。

        主仓库 node 把 provider usage_metadata 累加进 NovelState per-role 字段。
        """
        from novel_agent_eval.token_truth import annotate_usage, detect_provider

        total_input = 0
        total_output = 0
        total_cached = 0
        total_reasoning = 0
        roles = (
            "orchestrator",
            "writer",
            "editor",
            "continuity",
            "worldbuilding",
            "evolution",
        )
        for role in roles:
            total_input += int(values.get(f"{role}_input_tokens") or 0)
            total_output += int(values.get(f"{role}_output_tokens") or 0)
            total_cached += int(values.get(f"{role}_cached_tokens") or 0)
            total_reasoning += int(values.get(f"{role}_reasoning_tokens") or 0)
        usage = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "cached_tokens": total_cached,
            "reasoning_tokens": total_reasoning,
            "total_tokens": total_input + total_output,
        }
        # role → 模型：与主仓库 ModelRouter.route_for 对齐
        quality = os.environ.get("QUALITY_MODEL", "")
        budget = os.environ.get("BUDGET_MODEL", "")
        models = [m for m in (quality, budget) if m]
        provider = detect_provider(os.environ.get("OPENAI_BASE_URL", ""))
        return annotate_usage(usage, provider=provider, models=models)

    @staticmethod
    def _packet_hash(packet: dict) -> str | None:
        """对最终 context_packet 计算 sha256，作为前文状态可复现性的观测锚点。"""
        if not packet:
            return None
        return hashlib.sha256(
            json.dumps(packet, ensure_ascii=False, sort_keys=True, default=str).encode(
                "utf-8"
            )
        ).hexdigest()

    _ROLES = (
        "orchestrator",
        "writer",
        "editor",
        "continuity",
        "worldbuilding",
        "evolution",
    )

    @classmethod
    def _extract_cost_attribution(cls, values: dict) -> dict:
        """C-2：per-role cost attribution（model_calls / tokens / latency / tool calls）。

        全部来自 NovelState 观测字段；数值为 provider 原始报告（token 真值以
        token_truth validity 为准），latency 为节点 wall time。
        """
        attr: dict[str, dict] = {}
        for role in cls._ROLES:
            entry = {
                "model_calls": int(values.get(f"{role}_model_calls") or 0),
                "input_tokens": int(values.get(f"{role}_input_tokens") or 0),
                "output_tokens": int(values.get(f"{role}_output_tokens") or 0),
                "cached_tokens": int(values.get(f"{role}_cached_tokens") or 0),
                "reasoning_tokens": int(values.get(f"{role}_reasoning_tokens") or 0),
                "latency_seconds": round(
                    float(values.get(f"{role}_latency_seconds") or 0.0), 3
                ),
            }
            attr[role] = entry
        # writer 独有：tool / search 调用（其余角色无工具边界）
        attr["writer"]["tool_calls"] = int(values.get("writer_tool_calls") or 0)
        attr["writer"]["search_calls"] = int(values.get("writer_search_calls") or 0)
        return attr

    @staticmethod
    def _extract_context_sizes(values: dict) -> dict:
        """C-3：per-role context 投影尺寸（复用生产 ContextCompiler 投影函数，不重实现）。

        最终 state 的 context_packet 是 orchestrator context_needed 检索增强后的
        Writer 输入视角；editor/continuity/orchestrator 投影由生产静态投影函数
        对同一 packet 求出（与节点内实际调用一致）。
        """
        packet = values.get("context_packet") or {}
        if not packet:
            return {}
        from novel_agent.services.context import ContextCompiler

        def _size(text: object) -> int:
            return len(text) if isinstance(text, str) else 0

        def _packet_size(p: dict) -> dict:
            return {
                "character_context_chars": _size(p.get("character_context")),
                "world_context_chars": _size(p.get("world_context")),
                "recent_summary_chars": _size(p.get("recent_summary")),
                "foreshadowings": len(p.get("unresolved_foreshadowings") or []),
                "timeline_events": len(p.get("timeline_events") or []),
                "timeline_findings": len(p.get("timeline_findings") or []),
            }

        sizes: dict[str, dict] = {"writer_view": _packet_size(packet)}
        for role, fn in (
            ("orchestrator_view", ContextCompiler.for_orchestrator),
            ("editor_view", ContextCompiler.for_editor),
            ("continuity_view", ContextCompiler.for_continuity),
        ):
            try:
                sizes[role] = _packet_size(fn(packet))
            except Exception:  # noqa: BLE001 - 投影失败不阻断评测信号
                sizes[role] = {}
        return sizes

    @staticmethod
    def _extract_evolution_path(values: dict) -> list[dict]:
        """C-7：evolution 路径——每轮 revision 的触发/focus/reviewers/writer 成本/结果。"""
        path: list[dict] = []
        for h in values.get("evolution_history") or []:
            if not isinstance(h, dict):
                continue
            guard = h.get("quality_guard") or {}
            path.append(
                {
                    "revision": h.get("v"),
                    "composite": h.get("composite"),
                    "editor": h.get("editor"),
                    "continuity": h.get("continuity"),
                    "delta": h.get("delta"),
                    "focus": h.get("focus"),
                    "reviewers": h.get("reviewers"),
                    "writer_tokens": h.get("writer_tokens"),
                    "guard_violations": len(guard.get("violations") or []),
                }
            )
        return path

    @classmethod
    def _extract_meta(cls, values: dict, elapsed: float) -> dict:
        """从最终 state 提取评测信号（供 Task 9 internal_signals）。"""
        history = values.get("evolution_history") or []
        composites = [
            h.get("composite")
            for h in history
            if isinstance(h, dict) and h.get("composite") is not None
        ]
        # 每条 history entry 对应一版 draft：v0 = initial draft，v1..vN = rewrites。
        # Evolution 次数 = 重写次数 = len(history) - 1；禁止把 len(history) 当轮数。
        revision_count = max(len(history) - 1, 0)
        token_usage = NovelAgentAdapter._extract_token_usage(values)
        return {
            "adapter": "novel_agent",
            # composite_score 取进化历史里最高的 composite（无 history 时为 None）
            "composite_score": max(composites) if composites else None,
            "generation_round": 1,  # 单章一次生成会话
            "evolution_revision_count": revision_count,
            "evolution_rounds": revision_count,
            "evolution_history": history,
            "evolution_termination": values.get("evolution_termination", ""),
            "quality_guard_report": values.get("quality_guard_report", {}),
            "evolution_candidates": values.get("evolution_candidates", []),
            "evolution_best_candidate_version": values.get(
                "evolution_best_candidate_version"
            ),
            "evolution_best_version": values.get(
                "evolution_best_candidate_version",
                values.get("evolution_best_version"),
            ),
            "editor_overall": (values.get("editor_report") or {}).get("overall_score"),
            "continuity_overall": (values.get("continuity_report") or {}).get(
                "overall_score"
            ),
            "human_approved": values.get("human_approved"),
            "elapsed_seconds": round(elapsed, 3),
            "workflow_version": "v2",
            "orchestration": "discourse",
            "quality_gate_passed": values.get("quality_gate_passed"),
            "timeline_passed": values.get("timeline_passed"),
            "discourse_contract": (
                (values.get("orchestrator_strategy") or {}).get("discourse_contract")
                or {}
            ),
            "contract_report": values.get("contract_report") or {},
            "contract_passed": values.get("contract_passed"),
            "revision_brief": values.get("revision_brief") or "",
            "surface_applied": bool(values.get("surface_applied")),
            "quality_gate_report": values.get("quality_gate_report") or {},
            "editor_skipped": bool(values.get("editor_skipped")),
            "continuity_skipped": bool(values.get("continuity_skipped")),
            "worldbuilding_warnings": list(values.get("worldbuilding_warnings") or []),
            "revision_feedback_consumed": bool(
                values.get("revision_feedback_consumed")
            ),
            # 主仓库未在 state 落 hash 时，由最终 context_packet 计算，保证可复现
            "context_packet_hash": values.get("context_packet_hash")
            or NovelAgentAdapter._packet_hash(values.get("context_packet") or {}),
            "writing_run_id": values.get("writing_run_id"),
            # 真实 token trace：per-role + total（替代旧 tokens=None）
            "token_usage": token_usage,
            # 兼容既有 efficiency_score / report 的 meta["tokens"] 路径：取 total_tokens
            "tokens": token_usage["total_tokens"],
            # C-2 / C-3 / C-7：成本归因、上下文尺寸、evolution 路径
            "cost_attribution": cls._extract_cost_attribution(values),
            "context_sizes": cls._extract_context_sizes(values),
            "evolution_path": cls._extract_evolution_path(values),
        }

    async def generate(self, case: EvalCase) -> GeneratedChapter:
        persist_dir = self._persist_dir or case.persist_dir
        cleanup = None
        if persist_dir is None:
            persist_dir = tempfile.mkdtemp(prefix="novel_eval_")
            cleanup = lambda: _rmtree(persist_dir)

        manager = None
        run = None
        snapshot_hash = None
        try:
            if ProjectManager is not None:
                session_key = (
                    str(Path(persist_dir).resolve()),
                    case.project_id or case.name,
                )
                session = self._sessions.get(session_key)
                if session is None:
                    manager = ProjectManager(Path(persist_dir))
                    project_name = case.project_id or f"eval:{case.name}"
                    # resume：跨进程按 name 复用已有 project，不再每次 init_project 随机 id
                    project_id = None
                    if self.resume:
                        project_id = self._find_project_by_name(manager, project_name)
                    if project_id is None:
                        project_id = manager.init_project(
                            name=project_name,
                            title=project_name,
                            story_length=_STAGE_TO_STORY_LENGTH.get(case.stage, "long"),
                            target_chapter_words=case.word_target,
                        )
                    session = {
                        "manager": manager,
                        "project_id": project_id,
                        "runs": {},
                    }
                    self._sessions[session_key] = session
                manager = session["manager"]
                project_id = session["project_id"]
                chapter_number = self._chapter_number(case)
                # resume：已 approved 章节不重生，直接返缓存 content（跨进程恢复）
                if self.resume:
                    resumed = self._maybe_resume_chapter(
                        manager, project_id, chapter_number, case
                    )
                    if resumed is not None:
                        return resumed
                run = session["runs"].get(chapter_number)
                if run is None:
                    try:
                        run = manager.create_writing_run(
                            project_id,
                            chapter_number,
                            run_type="evaluation",
                            workflow_version="v2",
                        )
                    except ValueError as exc:
                        # 上次中断/失败遗留的 active run（queued/running/...）会触发
                        # 唯一约束——评测场景里该章已判失败，清掉 stale run 后重试一次。
                        if "already has an active run" not in str(exc):
                            raise
                        stale_active = {
                            "queued",
                            "running",
                            "waiting_review",
                            "waiting_user",
                            "retrying",
                        }
                        for stale in manager.list_writing_runs(
                            project_id, chapter_number
                        ):
                            if (
                                stale.get("status") in stale_active
                                and ChapterRunService is not None
                            ):
                                ChapterRunService(manager).cancel(stale["id"])
                        run = manager.create_writing_run(
                            project_id,
                            chapter_number,
                            run_type="evaluation",
                            workflow_version="v2",
                        )
                    session["runs"][chapter_number] = run
                context_state = (
                    ContextCompiler(manager)
                    .compile(project_id, chapter_number)
                    .to_state()
                )
                snapshot_hash = manager.get_canon_snapshot(run["input_snapshot_id"])[
                    "content_hash"
                ]
            else:
                context_state = {}
            initial_state = self._map_initial_state(case, persist_dir)
            # B-3 单一上下文契约：DB 投影（Production memory）默认优先；
            # synthetic_context=True 时反转（显式标记的 harness-context benchmark）。
            if context_state:
                initial_state["context_packet"] = self._merge_context_packet(
                    initial_state.get("context_packet") or {},
                    context_state.get("context_packet") or {},
                    prefer="synthetic" if self.synthetic_context else "production",
                )
            if run:
                initial_state["project_id"] = project_id
                initial_state["writing_run_id"] = run["id"]

            start = time.monotonic()
            try:
                values = await run_agent_loop(initial_state)
            except EmptyDraftError as exc:
                raise RuntimeError(f"empty content from S1 pipeline: {exc}") from exc
            elapsed = time.monotonic() - start
        finally:
            pass

        # post-gen DB writes 必须在 persist_dir rmtree 之前：attach_candidate / commit
        # 读写 novel.db，清理过早会让 get_writing_run 返回 None。
        content = (values.get("draft_content") or "").strip()
        # Rule Gate（§3.1 正文非空）：S1 通常在 loop 内抛 EmptyDraftError；
        # 此处兜底防止空章节被伪装成 completed。
        if not content:
            gate = values.get("quality_gate_report") or {}
            raise RuntimeError(
                f"empty content from pipeline; quality_gate blocked: "
                f"{gate.get('violations') or 'unknown'}"
            )
        try:
            if manager and run and content and ChapterRunService is not None:
                service = ChapterRunService(manager)
                version = service.attach_candidate(
                    run["id"],
                    content,
                    origin="evaluation",
                    scene_plan=values.get("scene_plan", []),
                    scene_drafts=values.get("scene_drafts", []),
                )
                worldbuilding = values.get("worldbuilding_report") or {}
                if worldbuilding:
                    manager.create_canon_proposal(
                        project_id,
                        run["chapter_number"],
                        "worldbuilding",
                        worldbuilding,
                        run_id=run["id"],
                        version_id=version["id"],
                    )
                for proposal in manager.list_canon_proposals(
                    project_id, run_id=run["id"], status="proposed"
                ):
                    manager.review_canon_proposal(
                        proposal["id"], "accepted", "evaluation approval"
                    )
                service.commit(run["id"])
        finally:
            if cleanup:
                cleanup()
        meta = self._extract_meta(values, elapsed)
        meta["synthetic_context"] = self.synthetic_context
        meta["adapter_max_rounds"] = self.max_rounds
        meta["adapter_deterministic_gate_first"] = self.deterministic_gate_first
        if run:
            meta.update(
                {
                    "writing_run_id": run["id"],
                    "project_id": run["project_id"],
                    "context_snapshot_hash": snapshot_hash,
                }
            )
        # C-3：Canon 存储增长（本章 commit 后的 DB 状态）
        if manager:
            try:
                meta["canon_counts"] = manager.get_canon_counts(
                    meta.get("project_id") or "",
                    after_chapter=values.get("chapter_number", 0),
                )
            except Exception:  # noqa: BLE001 - 统计失败不阻断评测
                meta["canon_counts"] = {}
        return GeneratedChapter(
            content=content,
            meta=meta,
        )


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
