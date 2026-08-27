# Production–Eval Alignment Audit

> Phase 0 联合基线冻结。本文档记录两个仓库的当前行为，对照是否一致，
> 标注不一致项及权威方。**Phase 0 不修改代码**，只建立可信基线。

---

## 1. Baseline Freeze

### 1.1 Novel-Agent (Production)

| 项 | 值 |
|---|---|
| 仓库路径 | `/Users/gaoyinrun/Desktop/qy/novel-agent` |
| commit SHA | `a29be27de0c6fb0cb7ba8429443729ac5432e228` |
| 分支 | `master` |
| 工作区状态 | **干净**（无未提交改动） |
| Python | 3.12（`.python-version`） |
| 包管理 | uv（`uv.lock` 存在，654 KB） |
| pytest 收集 | **423 tests** |
| ruff check | All checks passed |
| ruff format | 99 files already formatted（干净） |

#### Agents（6 个 LLM Agent）

| Agent | 文件 | TaskClass | 模型路由 | temperature |
|---|---|---|---|---|
| Orchestrator | `agents/orchestrator.py` | STRUCTURAL | BUDGET_MODEL | 0.4 |
| Writer | `agents/writer.py` | CREATIVE | QUALITY_MODEL | 0.85 |
| Editor | `agents/editor.py` | REVIEW | BUDGET_MODEL | 0.3 |
| Continuity | `agents/continuity.py` | REVIEW | BUDGET_MODEL | 0.3 |
| Worldbuilding | `agents/worldbuilding.py` | EXTRACTION | BUDGET_MODEL | 0.2 |
| EvolutionOrchestrator | `agents/evolution_orchestrator.py` | META_EVALUATION | BUDGET_MODEL | 0.2 |

- `QUALITY_MODEL` / `BUDGET_MODEL` 走环境变量；fallback = `deepseek-chat`。
- `QUALITY_API_KEY` / `BUDGET_API_KEY` 可独立配置，缺省回退 `OPENAI_API_KEY`。

#### Graph 拓扑（`graph/chapter.py`）

```
orchestrator
  → evolution_writer
  → route_after_writer ─┬─ evolution_editor        (gate 未过 / 需 review)
                        ├─ worldbuilding            (gate 过 / skip review)
                        └─ evolution_orchestrator   (style-only 修订跳过 editor)
  → evolution_editor → route_after_editor ─┬─ evolution_continuity
                                             ├─ evolution_worldbuilding
                                             └─ evolution_orchestrator
  → evolution_continuity → route_after_continuity ─┬─ evolution_worldbuilding
                                                      └─ evolution_orchestrator
  → evolution_worldbuilding → evolution_orchestrator
  → route_after_evolution ─┬─ evolution_writer        (继续迭代)
                            └─ evolution_select_best    (终止)
  → evolution_select_best → worldbuilding → human_review
  → route_after_human_evolution ─┬─ END              (approved)
                                   └─ evolution_writer (rejected, 最多 3 次)
```

- 入口：`orchestrator`；默认 mode = **Chapter-first**（`scene_first` 未在默认 state 设置）。
- `scene_first=True` 时才触发 `build_scene_plan` + 逐 scene 生成。

#### State 字段（`graph/state.py` — `NovelState`）

```text
项目上下文: project_id, writing_run_id, chapter_number, chapter_outline,
            story_length, target_chapter_words, narrative_mode, narrative_perspective
Agent 输出: draft_content, editor_report, continuity_report,
            worldbuilding_report, orchestrator_strategy, style_report,
            skip_orchestrator, skip_reviews, skip_worldbuilding,
            review_interval, skip_evolution_enrichment
上下文载体: context_packet (单一载体), scene_first, scene_plan, scene_drafts
人类审阅:   human_approved, human_feedback, evolution_human_rejects, human_review_exhausted
进化控制:   evolution_max_rounds, evolution_convergence_threshold,
            evolution_round, evolution_version
进化状态:   evolution_history, evolution_candidates, evolution_improvement_plan,
            evolution_termination, evolution_best_candidate_version,
            quality_guard_report, quality_gate_report, deterministic_gate_first
存储:       persist_dir
可观测性:   trace_id, evolution_rule_plan_calls, evolution_llm_enrichment_calls,
            writer_model_calls, writer_tool_calls, writer_search_calls,
            orchestrator_{input,output,cached,reasoning}_tokens,
            writer_{input,output,cached,reasoning}_tokens,
            editor_{input,output,cached,reasoning}_tokens
```

> **缺口**：State 只跟踪 orchestrator / writer / editor 三角的 token，
> **缺 continuity / worldbuilding token 字段**。

#### Context 投影（`services/context.py` — `ContextCompiler`）

| 方法 | 消费者 | 投影内容 |
|---|---|---|
| `for_orchestrator` | Orchestrator | 2K chars + 1K world + 2K summary + 10 foreshadow + 8 events |
| `for_writer` | Writer | 5K//3 char + summary + 5 foreshadow + 5 events（world_context="" 空）|
| `for_extension` | Narrative Extension | 1.5K char + 3 foreshadow（无 world/summary/timeline）|
| `for_editor` | Editor | budget//2 char + summary + 3 foreshadow |
| `for_continuity` | Continuity | 10 events + 5 findings + 8 foreshadow + 4K//3 char |
| `apply_context_needed` | Orchestrator → 全局 | 按 context_needed 查 Storage 真实实体/事件 |

- `compile()` 已使用 `get_relevant_foreshadowings` + `get_relevant_story_events`（非全表扫描）。
- `apply_context_needed` 查 `get_entities_by_names` + `get_story_events_by_subjects`（真实检索）。

> **结论**：Production 的 Context 检索已是 task-aware（Phase 4 的 P1/P2/P3/P4/P5/P6
> 大部分已落地）。Plan 文档中假设的 `get_all_foreshadowings` / `get_all_world_entities`
> / `get_all_story_events` 在当前代码中**不存在**。

#### LLM 成本观测（State 内累计字段）

```text
orchestrator: input / output / cached / reasoning
writer:       input / output / cached / reasoning
editor:       input / output / cached / reasoning
```

> **缺口**：continuity / worldbuilding / evolution_orchestrator 的 token 未进 State。

---

### 1.2 Novel-Agent-Eval (Evaluation)

| 项 | 值 |
|---|---|
| 仓库路径 | `/Users/gaoyinrun/Desktop/qy/novel-agent-eval` |
| commit SHA | `dd5024d01b5e30951874e244bdace8aa8097875d` |
| 分支 | `master` |
| 工作区状态 | **有未提交改动**（见下） |
| Python | 3.12（`.python-version`） |
| 包管理 | uv（`uv.lock` 存在，717 KB） |
| pytest 收集 | **190 / 191 tests**（1 deselected） |
| ruff check | All checks passed |
| ruff format | **60 files would be reformatted**（未格式化） |

#### 未提交改动

```text
Modified:
  novel_agent_eval/agents/novel_agent.py   (+36/-? adapter 改动)
  tests/test_adapter_contract.py
  tests/test_agents.py
  uv.lock                                 (1527 行变化，lock 文件大幅分叉)
Untracked:
  novel_agent_eval/baseline.py
  scripts/smoke_phase2_baseline.py
  tests/test_baseline.py
  tests/test_baseline_run.py
```

> **风险**：基线不干净。adapter 有未提交改动 + 未跟踪的 baseline 新文件。
> Phase 0 建议先 commit 或 stash，否则后续 Phase 1 的改动无法与基线区分。

#### Benchmark Cases

| 数据集 | 路径 | 数量 | Stage |
|---|---|---|---|
| self_built | `dataset/self_built/*.json` | 13 | opening(4) / middle(5) / long(4) |
| eqbench | `dataset/eqbench/prompts.json` | 12 prompts | — |
| external/constory | `dataset/external/constory/` | 由 `fetch.py` 下载 | long |
| external/litbench | `dataset/external/litbench/` | 由 `fetch.py` 下载 | — |
| external/others | doc_re3 / longwriter / storybench | 空（.gitkeep） | — |

- `EvalCase.word_target` 默认 3000（self_built）；eqbench longform 用 1000。
- `EvalCase.ground_truth`（continuity_bugs / foreshadowings / outline_points）已填但**当前无消费者**。

#### Judge 模型

| 项 | 值 |
|---|---|
| 模型 | `step-3.7-flash`（StepFun，reasoning 模型） |
| 维度 | 8 维（consistency, writing, ai_flavor, dialogue, plot, instruction, creativity, controllability） |
| temperature | 0.0 |
| max_tokens | 8192 |
| reasoning_effort | `"low"` |
| n_samples | 默认 1，横评 3（中位数采样） |
| valid 标志 | `JudgeScore.valid`（解析失败/维度缺失 = False） |
| 兜底 | 缺失维度 → 0；overall 缺失 → 8 维平均 |

#### ConStory 一致性检测

| 项 | 值 |
|---|---|
| 模型 | `step-3.7-flash`（同 Judge） |
| 子类型 | 19 个 → 3 类（character / timeline / worldbuilding） |
| 评分 | `consistency_score = max(0, 100 - 20 * total_errors)` |
| 失败处理 | `failed_categories` 记录；失败时该类子类型 = 空列表 |
| 失败时评分 | `dims["consistency"] = 0`（runner.py:266）|

#### Planner 模型（EQBench Bridge）

| 项 | 值 |
|---|---|
| 模型 | `deepseek-v4-pro`（外部，BRIDGE_MODEL） |
| 协议 | DeepSeek API（独立 key/base_url） |
| 流程 | 5 步 planning（brainstorm → plan → critique → final_plan → characters） |
| temperature | 0.9 |
| thinking | disabled |
| 与 Judge 关系 | **不同模型族**（deepseek vs stepfun）|

#### Token Metrics

| 项 | 值 |
|---|---|
| STAGE_WEIGHTS | opening / middle / long 三档，9 维（8 质量 + efficiency） |
| weighted_score | 按 stage 加权求和 |
| efficiency_score | time(0.7) + rounds(0.3)；**tokens 字段未参与计算** |

#### Longform Flow

```text
EQBenchBridge.plan(writing_prompt)
  → plan_to_cases(plan) → 8 个 EvalCase
  → run_longform: 逐章 generate + judge.score_chapter
  → previous_context 逐章累积回填（全量正文拼接）
  → eqbench_chapter_score(14 维) → mean × 5 = 0-100
  → degradation = tail_window - head_window
  → checkpoint: save_chapter_text + _save_chapter_checkpoint（原子）
  → resume: _load_chapter_checkpoint 跳过已完成章
```

#### Resume Flow（durable.py）

```text
run_durable(adapter, cases, persist_dir, checkpoint_path, resume)
  → V2 durable state (Project/ChapterVersion/Canon) = 真相源
  → adapter.resume=True: 跨进程按 name 复用已有 project，已 approved 章节不重生
  → eval checkpoint: 补全进程内消亡的 token/hash/score（supplementary）
  → classify_outcome: completed / partial / failed / timeout / invalid
  → 失败章原子落盘后中止，已完章 checkpoint 保留
```

---

## 2. Production vs Eval 行为对照

### 2.1 scene_first 默认值

| 维度 | Production | Eval Adapter | 一致？ | 权威 |
|---|---|---|---|---|
| 默认 mode | Chapter-first（`scene_first` 未设） | `scene_first=True` | **不一致** | Production |

- `NovelAgentAdapter.__init__` 第 58 行：`scene_first: bool = True`。
- Production `orchestrator_node` 只在 `state.get("scene_first")` 为真时才 `build_scene_plan`。
- **影响**：Eval 默认走 scene-first 路径（逐 scene 生成 + assemble），Production 默认走
  chapter-first（单次生成）。两者 Writer 调用次数、token 消耗、叙事结构均不同。
- **对应 Task E6**。

### 2.2 previous_context 作为 memory

| 维度 | Production | Eval Adapter | 一致？ | 权威 |
|---|---|---|---|---|
| 前文记忆 | `ContextCompiler.compile()` 真实检索（Canon/SQLite/Memory） | `_truncate_previous_context`（head 300 + tail 1200）塞入 `recent_summary` | **部分一致** | Production |

- Adapter 在有 ProjectManager 时会 `_merge_context_packet`（eval 投影 + DB 投影合并），
  recent_summary 优先取 eval 完整前文。
- 但 eval 的 `_truncate_previous_context` 把万字前文截成 head+tail，这不等于 Production
  的 Memory（Chroma 语义检索 + SQLite Canon）。
- **对应 Task E7**。需区分 `eval_harness_context` 与 `production_context`。

### 2.3 Token 记账

| 维度 | Production State | Eval Adapter | 一致？ | 权威 |
|---|---|---|---|---|
| 覆盖角色 | orchestrator / writer / editor | orchestrator / writer / editor | 一致（都缺） | — |
| continuity / worldbuilding | **缺** | **缺** | 一致（都缺） | — |
| 求和方式 | per-role 累加 | input+output+cached+reasoning 求和 | **可能重复计算** | — |

- Eval `_extract_token_usage`：`for role in (orchestrator, writer, editor): for kind in (input, output, cached, reasoning): total += ...`
- 若 provider `cached ⊂ input` 且 `reasoning ⊂ output`，则四项求和会重复计算。
- **对应 Task E3 + E4**。Production 和 Eval 都需修复：补全角色 + 去重求和。

### 2.4 evolution_rounds 计数

| 维度 | Production | Eval Adapter | 一致？ | 权威 |
|---|---|---|---|---|
| 轮次计数 | `evolution_round`（0-based，每次 +1） | `len(evolution_history)` | **不一致** | Production |

- Adapter meta `evolution_rounds = len(history)`（history 每轮追加一个 entry）。
- Production 的 `evolution_round` 记实际 Writer 重写次数（首轮记录 v0 不算重写）。
- **对应 Task E5**。应区分 generation_round / revision_round / evolution_round。

### 2.5 Judge Failure 处理

| 维度 | 当前行为 | Plan 要求 | 状态 |
|---|---|---|---|
| judge 失败时 | `JudgeScore.valid=False`，runner 将 8 维清零 | `judge_status` 枚举 + `quality_scores=null` | **部分实现** |
| aggregate | valid=False 的 run 分数=0 参与统计 | 只统计 valid judge runs | **不符合** |
| judge_failure_rate | 未报告 | 需报告 | **缺失** |

- `JudgeScore.valid` 标志已存在（好的开始），但没有 `judge_status` 枚举。
- `runner._run_once`：`if not js.valid: dims.update({d: 0 ...})` — 失败直接变 0 分参与加权，
  虽然注释说"避免兜底分抬高总分"，但 0 分仍会拉低 aggregate。
- **对应 Task E1**。

### 2.6 ConStory Failure 处理

| 维度 | 当前行为 | Plan 要求 | 状态 |
|---|---|---|---|
| 失败时 | `dims["consistency"] = 0`（runner.py:266） | `consistency_score = null` | **不符合** |
| failed_categories | 已记录 | 需 `constory_status` 枚举 | **部分实现** |

- 当前：类别失败 → consistency = 0（极端低分），可能拉低 aggregate。
- Plan 要求：失败 → `consistency_score = null`，不参与统计，单独报告 `consistency_validity`。
- **对应 Task E2**。

### 2.7 External Planner 标记

| 维度 | 当前行为 | Plan 要求 | 状态 |
|---|---|---|---|
| planner_model | `deepseek-v4-pro`（BRIDGE_MODEL） | benchmark metadata 记录 | **未标记** |
| 报告声明 | 无 | 需声明"包含 external planner" | **缺失** |

- EQBench Bridge 用独立 LLM 做 5 步 planning，结果不是 Agent autonomous。
- **对应 Task E9**。

### 2.8 Judge / Planner Model 独立性

| 维度 | 当前 | Plan 要求 | 状态 |
|---|---|---|---|
| judge_model | step-3.7-flash | 可配置 | **已支持**（env + 构造注入） |
| planner_model | deepseek-v4-pro | 可配置 | **已支持**（env + 构造注入） |
| 实验矩阵 | — | same-family / cross-family | **未实现** |

- 两者已是不同模型族（好），但缺少正式实验矩阵 + model correlation risk 报告。
- **对应 Task E10**。

### 2.9 Context 检索（Production 侧已落地情况）

| Plan Task | 假设 Production 当前行为 | 实际 Production 行为 | 状态 |
|---|---|---|---|
| P1 context_needed → retrieval | 追加提示字符串 | `apply_context_needed` 真实检索 | **已实现** |
| P2 get_relevant_foreshadowings | get_all_foreshadowings | `get_relevant_foreshadowings` | **已实现** |
| P3 get_relevant_story_events | get_all_story_events | `get_relevant_story_events` | **已实现** |
| P4 get_relevant_world_entities | get_all_world_entities | `get_relevant_world_entities` | **已实现** |
| P5 删除大包后裁剪 | full packet → projection | compile + for_* 投影 + bound | **已实现** |
| P6 for_orchestrator() | 不存在 | 已存在 | **已实现** |

> **重大发现**：Plan 文档 Phase 4（P1-P6）假设的 Production 行为与当前代码不符。
> Production commit `a29be27` 已实现 task-aware retrieval。Plan 需据此修正，
> 避免重复实现已完成任务。

### 2.10 Mode-aware Output（Orchestrator）

| Plan Task | 假设 | 实际 | 状态 |
|---|---|---|---|
| P7 mode-aware output | 全字段输出 | orchestrator_node 已按 scene_first 分支 | **部分实现** |
| P8 scene_first=false 不生成 scene | — | build_scene_plan 只在 scene_first 时触发 | **已实现** |
| P9 context_needed 有实际效果 | 无效果 | apply_context_needed 改变 packet | **已实现** |

- Orchestrator 的 mode-aware 差异化输出（P7）需检查 OrchestratorAgent.analyze 是否按
  narrative_mode / scene_first 输出不同字段——**待 Phase 4 验证**。

### 2.11 Evolution 条件化 Reviewer

| Plan Task | 假设 | 实际 | 状态 |
|---|---|---|---|
| P10 按 plan 决定 reviewer | 全部跑 | `required_reviewers(plan)` + route_after_editor/continuity | **已实现** |
| P11 禁止每轮全跑 | — | `_reviewers_for(state)` 按 plan 路由 | **已实现** |
| P12 StyleAnalyzer deterministic | — | `StyleAnalyzer().analyze()` 0 LLM | **已实现** |

---

## 3. 不一致汇总与权威裁定

| # | 不一致项 | 权威方 | 对应 Task | 影响 Phase |
|---|---|---|---|---|
| 1 | scene_first 默认值 | Production | E6 | Phase 2 |
| 2 | previous_context 截断 ≠ Memory | Production | E7 | Phase 2 |
| 3 | Token 求和可能重复 | — | E3 | Phase 1 |
| 4 | Token 缺 continuity/worldbuilding | — | E4 | Phase 1 |
| 5 | evolution_rounds = len(history) | Production | E5 | Phase 1 |
| 6 | Judge failure → 0 分参与统计 | Plan | E1 | Phase 1 |
| 7 | ConStory failure → 0 分 | Plan | E2 | Phase 1 |
| 8 | External planner 未标记 | Plan | E9 | Phase 2 |
| 9 | P1-P6 假设过时 | Production 现状 | — | Phase 4 需修正 |
| 10 | Eval 工作区不干净 | — | — | 阻塞 Phase 1 |

---

## 4. Phase 0 结论

1. **Production（a29be27）基线可信**：干净、423 tests、ruff 全过。
   Plan Phase 4 的 P1-P6 已在 Production 落地，Plan 需据此修正。

2. **Eval（dd5024d）基线不可信**：有未提交改动 + 未跟踪文件 + 60 文件未格式化。
   **建议**：进入 Phase 1 前先 commit / stash 当前改动并跑 `ruff format`。

3. **最高优先级**（Phase 1）：E1-E5 修复 Eval 评分正确性，其中 E3/E4 同时影响
   Production（token 记账双方都缺）。

4. **Plan 文档需更新**：Phase 4（P1-P6）和 Phase 6（P10-P12）的多数任务
   已在 Production `a29be27` 实现，不应重复执行。
