# novel-agent-eval

`novel-agent` 的评测框架（独立仓库）。评测对象是**生成章节的质量**（横评 / 消融 / 一致性检测），不是训练模型。`novel-agent` 以 editable 依赖引入，是本框架的「信号源」。

## 启动

```
uv sync                                       # 装依赖（含 ../novel-agent editable）
uv run pytest                                 # 单元/集成测试（mock client，不耗真实 API）
uv run ruff check .                           # lint（自动排除 vendor/）
STEPFUN_API_KEY=... REPEAT=3 uv run python scripts/run_horizontal_eval.py   # 横评
```

横评/探针脚本在 `scripts/`，不进 pytest/CI。

## 技术栈

Python 3.12+, Pydantic（schema）, OpenAI async client（StepFun）, pytest + ruff, uv。依赖见 `pyproject.toml`；`novel-agent` 的 editable 路径在 `[tool.uv.sources]`。

## 架构（数据流）

```
EvalCase ──> AgentAdapter.generate ──> Judge.score(8 质量维) ──> weighted_score(stage 加权) ──> BenchmarkRunner 聚合 mean±std ──> report 跑分卡
```

三层判分，各管一件事：

- **Judge**（`judge.py`）：外部 LLM-judge，8 质量维 + overall（StepFun 单模型）。
- **ConStory-Checker**（`constory.py`）：evidence-grounded 一致性检测，覆盖 Judge 的 consistency 维（`consistency_score = 100 - 20·errors`）。
- **内部信号**（`internal_signals.py`）：复用主仓库 `composite_score`/`extract_scores`，供「自评分 vs 外评分」一致性分析。

## 环境变量（三组，别混）

- **Judge** 读 `STEPFUN_API_KEY` / `STEPFUN_BASE_URL` / `STEPFUN_JUDGE_MODEL`。
- **novel-agent adapter** 读 `QUALITY`/`BUDGET_MODEL` + `OPENAI_*`（走主仓库 `ModelRouter`）。
- **vanilla 基线** 读 `BASELINE_*`。

`run_horizontal_eval.py` 把三者统一指向 StepFun `step-3.7-flash` 并注入 `*_IS_REASONING=true`（见脚本头部，照抄即可）。

## 项目约束

- `STEPFUN_API_KEY` 是凭证：只经 env 传，**绝不写进文件/日志/commit**。
- 测试用 mock client，不耗真实 API；需真实 LLM 的集成测试标 `@pytest.mark.slow`。
- 不直接操作主仓库内部状态，走 `agents/` 下的 adapter。
- 复用主仓库评分公式（`composite_score`/`extract_scores`），**不重复实现**。
- `novel_agent_eval/vendor/constory` 是官方复制的第三方代码：不改、不 lint（ruff 已 exclude）。
- 不用 `git add -A`，只 add 指定文件；不为单次使用引入新依赖。

## 关键约定

### 数据
- 评测集 `dataset/self_built/*.json`，`load_cases()` 读目录；`Stage` 三档 `opening`/`middle`/`long`。
- `EvalCase.ground_truth`（`continuity_bugs`/`foreshadowings`/`outline_points`）**已填但当前无消费者** —— 对账留待「Judge 校准」改进，见 `docs/评测改进计划.md`。

### 打分
- `weighted_score` 按 stage 加权（`metrics.STAGE_WEIGHTS`）；`CORE_WEIGHTS` 当前未用。
- **repeat 的 std 语义别混**：`BenchmarkResult.overall_std` = 同一 case 重跑方差（σ_gen + σ_judge）；`ComparisonReport.overall_std` = 跨 case 方差。判断「A vs B 差值是否有意义」用前者。
- Judge `n_samples` 中位数采样：默认 1，横评脚本用 3（抑制 consistency 维偶发极端分）。
- `Judge._to_score` 兜底：缺失维给 0，overall 缺失退化为 8 维平均。

### Judge 校准（规划中，未实现）
- 校准 Judge 时用 **Cohen's kappa**，不用 raw agreement —— 两个 judge 都 ~97% 一致率也可能 kappa≈0（毫无区分力）。
- 「差值有意义」= `d > MDD ≈ z·sqrt(σ_gen²/n + σ_judge²)`；三件套（repeat 均值 / 数据集提难度 / Judge 校准）的排期见 `docs/评测改进计划.md`。

## 已知坑

- **LLM 输出必须先过 `_strip_none`**：`dict.get(key, default)` 在键存在但值为 None 时返回 None，下游 `.get(...)` 直接崩溃。
- **`step-3.7-flash` 是 reasoning 模型**：`max_tokens=8192` + `reasoning_effort="low"` 缺一不可，否则 content 被 reasoning 挤空。
- **主仓库 `build_chapter_graph_async` 非空 `persist_dir`** 用 aiosqlite 非 daemon 线程，短生命周期进程必须 `aclose_checkpointers()` 收口，否则退出挂死。
- `run_horizontal_eval.py` 默认 `REPEAT=1`（待改 3，见计划文档改进一）。
