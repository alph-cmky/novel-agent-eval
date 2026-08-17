# novel-agent-eval

`novel-agent` 的长篇文本 Agent 评测框架，负责生成质量评测、对照实验和问题诊断，
不负责训练模型。

## 能力范围

- 自建 hard cases：覆盖 `opening`、`middle`、`long` 三个阶段。
- 单章横评：`novel-agent` 与 Vanilla LLM 基线。
- LLM-as-a-Judge：8 个质量维度，按 stage 加权聚合。
- ConStory 一致性检测：角色、时间线和世界观三类错误。
- 重复采样：输出均值和标准差；默认脚本的 `REPEAT` 仍为 1。
- EQ-Bench Longform：官方 5 步 planning、8 章连续生成和 14 维逐章评分。
- 对手适配器：部分适配器是真实外部 CLI 接入，部分是用于方法对照的本地策略模拟，见下文。

## 仓库关系

评测仓库通过 editable dependency 使用 `novel-agent`。本地开发时需要并列放置两个仓库：

```text
qy/
├── novel-agent/
└── novel-agent-eval/
```

```bash
cd novel-agent-eval
uv sync
uv run pytest
uv run ruff check .
```

当前评测框架依赖主仓库的本地路径，不是独立发布包。公开使用时请同时 clone 两个仓库，
或将 `pyproject.toml` 中的 source 改为已发布的 Git revision。

## 运行评测

测试用例默认使用 mock，不消耗真实 API。真实横评需要用户自行配置 API Key：

```bash
STEPFUN_API_KEY=... \
REPEAT=1 \
uv run python scripts/run_horizontal_eval.py \
  --agents novel_agent,vanilla_llm \
  --out /tmp/horizontal_eval.json
```

EQ-Bench Longform 需要独立的 bridge/judge 配置：

```bash
DEEPSEEK_API_KEY=... STEPFUN_API_KEY=... \
uv run python scripts/run_eqbench_longform.py
```

不要把 API Key 写入文件、报告或提交记录。

## 评测边界

- 自建集是 13 个带角色、时间线和世界观冲突的单章 hard cases，不等同于 50～100 章真实连续文本。
- `ground_truth` 字段已定义，但当前没有用于计算伏笔回收率、大纲覆盖率或 bug precision/recall。
- `NovelAgentAdapter` 为隔离评测使用空 `project_id`，不会读取真实项目数据库和完整向量记忆；结果不能直接代表生产项目的完整长程记忆能力。
- 主仓库已移除非进化路径，`evolution_enabled` 参数已弃用（保留仅为兼容旧调用）；评测侧不再声称能做进化开关消融。
- Judge 尚未完成人工标注校准；相关系数和单轮结果不应被解释为统计显著性证明。
- `tokens` 在部分 adapter 中不可用，效率分主要由耗时和进化轮次计算。

## Adapter 说明

- `NovelAgentAdapter`：调用本地 `novel-agent` StateGraph。
- `VanillaLLMAdapter`：一次 Prompt 的单模型基线。
- `InkOSAdapter`、`NovelWritingAgentAdapter`：外部 CLI 接入，依赖本地安装和额外配置。
- `StoryDiffusionAdapter`、`NovelForgeAdapter`、`StoryBibleAdapter`：本仓库实现的策略模拟器，
  不是对应开源项目的官方运行结果，不能用于宣称对这些项目完成了真实横评。

## 数据与第三方资源

- `dataset/self_built/` 是本项目自建案例。
- EQ-Bench 和 ConStory 的 prompt、规则及代码来自上游项目，版权归属与许可证见
  `THIRD_PARTY_NOTICES.md`。
- 外部数据下载脚本只下载到本地，不应把未获授权的数据集提交到仓库。

## 结果报告

本仓库不分发历史实验报告与原始分数归档；它们仅保存在作者本地作为实验记录。
公开仓库只提供可复现的评测代码、自建数据集与第三方评测接入逻辑，不对任何 Agent
做出排名结论。

## License

MIT。第三方代码和评测资源以其上游许可证为准。
