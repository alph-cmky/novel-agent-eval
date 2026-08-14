# novel_agent_eval/dataset/fetch.py
"""下载 ConStory-Bench 长篇子集 + LitBench 训练集到 dataset/external/（不提交 git）。"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

_MIRROR = "https://hf-mirror.com"

_CONSTORY_REPO = "jayden8888/ConStory-Bench"
_LITBENCH_REPO = "euclaise/WritingPrompts_preferences"


def download_constory_prompts(
    output_dir: Path, task_types: tuple[str, ...] = ("generation", "continuation")
) -> Path:
    """hf_hub_download 拉 prompts.parquet，pandas 读 + 筛 task_type，落盘 JSON。

    返回 output_dir/constory_prompts_longform.json（内容为 list[{id, language, task_type, prompt}]）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = hf_hub_download(
        repo_id=_CONSTORY_REPO,
        filename="prompts.parquet",
        repo_type="dataset",
        endpoint=_MIRROR,
    )
    df = pd.read_parquet(parquet_path)
    filtered = df[df["task_type"].isin(task_types)]
    records = filtered.to_dict("records")
    out = output_dir / "constory_prompts_longform.json"
    out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return out


def download_litbench_train(output_dir: Path) -> Path:
    """下载 euclaise/WritingPrompts_preferences 训练集到 output_dir/litbench/。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = output_dir / "litbench"
    snapshot_download(
        repo_id=_LITBENCH_REPO,
        repo_type="dataset",
        endpoint=_MIRROR,
        local_dir=local_dir,
    )
    return local_dir
