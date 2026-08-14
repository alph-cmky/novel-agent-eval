# tests/test_fetch.py
"""fetch mock 测试：不真实下载 HF 数据集、不联网。"""
import json

import pandas as pd

from novel_agent_eval.dataset.fetch import (
    download_constory_prompts,
    download_litbench_train,
)


def test_download_constory_prompts_filters_and_writes(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"id": 1, "language": "zh", "task_type": "generation", "prompt": "长篇生成A"},
            {"id": 2, "language": "zh", "task_type": "continuation", "prompt": "长篇续写B"},
            {"id": 3, "language": "en", "task_type": "expansion", "prompt": "短篇扩展C"},
            {"id": 4, "language": "en", "task_type": "completion", "prompt": "短篇完形D"},
        ]
    )
    parquet_path = tmp_path / "prompts.parquet"
    df.to_parquet(parquet_path)

    calls = {}

    def fake_hf_hub_download(**kwargs):
        calls.update(kwargs)
        return str(parquet_path)

    monkeypatch.setattr(
        "novel_agent_eval.dataset.fetch.hf_hub_download", fake_hf_hub_download
    )

    out_dir = tmp_path / "external"
    result = download_constory_prompts(out_dir)

    assert calls["repo_id"] == "jayden8888/ConStory-Bench"
    assert calls["filename"] == "prompts.parquet"
    assert calls["repo_type"] == "dataset"
    assert calls["endpoint"] == "https://hf-mirror.com"

    assert result == out_dir / "constory_prompts_longform.json"
    records = json.loads(result.read_text(encoding="utf-8"))
    # 只保留长篇子集（generation / continuation），字段 4 列齐全
    assert [r["id"] for r in records] == [1, 2]
    assert all(set(r) == {"id", "language", "task_type", "prompt"} for r in records)
    assert records[0]["prompt"] == "长篇生成A"  # ensure_ascii=False


def test_download_constory_prompts_custom_task_types(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"id": 1, "language": "zh", "task_type": "generation", "prompt": "A"},
            {"id": 2, "language": "zh", "task_type": "completion", "prompt": "D"},
        ]
    )
    parquet_path = tmp_path / "prompts.parquet"
    df.to_parquet(parquet_path)

    monkeypatch.setattr(
        "novel_agent_eval.dataset.fetch.hf_hub_download",
        lambda **kwargs: str(parquet_path),
    )

    out_dir = tmp_path / "external"
    result = download_constory_prompts(out_dir, task_types=("completion",))
    records = json.loads(result.read_text(encoding="utf-8"))
    assert [r["id"] for r in records] == [2]


def test_download_litbench_train_passes_args(tmp_path, monkeypatch):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path / "external" / "litbench")

    monkeypatch.setattr(
        "novel_agent_eval.dataset.fetch.snapshot_download", fake_snapshot_download
    )

    out_dir = tmp_path / "external"
    result = download_litbench_train(out_dir)

    assert calls["repo_id"] == "euclaise/WritingPrompts_preferences"
    assert calls["repo_type"] == "dataset"
    assert calls["endpoint"] == "https://hf-mirror.com"
    assert calls["local_dir"] == out_dir / "litbench"
    assert result == out_dir / "litbench"
