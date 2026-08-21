from novel_agent_eval.manifest import build_run_manifest


def test_manifest_contains_reproducibility_hashes_without_credentials(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text('{"1": {"title": "T"}}', encoding="utf-8")

    manifest = build_run_manifest(
        {"model": "test-model", "api_key": "must-not-be-copied"},
        prompts,
    )

    assert manifest["prompt_hash"]
    assert manifest["dataset_hash"]
    assert manifest["config"]["model"] == "test-model"
    assert "STEPFUN_API_KEY" not in str(manifest)
