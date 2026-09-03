# scripts/restore_evolution_paths.py
"""从 V2 DB（checkpoints.db writes 表）重建 evolution paths。

用途：run 中断后多次 resume 时，baseline_result.json 的 evolution.paths 只
保留最后一批新生成章节的路径（前序章节从 V2 DB 恢复，不经过 adapter 重放）。
本脚本重放 checkpoints.db 的 writes 流，重建每章完整的 evolution history。

前提：v2_state/novel.db 的 project 未被清空（writes 按 thread 追加，不删除）。

用法：
  uv run python scripts/restore_evolution_paths.py <run_dir_name>

输出：<run_dir>/evolution_paths_restored.json
  [{"chapter": int, "path": [...], "gains": [...], "best_version": int}, ...]
"""
import json
import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

VAULT_EVAL_DATA = Path(
    "/Users/gaoyinrun/Documents/Obsidian Vault/novel-agent/eval-data"
)

# 新版 editor 8 维标记：跳过旧 5 维 GLM 残留 thread
_NEW_DIMS_MARKER = "consistency"


def _restore_run(run_dir: str) -> list[dict]:
    db = VAULT_EVAL_DATA / run_dir / "v2_state" / "checkpoints.db"
    if not db.exists():
        raise FileNotFoundError(f"checkpoint DB 不存在: {db}")

    ser = JsonPlusSerializer()
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()

    threads = [t[0] for t in cur.execute(
        "SELECT DISTINCT thread_id FROM checkpoints"
    ).fetchall()]

    paths: list[dict] = []
    for tid in threads:
        rows = cur.execute(
            "SELECT checkpoint_id, channel, value, type FROM writes "
            "WHERE thread_id=? ORDER BY checkpoint_id, idx",
            (tid,),
        ).fetchall()

        history: list[dict] = []
        best_version = 0
        chapter_number = None
        for _, channel, val, vtype in rows:
            try:
                data = ser.loads_typed((vtype, val))
            except Exception:
                continue
            if channel == "evolution_history":
                history = data  # 每轮全量替换
            elif channel == "evolution_best_candidate_version":
                best_version = data
            elif channel == "chapter_number":
                chapter_number = data

        if not history or chapter_number is None:
            continue
        dims = history[-1].get("dimensions", {})
        if _NEW_DIMS_MARKER not in dims:
            continue  # 旧 5 维残留 thread，跳过

        path = [
            {
                "revision": e.get("v", 0),
                "composite": e.get("composite"),
                "editor": e.get("editor"),
                "continuity": e.get("continuity"),
                "delta": e.get("delta"),
                "focus": e.get("focus"),
                "reviewers": e.get("reviewers"),
                "writer_tokens": e.get("writer_tokens"),
            }
            for e in history
        ]
        gains = []
        for prev, curr in zip(history, history[1:]):
            gains.append({
                "chapter": chapter_number,
                "from_v": prev.get("v"),
                "to_v": curr.get("v"),
                "focus": curr.get("focus"),
                "quality_gain": round(
                    (curr.get("composite") or 0) - (prev.get("composite") or 0), 1
                ),
                "writer_input_delta": (curr.get("writer_tokens") or {}).get("input", 0)
                - (prev.get("writer_tokens") or {}).get("input", 0),
                "writer_output_delta": (curr.get("writer_tokens") or {}).get("output", 0)
                - (prev.get("writer_tokens") or {}).get("output", 0),
            })

        paths.append({
            "chapter": chapter_number,
            "path": path,
            "gains": gains,
            "best_version": best_version,
        })

    conn.close()
    paths.sort(key=lambda p: p["chapter"])
    return paths


def main() -> None:
    if len(sys.argv) != 2:
        print("用法: uv run python scripts/restore_evolution_paths.py <run_dir_name>")
        sys.exit(1)
    run_dir = sys.argv[1]
    paths = _restore_run(run_dir)

    out = VAULT_EVAL_DATA / run_dir / "evolution_paths_restored.json"
    out.write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"恢复 paths: {len(paths)}/20")
    for p in paths:
        n = len(p["path"])
        v0 = p["path"][0]["composite"]
        if n > 1:
            v1 = p["path"][-1]["composite"]
            print(f"  ch{p['chapter']}: revs={n} v0={v0} gain={v1-v0:+.1f}")
        else:
            print(f"  ch{p['chapter']}: revs=1 v0={v0} 跳过/无重写")
    print(f"保存 → {out}")


if __name__ == "__main__":
    main()
