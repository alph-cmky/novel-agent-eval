"""Start and monitor a detached long-form evaluation process."""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _paths(output: Path) -> dict[str, Path]:
    return {
        "status": output.with_name(f"{output.stem}.status.json"),
        "log": output.with_name(f"{output.stem}.log"),
        "pid": output.with_name(f"{output.stem}.pid"),
    }


def _progress(output: Path) -> dict:
    partial = output.with_name(f"{output.stem}.partial_results.json")
    failures = output.with_name(f"{output.stem}.failures.json")
    result = {"completed_samples": 0, "failures": 0, "chapters_saved": 0}
    if partial.exists():
        try:
            result["completed_samples"] = len(
                json.loads(partial.read_text()).get("results", [])
            )
        except (OSError, json.JSONDecodeError):
            pass
    if failures.exists():
        try:
            result["failures"] = len(
                json.loads(failures.read_text()).get("failures", [])
            )
        except (OSError, json.JSONDecodeError):
            pass
    chapter_root = output.with_name(f"{output.stem}.chapters")
    if chapter_root.exists():
        result["chapters_saved"] = len(list(chapter_root.rglob("chapter_*.txt")))
    return result


_PRESETS = {
    "smoke8": {
        "prompts": 1, "repeat": 1, "chapters": 8,
        "agents": "novel_agent", "max_rounds": 0,
    },
    "longform20": {
        "prompts": 2, "repeat": 1, "chapters": 20,
        "agents": "novel_agent,vanilla_llm", "max_rounds": 0,
    },
    "prompt_v2_20": {
        "prompts": 2, "repeat": 1, "chapters": 20,
        "agents": "novel_agent", "max_rounds": 0,
        "prompt_profile": "v2",
    },
    "consistency30": {
        "prompts": 6, "repeat": 3, "chapters": 30,
        "agents": "novel_agent,vanilla_llm", "max_rounds": 0,
    },
    "holdout50": {
        "prompts": 10, "repeat": 3, "chapters": 50,
        "agents": "novel_agent,vanilla_llm", "max_rounds": 1,
    },
}


def start(args: argparse.Namespace) -> int:
    preset = _PRESETS.get(args.preset, {})
    for key, value in preset.items():
        setattr(args, key, value)
    paths = _paths(args.output)
    if paths["status"].exists():
        current = json.loads(paths["status"].read_text())
        if current.get("status") == "running" and _alive(current.get("pid")):
            print(f"already running: pid={current['pid']}")
            return 1

    if args.mode == "horizontal":
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_horizontal_eval.py")),
            "--agents", args.agents,
            "--repeat", str(args.repeat),
            "--concurrency", str(args.concurrency),
            "--out", str(args.output),
        ]
        if args.resume:
            command.append("--resume")
    else:
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_eqbench_longform.py")),
        ]
    env = os.environ.copy()
    env.update({
        "N_PROMPTS": str(args.prompts),
        "REPEAT": str(args.repeat),
        "AGENTS": args.agents,
        "CONCURRENCY": str(args.concurrency),
        "STORY_TIMEOUT": str(args.story_timeout),
        "EQBENCH_OUT": str(args.output),
        "RESUME": "1" if args.resume else "0",
        "N_CHAPTERS": str(args.chapters),
        "CHAPTER_TIMEOUT": str(args.chapter_timeout),
        "BRIDGE_TIMEOUT": str(args.bridge_timeout),
        "NOVEL_MAX_ROUNDS": str(args.max_rounds),
        "NOVEL_WRITER_PROMPT_PROFILE": args.prompt_profile,
        "NOVEL_SKIP_ORCHESTRATOR": "1" if args.skip_orchestrator else "0",
        "NOVEL_SKIP_REVIEWS": "1" if args.skip_reviews else "0",
        "NOVEL_SKIP_ENRICHMENT": "1" if args.skip_enrichment else "0",
    })
    if args.skip_bridge:
        env["SKIP_BRIDGE"] = "1"
    if args.prompt_index is not None:
        env["PROMPT_INDEX"] = str(args.prompt_index)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    log = paths["log"].open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).parents[1],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    paths["pid"].write_text(str(process.pid) + "\n")
    _write_json(paths["status"], {
        "status": "running",
        "pid": process.pid,
        "started_at": _now(),
        "output": str(args.output),
        "log": str(paths["log"]),
        "config": {
            "prompts": args.prompts,
            "repeat": args.repeat,
            "agents": args.agents,
            "concurrency": args.concurrency,
            "story_timeout": args.story_timeout,
            "resume": args.resume,
            "chapters": args.chapters,
            "prompt_profile": args.prompt_profile,
            "skip_bridge": args.skip_bridge,
            "max_rounds": args.max_rounds,
            "chapter_timeout": args.chapter_timeout,
            "preset": args.preset,
            "mode": args.mode,
        },
    })
    print(f"started: pid={process.pid}")
    print(f"status: {paths['status']}")
    print(f"log: {paths['log']}")
    return 0


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def status(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    if not paths["status"].exists():
        print(json.dumps({"status": "not_started"}, ensure_ascii=False))
        return 0
    payload = json.loads(paths["status"].read_text())
    if payload.get("status") in {"running", "stopping", "orphaned"} and not _alive(payload.get("pid")):
        output = Path(payload.get("output", ""))
        if payload["status"] == "stopping":
            payload["status"] = "stopped"
        elif output.exists():
            payload["status"] = "completed"
        else:
            payload["status"] = "orphaned"
        payload["ended_at"] = _now()
        _write_json(paths["status"], payload)
    payload["progress"] = _progress(Path(payload.get("output", args.output)))
    if args.compact:
        print(json.dumps({
            "status": payload.get("status"),
            "pid": payload.get("pid"),
            "progress": payload["progress"],
            "output": payload.get("output"),
        }, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def stop(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    if not paths["status"].exists():
        print("not started")
        return 1
    payload = json.loads(paths["status"].read_text())
    pid = payload.get("pid")
    if not _alive(pid):
        payload["status"] = "stopped"
    else:
        os.killpg(int(pid), signal.SIGTERM)
        payload["status"] = "stopping"
    payload["ended_at"] = _now()
    _write_json(paths["status"], payload)
    print(f"{payload['status']}: pid={pid}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--agents", default="novel_agent,vanilla_llm")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--story-timeout", type=int, default=3600)
    parser.add_argument("--chapters", type=int, default=8)
    parser.add_argument("--chapter-timeout", type=int, default=600)
    parser.add_argument("--bridge-timeout", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--prompt-profile", default="v1")
    parser.add_argument("--prompt-index", type=int, default=None)
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument("--skip-orchestrator", action="store_true", default=True)
    parser.add_argument("--skip-reviews", action="store_true")
    parser.add_argument("--skip-enrichment", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--preset", choices=tuple(_PRESETS), default=None)
    parser.add_argument("--mode", choices=("longform", "horizontal"), default="longform")
    args = parser.parse_args()
    return {"start": start, "status": status, "stop": stop}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
