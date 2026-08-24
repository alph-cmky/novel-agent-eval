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


def start(args: argparse.Namespace) -> int:
    paths = _paths(args.output)
    if paths["status"].exists():
        current = json.loads(paths["status"].read_text())
        if current.get("status") == "running" and _alive(current.get("pid")):
            print(f"already running: pid={current['pid']}")
            return 1

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
    })
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
    if payload.get("status") in {"running", "stopping"} and not _alive(payload.get("pid")):
        payload["status"] = "stopped" if payload["status"] == "stopping" else "orphaned"
        payload["ended_at"] = _now()
        _write_json(paths["status"], payload)
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
    args = parser.parse_args()
    return {"start": start, "status": status, "stop": stop}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
