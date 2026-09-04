# scripts/blind_review.py
"""本地盲测页：读 chapters/chXX.md，8 维打分，写入 annotations/human_scores.json。

默认不展示 Judge 分。提交后可用同一 run 的 baseline_result.json 算 kappa。

用法：
  uv run python scripts/blind_review.py --run-dir "/path/to/eval-data/<run_tag>"
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from novel_agent_eval.human_review import (
    calibrate_human_vs_judge,
    empty_human_scores,
    export_run_chapters,
    judge_dims_from_baseline,
    load_human_scores,
    save_human_scores,
)
from novel_agent_eval.judge import QUALITY_DIMS
from novel_agent_eval.report import DIM_LABELS

_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>盲测</title>
<style>
body { font-family: sans-serif; max-width: 960px; margin: 24px auto; color: #222; }
pre { white-space: pre-wrap; background: #f6f6f6; padding: 12px; }
.dim { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
.dim label { width: 7em; }
</style>
</head>
<body>
<h1>盲测打分</h1>
<p id="meta"></p>
<p>
  <button id="prev">上一章</button>
  <select id="picker"></select>
  <button id="next">下一章</button>
</p>
<article id="body"></article>
<form id="form"></form>
<p><button id="save" type="button">保存本章</button> <span id="status"></span></p>
<pre id="kappa"></pre>
<script>
const DIMS = %s;
const LABELS = %s;
let chapters = [];
let current = 0;
let payload = null;

async function load() {
  const res = await fetch("/api/state");
  const data = await res.json();
  chapters = data.chapters;
  payload = data.scores;
  document.getElementById("meta").textContent =
    "run " + data.run_tag + " · 共 " + chapters.length + " 章 · 不显示 Judge 分";
  const picker = document.getElementById("picker");
  picker.innerHTML = chapters.map((n, i) => `<option value="${i}">第${n}章</option>`).join("");
  picker.onchange = () => { current = Number(picker.value); render(); };
  document.getElementById("prev").onclick = () => { current = Math.max(0, current - 1); render(); };
  document.getElementById("next").onclick = () => { current = Math.min(chapters.length - 1, current + 1); render(); };
  document.getElementById("save").onclick = save;
  const form = document.getElementById("form");
  form.innerHTML = DIMS.map(d =>
    `<div class="dim"><label>${LABELS[d] || d}</label>` +
    `<input name="${d}" type="range" min="0" max="100" value="70"/>` +
    `<output id="o-${d}">70</output></div>`
  ).join("") + `<p>备注 <input name="notes" style="width:70%%"/></p>`;
  form.querySelectorAll("input[type=range]").forEach(el => {
    el.oninput = () => { document.getElementById("o-" + el.name).value = el.value; };
  });
  render();
  refreshKappa();
}

function rowFor(n) {
  return (payload.scores || []).find(s => s.chapter_number === n)
    || {chapter_number: n, dimensions: {}, notes: ""};
}

async function render() {
  const n = chapters[current];
  document.getElementById("picker").value = String(current);
  const md = await (await fetch("/api/chapter/" + n)).text();
  document.getElementById("body").innerHTML = "<pre></pre>";
  document.querySelector("#body pre").textContent = md;
  const row = rowFor(n);
  const form = document.getElementById("form");
  DIMS.forEach(d => {
    const v = row.dimensions[d];
    const input = form.querySelector(`[name="${d}"]`);
    input.value = (v === null || v === undefined) ? 70 : v;
    document.getElementById("o-" + d).value = input.value;
  });
  form.notes.value = row.notes || "";
}

async function save() {
  const n = chapters[current];
  const form = document.getElementById("form");
  const dimensions = {};
  DIMS.forEach(d => { dimensions[d] = Number(form.querySelector(`[name="${d}"]`).value); });
  const overall = Math.round(DIMS.reduce((s, d) => s + dimensions[d], 0) / DIMS.length);
  const idx = (payload.scores || []).findIndex(s => s.chapter_number === n);
  const row = {chapter_number: n, dimensions, overall, notes: form.notes.value};
  if (idx >= 0) payload.scores[idx] = row; else payload.scores.push(row);
  const res = await fetch("/api/scores", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  document.getElementById("status").textContent = res.ok ? "已保存" : "保存失败";
  refreshKappa();
}

async function refreshKappa() {
  const res = await fetch("/api/kappa");
  document.getElementById("kappa").textContent = JSON.stringify(await res.json(), null, 2);
}

load();
</script>
</body>
</html>
"""


def _scores_path(run_dir: Path) -> Path:
    return run_dir / "annotations" / "human_scores.json"


def _chapter_numbers(run_dir: Path) -> list[int]:
    files = sorted((run_dir / "chapters").glob("ch*.md"))
    numbers = []
    for path in files:
        try:
            numbers.append(int(path.stem[2:]))
        except ValueError:
            continue
    return numbers


def _ensure_scores(run_dir: Path) -> dict:
    path = _scores_path(run_dir)
    if path.exists():
        return load_human_scores(path)
    numbers = _chapter_numbers(run_dir)
    payload = empty_human_scores(
        run_tag=run_dir.name, rater_id="local", chapter_numbers=numbers
    )
    save_human_scores(path, payload)
    return payload


def _kappa(run_dir: Path, human: dict) -> dict:
    result_path = run_dir / "baseline_result.json"
    if not result_path.exists():
        return {"error": "missing baseline_result.json"}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return calibrate_human_vs_judge(human, judge_dims_from_baseline(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export", action="store_true", help="启动前先从 v2_state 导出 md")
    args = parser.parse_args()
    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"run-dir 不存在: {run_dir}", file=sys.stderr)
        sys.exit(1)
    if args.export or not (run_dir / "chapters").exists():
        n = export_run_chapters(run_dir)
        print(f"exported {n} chapters", flush=True)
    _ensure_scores(run_dir)

    labels = {d: DIM_LABELS.get(d, d) for d in QUALITY_DIMS}
    page = _PAGE % (json.dumps(QUALITY_DIMS), json.dumps(labels, ensure_ascii=False))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                payload = _ensure_scores(run_dir)
                body = json.dumps(
                    {
                        "run_tag": run_dir.name,
                        "chapters": _chapter_numbers(run_dir),
                        "scores": payload,
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if parsed.path.startswith("/api/chapter/"):
                try:
                    number = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    self._send(400, b"bad chapter", "text/plain")
                    return
                path = run_dir / "chapters" / f"ch{number:02d}.md"
                if not path.exists():
                    self._send(404, b"missing", "text/plain")
                    return
                self._send(200, path.read_bytes(), "text/plain; charset=utf-8")
                return
            if parsed.path == "/api/kappa":
                human = _ensure_scores(run_dir)
                body = json.dumps(_kappa(run_dir, human), ensure_ascii=False, indent=2).encode()
                self._send(200, body, "application/json; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/scores":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            save_human_scores(_scores_path(run_dir), payload)
            self._send(200, b'{"ok": true}', "application/json")

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"blind review → http://{args.host}:{args.port}  ({run_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
