"""Audit local open benchmark assets without making any API calls."""

import argparse
import json
from pathlib import Path

from novel_agent_eval.dataset.open_benchmarks import audit_open_assets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = json.dumps(audit_open_assets(), ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
