# scripts/export_chapters.py
"""从已完成的 baseline run 目录导出 chapters/chXX.md（不重新生成）。

用法：
  uv run python scripts/export_chapters.py --run-dir "/path/to/eval-data/<run_tag>"
"""
import argparse
import sys
from pathlib import Path

from novel_agent_eval.human_review import export_run_chapters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        print(f"run-dir 不存在: {args.run_dir}", file=sys.stderr)
        sys.exit(1)
    n = export_run_chapters(args.run_dir)
    print(f"exported {n} chapters → {args.run_dir / 'chapters'}")


if __name__ == "__main__":
    main()
