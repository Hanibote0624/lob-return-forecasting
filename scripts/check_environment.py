#!/usr/bin/env python3
"""Print a version inventory; --gpu additionally requires a visible GPU."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime_environment import environment_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Optionally retain a private environment inventory."
    )
    args = parser.parse_args()
    report = environment_report(gpu=args.gpu, required=("numpy", "pandas", "scipy"))
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
