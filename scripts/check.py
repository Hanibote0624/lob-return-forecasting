#!/usr/bin/env python3
"""Run the same lightweight correctness checks used in GitHub CI."""

from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.configuration import load_config, validate_config


def main():
    # Compile without importing optional dependencies or creating bytecode files.
    sources = [
        path for folder in ("src", "scripts", "tests", "model_tests") for path in (ROOT / folder).rglob("*.py")
    ]
    for path in sources:
        compile(path.read_bytes(), str(path), "exec")
    with (ROOT / "pyproject.toml").open("rb") as handle:
        tomllib.load(handle)
    examples = sorted((ROOT / "config").glob("*.example.json"))
    if not examples:
        raise RuntimeError("no public configuration examples found")
    for path in examples:
        validate_config(load_config(path))
    print(
        f"Syntax/configuration checks passed: {len(sources)} Python files, {len(examples)} example(s).",
        flush=True,
    )
    for command in (
        [sys.executable, "-m", "pip", "check"],
        [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests", "model_tests"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    ):
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
