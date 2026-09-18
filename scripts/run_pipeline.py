#!/usr/bin/env python3
"""Run the checkout's pipeline from any working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
