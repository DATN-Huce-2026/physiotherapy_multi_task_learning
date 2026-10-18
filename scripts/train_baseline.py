#!/usr/bin/env python3
"""Compatibility launcher for scripts/keraal/train_baseline.py."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "keraal" / "train_baseline.py"
    runpy.run_path(str(target), run_name="__main__")
