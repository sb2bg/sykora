#!/usr/bin/env python3
"""Compatibility wrapper for moved script."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "match" / "selfplay.py"
    runpy.run_path(str(target), run_name="__main__")

