#!/usr/bin/env python
"""
Backward-compatible alias for ``scripts/run_analysis.py``.

Usage:
    python scripts/fit_grm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from judgecheck.pipeline import run_full_analysis  # noqa: E402


def main() -> None:
    out = ROOT / "outputs"
    run_full_analysis(out)


if __name__ == "__main__":
    main()
