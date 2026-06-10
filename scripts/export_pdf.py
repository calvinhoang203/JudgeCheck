#!/usr/bin/env python
"""
Rebuild report.pdf from existing pipeline outputs (no re-analysis).

Usage:
    python scripts/export_pdf.py
    python scripts/export_pdf.py outputs --mode full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from judgecheck.pdf_report import ReportMode, generate_pdf_report  # noqa: E402


def _detect_mode(output_dir: Path) -> ReportMode:
    has_pairwise = (output_dir / "human_item_parameters.csv").exists()
    has_scores = (output_dir / "score_item_parameters.csv").exists()
    if has_pairwise and has_scores:
        return "full"
    if has_pairwise:
        return "pairwise"
    return "scores"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export JudgeCheck report.pdf")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=ROOT / "outputs",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "pairwise", "scores", "auto"],
        default="auto",
        help="Which figures to include (default: auto-detect)",
    )
    args = parser.parse_args()

    mode: ReportMode = _detect_mode(args.output) if args.mode == "auto" else args.mode
    path = generate_pdf_report(args.output, mode=mode)
    if path is None:
        print("Nothing to export — need SUMMARY.txt or PNG figures in the output folder.")
        sys.exit(1)
    print(f"Wrote {path.resolve()} (mode: {mode})")


if __name__ == "__main__":
    main()
