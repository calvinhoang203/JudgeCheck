#!/usr/bin/env python
"""
JudgeCheck main entry point.

Usage (from repo root, venv activated):
    python scripts/run_analysis.py              # full analysis (default)
    python scripts/run_analysis.py --pairwise-only
    python scripts/run_analysis.py --scores-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from judgecheck.pipeline import (  # noqa: E402
    _finalize_outputs,
    run_full_analysis,
    run_pairwise_analysis,
    run_score_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="JudgeCheck MT-Bench IRT analysis")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs",
        help="Output directory (default: outputs/)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pairwise-only",
        action="store_true",
        help="Run only pairwise human + GPT-4 analysis",
    )
    group.add_argument(
        "--scores-only",
        action="store_true",
        help="Run only GPT-4 single-score (1–10) analysis",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("JudgeCheck v0.3.1 — MT-Bench IRT analysis")
    print("=" * 60)

    if args.pairwise_only:
        print("\nMode: pairwise only")
        pairwise = run_pairwise_analysis(args.output)
        _finalize_outputs(args.output, pairwise, None)
    elif args.scores_only:
        print("\nMode: score ratings only")
        scores = run_score_analysis(args.output)
        _finalize_outputs(args.output, None, scores)
    else:
        print("\nMode: full (pairwise + score ratings)")
        run_full_analysis(args.output)

    print(f"\nDone. Outputs: {args.output.resolve()}")
    print(f"  Summary: {(args.output / 'SUMMARY.txt').resolve()}")
    if not args.scores_only:
        print(f"  Report:  {(args.output / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
