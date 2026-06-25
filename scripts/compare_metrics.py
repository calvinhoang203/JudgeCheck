#!/usr/bin/env python
"""
Compare two JudgeCheck metrics.json files (e.g. different coverage settings).

Usage:
    python scripts/compare_metrics.py outputs/run_a/metrics.json outputs/run_b/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _flatten(obj: object, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, path))
    else:
        flat[prefix] = obj
    return flat


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def compare_metrics(path_a: Path, path_b: Path) -> list[tuple[str, str, str, str]]:
    a = _flatten(json.loads(path_a.read_text(encoding="utf-8")))
    b = _flatten(json.loads(path_b.read_text(encoding="utf-8")))
    keys = sorted(set(a) | set(b))
    rows: list[tuple[str, str, str, str]] = []
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if va == vb:
            continue
        rows.append((key, _format(va) if va is not None else "—", _format(vb) if vb is not None else "—", "≠"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two JudgeCheck metrics.json files")
    parser.add_argument("metrics_a", type=Path, help="First metrics.json")
    parser.add_argument("metrics_b", type=Path, help="Second metrics.json")
    args = parser.parse_args()

    rows = compare_metrics(args.metrics_a, args.metrics_b)
    if not rows:
        print("No differences.")
        return

    width = max(len(r[0]) for r in rows)
    print(f"{'metric':<{width}}  {'A':>12}  {'B':>12}")
    print("-" * (width + 28))
    for key, va, vb, _ in rows:
        print(f"{key:<{width}}  {va:>12}  {vb:>12}")


if __name__ == "__main__":
    main()
