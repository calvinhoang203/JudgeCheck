"""Unit tests for metrics manifest export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import pandas as pd

from judgecheck.config import AnalysisConfig
from judgecheck.metrics import build_metrics_manifest, write_metrics_json


class TestMetrics(unittest.TestCase):
    def test_build_metrics_manifest_pairwise_only(self) -> None:
        class Pairwise:
            comparison = pd.DataFrame(
                [
                    {"judge_system": "human_experts", "mean_discrimination": 0.75},
                    {"judge_system": "gpt4_judge", "mean_discrimination": 2.5},
                ]
            )
            winner_agreement = pd.DataFrame([{"agreement_rate": 0.61}])
            tie_rates = pd.DataFrame(
                [
                    {"judge_system": "human_experts", "tie_rate": 0.1},
                    {"judge_system": "gpt4_judge", "tie_rate": 0.05},
                ]
            )
            judge_summary = None
            item_disc_agreement_summary = None
            recommended_pairwise_items = None
            human_information = None

        manifest = build_metrics_manifest(
            config=AnalysisConfig(),
            pairwise=Pairwise(),  # type: ignore[arg-type]
        )
        self.assertEqual(manifest["version"], "0.4.4")
        self.assertAlmostEqual(manifest["pairwise"]["human_mean_discrimination"], 0.75)
        self.assertAlmostEqual(manifest["pairwise"]["winner_agreement_rate"], 0.61)

    def test_write_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_metrics_json(Path(tmp), {"version": "0.4.2", "pairwise": {}})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["version"], "0.4.2")

    def test_compare_metrics_script(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compare_metrics",
            ROOT / "scripts" / "compare_metrics.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.json"
            b = Path(tmp) / "b.json"
            a.write_text(
                json.dumps({"pairwise": {"winner_agreement_rate": 0.6}}),
                encoding="utf-8",
            )
            b.write_text(
                json.dumps({"pairwise": {"winner_agreement_rate": 0.7}}),
                encoding="utf-8",
            )
            rows = mod.compare_metrics(a, b)
            self.assertEqual(len(rows), 1)
            self.assertIn("winner_agreement_rate", rows[0][0])


if __name__ == "__main__":
    unittest.main()
