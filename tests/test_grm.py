"""Unit tests for GRM helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from judgecheck.grm import compare_category_discrimination


class TestGrm(unittest.TestCase):
    def test_compare_category_discrimination(self) -> None:
        human = pd.DataFrame(
            [
                {"category": "math", "category_label": "Math", "n_items": 10, "mean_discrimination": 0.8, "median_discrimination": 0.7},
                {"category": "writing", "category_label": "Writing", "n_items": 8, "mean_discrimination": 0.5, "median_discrimination": 0.4},
            ]
        )
        gpt4 = pd.DataFrame(
            [
                {"category": "math", "category_label": "Math", "n_items": 10, "mean_discrimination": 2.0, "median_discrimination": 1.9},
                {"category": "writing", "category_label": "Writing", "n_items": 8, "mean_discrimination": 2.5, "median_discrimination": 2.4},
            ]
        )

        result = compare_category_discrimination(human, gpt4)
        self.assertEqual(len(result), 2)
        math = result.loc[result["category"] == "math"].iloc[0]
        self.assertAlmostEqual(math["discrimination_gap"], -1.2)
        self.assertEqual(result.iloc[0]["category"], "writing")


if __name__ == "__main__":
    unittest.main()
