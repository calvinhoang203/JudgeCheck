"""Unit tests for derived diagnostics."""

from __future__ import annotations

import unittest

import pandas as pd

from judgecheck.insights import (
    compare_recommended_sets,
    pairwise_agreement_by_category,
    pairwise_winner_agreement,
    select_weak_items,
)


class TestInsights(unittest.TestCase):
    def test_select_weak_items(self) -> None:
        params = pd.DataFrame(
            {
                "item_id": ["a", "b", "c"],
                "discrimination": [1.0, 0.2, 0.5],
            }
        )
        weak = select_weak_items(params, n=2)
        self.assertEqual(weak["item_id"].tolist(), ["b", "c"])

    def test_pairwise_winner_agreement(self) -> None:
        human = pd.DataFrame(
            [
                {"item_id": "1_t1", "model_a": "m1", "model_b": "m2", "winner": "model_a"},
                {"item_id": "1_t1", "model_a": "m1", "model_b": "m2", "winner": "model_a"},
                {"item_id": "2_t1", "model_a": "m1", "model_b": "m3", "winner": "model_b"},
            ]
        )
        gpt4 = pd.DataFrame(
            [
                {"item_id": "1_t1", "model_a": "m1", "model_b": "m2", "winner": "model_a"},
                {"item_id": "2_t1", "model_a": "m1", "model_b": "m3", "winner": "model_a"},
            ]
        )
        catalog = pd.DataFrame(
            [
                {"item_id": "1_t1", "category": "math", "category_label": "Math"},
                {"item_id": "2_t1", "category": "writing", "category_label": "Writing"},
            ]
        )

        overall, by_item, by_category = pairwise_winner_agreement(
            human, gpt4, catalog=catalog
        )

        self.assertEqual(overall.loc[0, "n_comparisons"], 2)
        self.assertAlmostEqual(overall.loc[0, "agreement_rate"], 0.5)
        self.assertEqual(len(by_item), 2)
        self.assertEqual(len(by_category), 2)
        self.assertEqual(by_category.iloc[0]["category_label"], "Writing")
        self.assertAlmostEqual(by_category.iloc[0]["agreement_rate"], 0.0)

    def test_compare_recommended_sets(self) -> None:
        pairwise = pd.DataFrame(
            {
                "item_id": ["1_t1", "2_t1", "3_t1"],
                "short_label": ["Q1", "Q2", "Q3"],
            }
        )
        score = pd.DataFrame(
            {
                "item_id": ["2_t1", "3_t1", "4_t1"],
                "short_label": ["Q2", "Q3", "Q4"],
            }
        )
        summary, detail = compare_recommended_sets(pairwise, score)
        self.assertEqual(summary.loc[0, "n_both"], 2)
        self.assertAlmostEqual(summary.loc[0, "jaccard"], 0.5)
        self.assertEqual(len(detail), 4)
        self.assertEqual(len(detail[detail["overlap_group"] == "both"]), 2)

    def test_pairwise_agreement_by_category_empty_catalog(self) -> None:
        merged = pd.DataFrame(
            columns=["item_id", "human_winner", "gpt4_winner", "agree"]
        )
        result = pairwise_agreement_by_category(merged, catalog=None)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
