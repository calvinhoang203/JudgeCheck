"""Unit tests for derived diagnostics."""

from __future__ import annotations

import unittest

import pandas as pd

from judgecheck.insights import (
    compare_recommended_sets,
    enrich_judge_abilities,
    item_discrimination_agreement,
    pairwise_tie_rates,
    pairwise_agreement_by_category,
    pairwise_winner_agreement,
    select_weak_items,
    summarize_judge_abilities,
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

    def test_pairwise_tie_rates(self) -> None:
        human = pd.DataFrame(
            [
                {"item_id": "1_t1", "winner": "tie"},
                {"item_id": "1_t1", "winner": "model_a"},
                {"item_id": "2_t1", "winner": "tie (inconsistent)"},
            ]
        )
        gpt4 = pd.DataFrame(
            [
                {"item_id": "1_t1", "winner": "model_a"},
                {"item_id": "2_t1", "winner": "tie"},
            ]
        )
        catalog = pd.DataFrame(
            [
                {"item_id": "1_t1", "category": "math", "category_label": "Math"},
                {"item_id": "2_t1", "category": "writing", "category_label": "Writing"},
            ]
        )
        overall, by_cat = pairwise_tie_rates(human, gpt4, catalog=catalog)
        human_row = overall.loc[overall["judge_system"] == "human_experts"].iloc[0]
        self.assertAlmostEqual(human_row["tie_rate"], 2 / 3)
        self.assertEqual(len(by_cat), 2)

    def test_summarize_judge_abilities(self) -> None:
        judges = pd.DataFrame(
            {
                "judge_id": ["j1", "j2", "j3"],
                "ability_theta": [1.0, 0.0, -1.0],
                "ability_rank": [1, 2, 3],
            }
        )
        summary = summarize_judge_abilities(judges)
        self.assertEqual(summary.loc[0, "n_judges"], 3)
        self.assertAlmostEqual(summary.loc[0, "mean_theta"], 0.0)
        self.assertEqual(summary.loc[0, "most_decisive_judge"], "j1")

    def test_enrich_judge_abilities(self) -> None:
        judges = pd.DataFrame(
            {"judge_id": ["j1"], "ability_theta": [0.5], "ability_rank": [1]}
        )
        human = pd.DataFrame(
            [
                {"judge": "j1", "item_id": "1_t1"},
                {"judge": "j1", "item_id": "2_t1"},
            ]
        )
        enriched = enrich_judge_abilities(judges, human)
        self.assertEqual(enriched.loc[0, "n_judgments"], 2)
        self.assertEqual(enriched.loc[0, "n_items"], 2)

    def test_item_discrimination_agreement(self) -> None:
        human = pd.DataFrame(
            {
                "item_id": ["a", "b", "c", "d"],
                "discrimination": [1.0, 0.8, 0.3, 0.1],
            }
        )
        gpt4 = pd.DataFrame(
            {
                "item_id": ["a", "b", "c", "d"],
                "discrimination": [3.0, 2.5, 1.0, 0.5],
            }
        )
        summary, detail = item_discrimination_agreement(human, gpt4, top_pct=0.25)
        self.assertEqual(summary.loc[0, "n_items"], 4)
        self.assertGreaterEqual(summary.loc[0, "n_both_sharp"], 1)
        self.assertIn("sharp_group", detail.columns)

    def test_pairwise_agreement_by_category_empty_catalog(self) -> None:
        merged = pd.DataFrame(
            columns=["item_id", "human_winner", "gpt4_winner", "agree"]
        )
        result = pairwise_agreement_by_category(merged, catalog=None)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
