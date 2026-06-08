"""Unit tests for data loading helpers (no network)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from judgecheck.data import WINNER_TO_ORDINAL, prepare_grm_matrix


class TestData(unittest.TestCase):
    def test_winner_ordinal_mapping(self) -> None:
        self.assertEqual(WINNER_TO_ORDINAL["model_a"], 3)
        self.assertEqual(WINNER_TO_ORDINAL["tie"], 2)
        self.assertEqual(WINNER_TO_ORDINAL["model_b"], 1)

    def test_prepare_grm_matrix_shape(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "item_id": "1_t1",
                    "judge": "j1",
                    "ordinal": 3,
                    "question_id": 1,
                    "turn": 1,
                },
                {
                    "item_id": "1_t1",
                    "judge": "j2",
                    "ordinal": 1,
                    "question_id": 1,
                    "turn": 1,
                },
                {
                    "item_id": "2_t1",
                    "judge": "j1",
                    "ordinal": 2,
                    "question_id": 2,
                    "turn": 1,
                },
            ]
        )
        matrix, item_ids, judge_ids = prepare_grm_matrix(df)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(item_ids, ["1_t1", "2_t1"])
        self.assertEqual(sorted(judge_ids), ["j1", "j2"])
        self.assertTrue(np.isnan(matrix[1, 1]))


if __name__ == "__main__":
    unittest.main()
