"""
Graded Response Model (GRM) fitting wrappers around girth.

In JudgeCheck framing:
  - Items  = MT-Bench questions (how discriminating is each benchmark item?)
  - People = human annotators or GPT-4 comparison slots (latent judging ability θ)
  - Ratings = ordinal preference from pairwise comparisons (1–3 scale here)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from girth import grm_mml, tag_missing_data
from scipy import stats

INVALID_FILL = -9999
VALID_RESPONSES = [1, 2, 3]


@dataclass
class GRMResults:
    """Container for a fitted Graded Response Model."""

    discrimination: np.ndarray
    difficulty: np.ndarray
    item_ids: list[str]
    participant_ids: list[str]
    judge_label: str
    aic: dict[str, float]
    bic: dict[str, float]
    n_items: int
    n_participants: int
    n_responses: int

    @property
    def mean_discrimination(self) -> float:
        return float(np.mean(self.discrimination))

    @property
    def reliability_proxy(self) -> float:
        """
        Simple reliability summary: mean item discrimination.

        Higher average discrimination suggests the judge/system more consistently
        separates strong vs weak responses across benchmark items.
        """
        return self.mean_discrimination


def _tag_matrix(matrix: np.ndarray) -> np.ndarray:
    tagged = tag_missing_data(matrix, VALID_RESPONSES)
    tagged = np.where(np.isnan(tagged), INVALID_FILL, tagged)
    return tagged.astype(int)


def fit_grm(
    matrix: np.ndarray,
    item_ids: list[str],
    participant_ids: list[str],
    *,
    judge_label: str = "judge",
    options: dict[str, Any] | None = None,
) -> GRMResults:
    """
    Fit a Graded Response Model via marginal maximum likelihood (girth).

    Parameters
    ----------
    matrix : array (n_items, n_participants)
        Ordinal responses; NaN for missing judgments.
    """
    tagged = _tag_matrix(matrix)
    estimates = grm_mml(tagged, options=options or {})

    n_observed = int(np.sum(np.isin(tagged, VALID_RESPONSES)))

    return GRMResults(
        discrimination=np.asarray(estimates["Discrimination"]),
        difficulty=np.asarray(estimates["Difficulty"]),
        item_ids=item_ids,
        participant_ids=participant_ids,
        judge_label=judge_label,
        aic=estimates["AIC"],
        bic=estimates["BIC"],
        n_items=len(item_ids),
        n_participants=len(participant_ids),
        n_responses=n_observed,
    )


def grm_results_to_frame(results: GRMResults) -> pd.DataFrame:
    """Convert GRM item parameters to a tidy DataFrame."""
    n_thresholds = results.difficulty.shape[1]
    threshold_cols = [f"threshold_{k + 1}" for k in range(n_thresholds)]

    frame = pd.DataFrame(
        {
            "item_id": results.item_ids,
            "discrimination": results.discrimination,
            **{
                col: results.difficulty[:, i]
                for i, col in enumerate(threshold_cols)
            },
        }
    )
    frame["question_id"] = frame["item_id"].str.extract(r"^(\d+)_")[0].astype(int)
    frame["turn"] = frame["item_id"].str.extract(r"_t(\d+)$")[0].astype(int)
    return frame.sort_values("discrimination", ascending=False).reset_index(drop=True)


def compare_judges(
    human: GRMResults,
    gpt4: GRMResults,
) -> pd.DataFrame:
    """Compare reliability metrics between two fitted judge systems."""
    rows = []
    for label, res in [("human_experts", human), ("gpt4_judge", gpt4)]:
        rows.append(
            {
                "judge_system": label,
                "mean_discrimination": res.mean_discrimination,
                "median_discrimination": float(np.median(res.discrimination)),
                "n_items": res.n_items,
                "n_participants": res.n_participants,
                "n_responses": res.n_responses,
                "aic": (res.aic or {}).get("final_model"),
                "bic": (res.bic or {}).get("final_model"),
            }
        )

    comparison = pd.DataFrame(rows)

    # Correlation of item discriminations (aligned items only).
    h = grm_results_to_frame(human).set_index("item_id")["discrimination"]
    g = grm_results_to_frame(gpt4).set_index("item_id")["discrimination"]
    aligned = pd.concat([h, g], axis=1, keys=["human", "gpt4"]).dropna()
    if len(aligned) >= 3:
        rho, pval = stats.spearmanr(aligned["human"], aligned["gpt4"])
        comparison.attrs["discrimination_spearman_r"] = rho
        comparison.attrs["discrimination_spearman_p"] = pval

    return comparison
