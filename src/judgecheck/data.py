"""
Load and reshape MT-Bench human judgments for IRT analysis.

MT-Bench annotations are *pairwise* (model A vs model B), not direct 1–5 scores.
We map each judgment to a 3-level ordinal preference so a Graded Response Model
can estimate how well each benchmark question separates response quality.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from datasets import load_dataset

SplitName = Literal["human", "gpt4_pair"]

# Map pairwise outcomes to ordinal categories (1-indexed for girth).
WINNER_TO_ORDINAL: dict[str, int] = {
    "model_b": 1,  # prefers the second model (lower relative quality for A)
    "tie": 2,
    "tie (inconsistent)": 2,
    "model_a": 3,  # prefers the first model (higher relative quality for A)
}


def load_mt_bench_judgments(split: SplitName = "human") -> pd.DataFrame:
    """Load one split from lmsys/mt_bench_human_judgments on HuggingFace."""
    dataset = load_dataset("lmsys/mt_bench_human_judgments", split=split)
    df = dataset.to_pandas()
    df["item_id"] = (
        df["question_id"].astype(str) + "_t" + df["turn"].astype(str)
    )
    df["ordinal"] = df["winner"].map(WINNER_TO_ORDINAL)
    unknown = df["ordinal"].isna()
    if unknown.any():
        bad = df.loc[unknown, "winner"].unique().tolist()
        raise ValueError(f"Unmapped winner values: {bad}")
    return df


def summarize_dataset(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """Return a one-row summary table suitable for display in notebooks."""
    return pd.DataFrame(
        [
            {
                "split": name,
                "n_judgments": len(df),
                "n_judges": df["judge"].nunique(),
                "n_items": df["item_id"].nunique(),
                "n_questions": df["question_id"].nunique(),
                "pct_model_a": (df["winner"] == "model_a").mean() * 100,
                "pct_model_b": (df["winner"] == "model_b").mean() * 100,
                "pct_tie": df["winner"].str.startswith("tie").mean() * 100,
            }
        ]
    )


def prepare_grm_matrix(
    df: pd.DataFrame,
    *,
    participant_col: str = "judge",
    item_col: str = "item_id",
    response_col: str = "ordinal",
    aggregate: Literal["median", "first"] = "median",
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a GRM response matrix for girth: shape (n_items, n_participants).

    Rows = benchmark items (questions), columns = judges (or comparison slots).
    Missing cells are filled with NaN and should be tagged before estimation.

    When a participant rates the same item more than once (different model pairs),
    responses are aggregated with the median ordinal score by default.
    """
    work = df[[participant_col, item_col, response_col]].copy()

    if aggregate == "median":
        work = (
            work.groupby([item_col, participant_col], as_index=False)[response_col]
            .median()
            .assign(**{response_col: lambda d: d[response_col].round().astype(int)})
        )
    elif aggregate == "first":
        work = work.drop_duplicates(subset=[item_col, participant_col], keep="first")
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}")

    pivot = work.pivot(index=item_col, columns=participant_col, values=response_col)
    item_ids = pivot.index.tolist()
    participant_ids = pivot.columns.tolist()
    matrix = pivot.to_numpy(dtype=float)
    return matrix, item_ids, participant_ids


def prepare_gpt4_comparison_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Reshape GPT-4 pairwise judgments into (n_items, 15) matrix.

    Each MT-Bench item has 15 model-pair comparisons judged by GPT-4.
    We treat each comparison slot as a pseudo-participant so GRM can estimate
    item discrimination for the automated judge.
    """
    work = df.copy()
    work["comparison_idx"] = work.groupby("item_id").cumcount()
    pivot = work.pivot(index="item_id", columns="comparison_idx", values="ordinal")
    item_ids = pivot.index.tolist()
    slot_ids = [f"pair_{i}" for i in pivot.columns.tolist()]
    return pivot.to_numpy(dtype=float), item_ids, slot_ids
