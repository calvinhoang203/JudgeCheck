"""Derived diagnostics beyond core GRM fits."""

from __future__ import annotations

import pandas as pd


def select_weak_items(
    item_params: pd.DataFrame,
    *,
    n: int = 15,
) -> pd.DataFrame:
    """Return the least discriminating benchmark items (candidates to revise or drop)."""
    cols = ["item_id", "discrimination"]
    optional = ["short_label", "category_label", "question_id", "turn"]
    keep = [c for c in cols + optional if c in item_params.columns]
    return (
        item_params[keep]
        .sort_values("discrimination", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


def _pairwise_comparison_frame(
    human_df: pd.DataFrame,
    gpt4_df: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["item_id", "model_a", "model_b"]
    human_majority = (
        human_df.groupby(keys, as_index=False)["winner"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        .rename(columns={"winner": "human_winner"})
    )
    gpt4_winners = gpt4_df[keys + ["winner"]].rename(columns={"winner": "gpt4_winner"})
    merged = human_majority.merge(gpt4_winners, on=keys, how="inner")
    merged["agree"] = merged["human_winner"] == merged["gpt4_winner"]
    return merged


def pairwise_winner_agreement(
    human_df: pd.DataFrame,
    gpt4_df: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare human majority vote vs GPT-4 pairwise winner on shared comparisons.

    Returns
    -------
    overall : one-row summary (agreement rate, n comparisons)
    by_item : per-item agreement rates
    by_category : per-category agreement rates (needs catalog with category_label)
    """
    merged = _pairwise_comparison_frame(human_df, gpt4_df)

    overall = pd.DataFrame(
        [
            {
                "n_comparisons": len(merged),
                "agreement_rate": merged["agree"].mean(),
                "pct_agreement": merged["agree"].mean() * 100,
            }
        ]
    )

    by_item = (
        merged.groupby("item_id", as_index=False)
        .agg(
            n_comparisons=("agree", "size"),
            agreement_rate=("agree", "mean"),
        )
        .assign(pct_agreement=lambda d: d["agreement_rate"] * 100)
        .sort_values("agreement_rate")
        .reset_index(drop=True)
    )

    by_category = pairwise_agreement_by_category(merged, catalog)

    return overall, by_item, by_category


def pairwise_agreement_by_category(
    merged: pd.DataFrame,
    catalog: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aggregate winner agreement by MT-Bench category."""
    if catalog is None or merged.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "category_label",
                "n_comparisons",
                "agreement_rate",
                "pct_agreement",
            ]
        )

    labeled = merged.merge(
        catalog[["item_id", "category", "category_label"]],
        on="item_id",
        how="left",
    )
    return (
        labeled.groupby(["category", "category_label"], as_index=False)
        .agg(
            n_comparisons=("agree", "size"),
            agreement_rate=("agree", "mean"),
        )
        .assign(pct_agreement=lambda d: d["agreement_rate"] * 100)
        .sort_values("agreement_rate")
        .reset_index(drop=True)
    )
