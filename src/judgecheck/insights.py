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


def compare_recommended_sets(
    pairwise_items: pd.DataFrame,
    score_items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare recommended question sets from pairwise and score tracks.

    Returns
    -------
    summary : one-row overlap statistics
    detail : union of recommended items with membership flags
    """
    pw_ids = set(pairwise_items["item_id"])
    sc_ids = set(score_items["item_id"])
    union = pw_ids | sc_ids
    both = pw_ids & sc_ids

    summary = pd.DataFrame(
        [
            {
                "n_pairwise": len(pw_ids),
                "n_score": len(sc_ids),
                "n_both": len(both),
                "n_only_pairwise": len(pw_ids - sc_ids),
                "n_only_score": len(sc_ids - pw_ids),
                "jaccard": len(both) / len(union) if union else 0.0,
                "pct_pairwise_also_in_score": (
                    len(both) / len(pw_ids) * 100 if pw_ids else 0.0
                ),
            }
        ]
    )

    label_lookup = pd.concat(
        [
            pairwise_items.set_index("item_id"),
            score_items.set_index("item_id"),
        ],
        axis=0,
    )
    label_lookup = label_lookup[~label_lookup.index.duplicated(keep="first")]

    rows: list[dict] = []
    for item_id in sorted(union):
        in_pw = item_id in pw_ids
        in_sc = item_id in sc_ids
        if in_pw and in_sc:
            group = "both"
        elif in_pw:
            group = "pairwise_only"
        else:
            group = "score_only"
        row: dict = {
            "item_id": item_id,
            "in_pairwise": in_pw,
            "in_score": in_sc,
            "overlap_group": group,
        }
        if item_id in label_lookup.index:
            meta = label_lookup.loc[item_id]
            for col in ("short_label", "category_label"):
                if col in meta.index and pd.notna(meta[col]):
                    row[col] = meta[col]
        rows.append(row)

    detail = pd.DataFrame(rows)
    group_order = {"both": 0, "pairwise_only": 1, "score_only": 2}
    detail["_sort"] = detail["overlap_group"].map(group_order)
    detail = (
        detail.sort_values(["_sort", "item_id"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    return summary, detail


def _tie_mask(series: pd.Series) -> pd.Series:
    return series.str.startswith("tie")


def pairwise_tie_rates(
    human_df: pd.DataFrame,
    gpt4_df: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare tie (indecisive) judgment rates for human experts vs GPT-4 pairwise.

    Returns
    -------
    overall : tie rate per judge system
    by_category : tie rates by MT-Bench category (needs catalog)
    """
    overall = pd.DataFrame(
        [
            {
                "judge_system": "human_experts",
                "n_judgments": len(human_df),
                "tie_rate": _tie_mask(human_df["winner"]).mean(),
            },
            {
                "judge_system": "gpt4_judge",
                "n_judgments": len(gpt4_df),
                "tie_rate": _tie_mask(gpt4_df["winner"]).mean(),
            },
        ]
    )
    overall["pct_tie"] = overall["tie_rate"] * 100

    if catalog is None or human_df.empty or gpt4_df.empty:
        return overall, pd.DataFrame(
            columns=[
                "category",
                "category_label",
                "tie_rate_human",
                "tie_rate_gpt4",
                "tie_rate_gap",
                "pct_tie_human",
                "pct_tie_gpt4",
            ]
        )

    human = human_df.merge(
        catalog[["item_id", "category", "category_label"]], on="item_id", how="left"
    )
    gpt4 = gpt4_df.merge(
        catalog[["item_id", "category", "category_label"]], on="item_id", how="left"
    )
    human_cat = (
        human.groupby(["category", "category_label"], as_index=False)
        .agg(tie_rate_human=("winner", lambda s: _tie_mask(s).mean()))
        .assign(pct_tie_human=lambda d: d["tie_rate_human"] * 100)
    )
    gpt4_cat = (
        gpt4.groupby(["category", "category_label"], as_index=False)
        .agg(tie_rate_gpt4=("winner", lambda s: _tie_mask(s).mean()))
        .assign(pct_tie_gpt4=lambda d: d["tie_rate_gpt4"] * 100)
    )
    by_category = human_cat.merge(
        gpt4_cat, on=["category", "category_label"], how="inner"
    )
    by_category["tie_rate_gap"] = (
        by_category["tie_rate_human"] - by_category["tie_rate_gpt4"]
    )
    return overall, by_category.sort_values("tie_rate_gap").reset_index(drop=True)


def judge_workload(human_df: pd.DataFrame) -> pd.DataFrame:
    """Count judgments and unique items per human annotator."""
    return (
        human_df.groupby("judge", as_index=False)
        .agg(n_judgments=("item_id", "size"), n_items=("item_id", "nunique"))
        .rename(columns={"judge": "judge_id"})
        .sort_values("n_judgments", ascending=False)
        .reset_index(drop=True)
    )


def enrich_judge_abilities(
    judge_params: pd.DataFrame,
    human_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach workload counts to the ranked judge ability table."""
    return judge_params.merge(judge_workload(human_df), on="judge_id", how="left")


def summarize_judge_abilities(judge_params: pd.DataFrame) -> pd.DataFrame:
    """One-row summary of human annotator θ estimates."""
    theta = judge_params["ability_theta"]
    top = judge_params.iloc[0]
    bottom = judge_params.iloc[-1]
    return pd.DataFrame(
        [
            {
                "n_judges": len(judge_params),
                "mean_theta": float(theta.mean()),
                "sd_theta": float(theta.std(ddof=0)),
                "min_theta": float(theta.min()),
                "max_theta": float(theta.max()),
                "most_decisive_judge": top["judge_id"],
                "most_decisive_theta": float(top["ability_theta"]),
                "least_decisive_judge": bottom["judge_id"],
                "least_decisive_theta": float(bottom["ability_theta"]),
            }
        ]
    )
