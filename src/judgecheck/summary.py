"""Text summaries written after each pipeline run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from judgecheck import __version__


def model_score_ranking(score_df: pd.DataFrame) -> pd.DataFrame:
    """Rank evaluated LLMs by mean GPT-4 score (simple, interpretable)."""
    ranking = (
        score_df.groupby("model", as_index=False)
        .agg(mean_score=("score", "mean"), n_items=("score", "count"))
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking


def write_text_summary(
    output_dir: Path,
    *,
    pairwise_comparison: pd.DataFrame | None = None,
    human_top_item: str | None = None,
    human_weak_item: str | None = None,
    winner_agreement_rate: float | None = None,
    winner_agreement_by_category: pd.DataFrame | None = None,
    category_discrimination_comparison: pd.DataFrame | None = None,
    tie_rates: pd.DataFrame | None = None,
    recommended_pairwise_items: pd.DataFrame | None = None,
    human_peak_theta: float | None = None,
    recommended_items: pd.DataFrame | None = None,
    model_ranking: pd.DataFrame | None = None,
    peak_theta: float | None = None,
    coverage_target: float = 0.8,
    recommended_overlap_summary: pd.DataFrame | None = None,
) -> Path:
    """Write ``outputs/SUMMARY.txt``."""
    lines = [
        "JudgeCheck Run Summary",
        f"Version: {__version__}",
        "=" * 50,
        "",
    ]

    if pairwise_comparison is not None:
        human = pairwise_comparison.loc[
            pairwise_comparison["judge_system"] == "human_experts"
        ].iloc[0]
        gpt4 = pairwise_comparison.loc[
            pairwise_comparison["judge_system"] == "gpt4_judge"
        ].iloc[0]
        lines.extend(
            [
                "PAIRWISE JUDGMENTS (Part A)",
                f"  Human mean discrimination: {human['mean_discrimination']:.2f}",
                f"  GPT-4 mean discrimination: {gpt4['mean_discrimination']:.2f}",
            ]
        )
        rho = pairwise_comparison.attrs.get("discrimination_spearman_r")
        if rho is not None:
            lines.append(f"  Human vs GPT-4 agreement: {rho:.2f} (Spearman)")
        if human_top_item:
            lines.append(f"  Sharpest human item: {human_top_item}")
        if human_weak_item:
            lines.append(f"  Weakest human item: {human_weak_item}")
            lines.append("  See weak_benchmark_items.csv for more low-value questions.")
        if winner_agreement_rate is not None:
            lines.append(
                f"  Human vs GPT-4 winner agreement: {winner_agreement_rate * 100:.1f}%"
            )
            lines.append("  See pairwise_winner_agreement.csv for details.")
        if (
            winner_agreement_by_category is not None
            and not winner_agreement_by_category.empty
        ):
            low = winner_agreement_by_category.iloc[0]
            high = winner_agreement_by_category.iloc[-1]
            lines.append(
                f"  Lowest category agreement: {low['category_label']} "
                f"({low['pct_agreement']:.1f}%)"
            )
            lines.append(
                f"  Highest category agreement: {high['category_label']} "
                f"({high['pct_agreement']:.1f}%)"
            )
            lines.append("  See pairwise_agreement_by_category.csv.")
        if (
            category_discrimination_comparison is not None
            and not category_discrimination_comparison.empty
        ):
            low = category_discrimination_comparison.iloc[0]
            high = category_discrimination_comparison.iloc[-1]
            lines.append(
                f"  Largest human<GPT-4 discrimination gap: {low['category_label']} "
                f"({low['discrimination_gap']:.2f})"
            )
            lines.append(
                f"  Largest human>GPT-4 discrimination gap: {high['category_label']} "
                f"({high['discrimination_gap']:.2f})"
            )
            lines.append("  See category_discrimination_comparison.csv.")
        if tie_rates is not None and not tie_rates.empty:
            human_tie = tie_rates.loc[
                tie_rates["judge_system"] == "human_experts", "pct_tie"
            ].iloc[0]
            gpt4_tie = tie_rates.loc[
                tie_rates["judge_system"] == "gpt4_judge", "pct_tie"
            ].iloc[0]
            lines.append(f"  Human tie rate: {human_tie:.1f}%")
            lines.append(f"  GPT-4 tie rate: {gpt4_tie:.1f}%")
            lines.append("  See pairwise_tie_rates_by_category.csv.")
        if (
            recommended_pairwise_items is not None
            and not recommended_pairwise_items.empty
        ):
            n = len(recommended_pairwise_items)
            cov = recommended_pairwise_items["cumulative_pct"].iloc[-1]
            lines.extend(
                [
                    "BENCHMARK DESIGNER — PAIRWISE (Part A)",
                    f"  {n} questions cover ~{cov:.0f}% of diagnostic information "
                    f"(target {coverage_target * 100:.0f}%).",
                    "  See recommended_pairwise_items.csv for the list.",
                ]
            )
            if human_peak_theta is not None:
                lines.append(
                    f"  Human pairwise benchmark peaks at θ ≈ {human_peak_theta:.1f}"
                )
            lines.append("")

    if recommended_items is not None and not recommended_items.empty:
        n = len(recommended_items)
        cov = recommended_items["cumulative_pct"].iloc[-1]
        lines.extend(
            [
                "BENCHMARK DESIGNER (Part B)",
                f"  {n} questions cover ~{cov:.0f}% of diagnostic information "
                f"(target {coverage_target * 100:.0f}%).",
                "  See recommended_benchmark_items.csv for the list.",
            ]
        )
        if peak_theta is not None:
            lines.append(f"  Benchmark works hardest around quality level θ ≈ {peak_theta:.1f}")
        lines.append("")

    if recommended_overlap_summary is not None and not recommended_overlap_summary.empty:
        row = recommended_overlap_summary.iloc[0]
        lines.extend(
            [
                "RECOMMENDED SET OVERLAP",
                f"  Items in both pairwise and score sets: {int(row['n_both'])}",
                f"  Jaccard similarity: {row['jaccard']:.2f}",
                f"  {row['pct_pairwise_also_in_score']:.0f}% of pairwise picks also in score set",
                "  See recommended_items_overlap_detail.csv",
                "",
            ]
        )

    if model_ranking is not None and not model_ranking.empty:
        top = model_ranking.iloc[0]
        bottom = model_ranking.iloc[-1]
        lines.extend(
            [
                "MODEL RANKINGS (mean GPT-4 score)",
                f"  #1: {top['model']} ({top['mean_score']:.2f})",
                f"  Last: {bottom['model']} ({bottom['mean_score']:.2f})",
                "  Full list: model_score_ranking.csv",
                "",
            ]
        )

    lines.extend(
        [
            "FILES",
            "  report.html, report.pdf, SUMMARY.txt",
            "  docs/GUIDE.md — metric definitions",
            "  AGENTS.md — codebase notes for development",
            "",
        ]
    )

    path = output_dir / "SUMMARY.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_console_summary(
    *,
    pairwise_comparison: pd.DataFrame | None = None,
    recommended_items: pd.DataFrame | None = None,
    recommended_pairwise_items: pd.DataFrame | None = None,
    model_ranking: pd.DataFrame | None = None,
    winner_agreement_rate: float | None = None,
    recommended_overlap_summary: pd.DataFrame | None = None,
    coverage_target: float = 0.8,
) -> None:
    """Print headline findings after a pipeline run."""
    print("\n--- Quick findings ---")
    if pairwise_comparison is not None:
        human = pairwise_comparison.loc[
            pairwise_comparison["judge_system"] == "human_experts"
        ].iloc[0]
        print(f"  Human discrimination (mean): {human['mean_discrimination']:.2f}")
    if winner_agreement_rate is not None:
        print(f"  Human vs GPT-4 winner agreement: {winner_agreement_rate * 100:.1f}%")
    if recommended_pairwise_items is not None and not recommended_pairwise_items.empty:
        n = len(recommended_pairwise_items)
        cov = recommended_pairwise_items["cumulative_pct"].iloc[-1]
        print(
            f"  Pairwise: {n} items cover ~{cov:.0f}% of information "
            f"(target {coverage_target * 100:.0f}%)"
        )
    if recommended_items is not None and not recommended_items.empty:
        n = len(recommended_items)
        cov = recommended_items["cumulative_pct"].iloc[-1]
        print(
            f"  Scores: {n} items cover ~{cov:.0f}% of information "
            f"(target {coverage_target * 100:.0f}%)"
        )
    if recommended_overlap_summary is not None and not recommended_overlap_summary.empty:
        row = recommended_overlap_summary.iloc[0]
        print(
            f"  Recommended overlap: {int(row['n_both'])} items in both sets "
            f"(Jaccard {row['jaccard']:.2f})"
        )
    if model_ranking is not None and not model_ranking.empty:
        top = model_ranking.iloc[0]
        print(f"  Top model by GPT-4 score: {top['model']} ({top['mean_score']:.2f})")
    print("  See outputs/SUMMARY.txt and docs/GUIDE.md")
