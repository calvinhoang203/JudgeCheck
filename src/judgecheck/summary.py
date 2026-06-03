"""Plain-language run summaries for non-expert audiences."""

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
    recommended_items: pd.DataFrame | None = None,
    model_ranking: pd.DataFrame | None = None,
    peak_theta: float | None = None,
    coverage_target: float = 0.8,
) -> Path:
    """Write ``SUMMARY.txt`` — quick read without opening HTML or CSVs."""
    lines = [
        "JudgeCheck Run Summary",
        f"Version: {__version__}",
        "=" * 50,
        "",
        "WHAT THIS MEANS (plain language)",
        "  JudgeCheck checks which MT-Bench questions actually help",
        "  tell good AI answers from bad ones.",
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
            "FILES TO OPEN",
            "  recommended_benchmark_items.csv — high-value question set",
            "  weak_benchmark_items.csv — low discrimination (pairwise)",
            "  weak_score_items.csv     — low discrimination (1–10 scores)",
            "  pairwise_winner_agreement.csv — human vs GPT-4 picks",
            "  report.html          — visual report (best for sharing)",
            "  SUMMARY.txt          — this file",
            "  docs/GETTING_STARTED.md — how to read the numbers",
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
    model_ranking: pd.DataFrame | None = None,
    winner_agreement_rate: float | None = None,
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
    if recommended_items is not None and not recommended_items.empty:
        n = len(recommended_items)
        cov = recommended_items["cumulative_pct"].iloc[-1]
        print(
            f"  {n} items cover ~{cov:.0f}% of information "
            f"(target {coverage_target * 100:.0f}%)"
        )
    if model_ranking is not None and not model_ranking.empty:
        top = model_ranking.iloc[0]
        print(f"  Top model by GPT-4 score: {top['model']} ({top['mean_score']:.2f})")
    print("  Open outputs/SUMMARY.txt or outputs/report.html for details.")
