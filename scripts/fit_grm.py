#!/usr/bin/env python
"""
Fit the first JudgeCheck GRM models on MT-Bench human judgments.

Usage (from repo root, with venv activated):
    python scripts/fit_grm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from judgecheck.data import (  # noqa: E402
    load_mt_bench_judgments,
    prepare_gpt4_comparison_matrix,
    prepare_grm_matrix,
    summarize_dataset,
)
from judgecheck.grm import compare_judges, fit_grm, grm_results_to_frame  # noqa: E402
from judgecheck.viz import (  # noqa: E402
    plot_discrimination_comparison,
    plot_item_discrimination,
)


def main() -> None:
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    print("=" * 60)
    print("JudgeCheck — First GRM fit on MT-Bench judgments")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/4] Loading MT-Bench human judgments from HuggingFace...")
    human_df = load_mt_bench_judgments("human")
    gpt4_df = load_mt_bench_judgments("gpt4_pair")

    print(summarize_dataset(human_df, "human").to_string(index=False))
    print(summarize_dataset(gpt4_df, "gpt4_pair").to_string(index=False))

    # --- Build response matrices ---
    print("\n[2/4] Building ordinal response matrices for GRM...")
    human_matrix, human_items, human_judges = prepare_grm_matrix(human_df)
    gpt4_matrix, gpt4_items, gpt4_slots = prepare_gpt4_comparison_matrix(gpt4_df)

    print(f"  Human matrix: {human_matrix.shape[0]} items × {human_matrix.shape[1]} judges")
    print(f"  GPT-4 matrix: {gpt4_matrix.shape[0]} items × {gpt4_matrix.shape[1]} comparison slots")
    print(
        "  Note: MT-Bench uses pairwise preferences, mapped to 3-level ordinal "
        "scores (model_b=1, tie=2, model_a=3) for GRM estimation."
    )

    # --- Fit GRM ---
    print("\n[3/4] Fitting Graded Response Models (marginal MLE via girth)...")
    human_results = fit_grm(
        human_matrix,
        human_items,
        human_judges,
        judge_label="human_experts",
    )
    gpt4_results = fit_grm(
        gpt4_matrix,
        gpt4_items,
        gpt4_slots,
        judge_label="gpt4_judge",
    )

    human_params = grm_results_to_frame(human_results)
    gpt4_params = grm_results_to_frame(gpt4_results)
    comparison = compare_judges(human_results, gpt4_results)

    human_params.to_csv(out / "human_item_parameters.csv", index=False)
    gpt4_params.to_csv(out / "gpt4_item_parameters.csv", index=False)
    comparison.to_csv(out / "judge_comparison.csv", index=False)

    print("\n  Reliability summary (mean item discrimination):")
    print(comparison.to_string(index=False))

    print("\n  Top 5 most discriminating items (human experts):")
    print(
        human_params[["item_id", "discrimination"]]
        .head()
        .to_string(index=False)
    )

    print("\n  Bottom 5 least discriminating items (human experts):")
    print(
        human_params[["item_id", "discrimination"]]
        .tail()
        .to_string(index=False)
    )

    # --- Visualize ---
    print("\n[4/4] Saving figures to outputs/...")
    plot_item_discrimination(
        human_results,
        top_n=20,
        save_path=out / "human_top_discrimination.png",
    )
    plot_item_discrimination(
        gpt4_results,
        top_n=20,
        title="Top 20 MT-Bench items by discrimination (GPT-4 judge)",
        save_path=out / "gpt4_top_discrimination.png",
    )
    plot_discrimination_comparison(
        human_results,
        gpt4_results,
        save_path=out / "human_vs_gpt4_discrimination.png",
    )

    print("\nDone. Outputs written to:", out.resolve())
    print("Next: open notebooks/01_explore_and_fit_grm.ipynb for an interactive walkthrough.")


if __name__ == "__main__":
    main()
