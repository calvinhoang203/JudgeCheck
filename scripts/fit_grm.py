#!/usr/bin/env python
"""
Run the full JudgeCheck analysis on MT-Bench human judgments.

Usage (from repo root, with venv activated):
    python scripts/fit_grm.py

Outputs land in ``outputs/`` including ``report.html`` for non-expert readers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from judgecheck.data import (  # noqa: E402
    build_item_catalog,
    enrich_with_labels,
    load_mt_bench_judgments,
    prepare_gpt4_comparison_matrix,
    prepare_grm_matrix,
    summarize_dataset,
)
from judgecheck.grm import (  # noqa: E402
    compare_judges,
    fit_grm,
    grm_results_to_frame,
    judge_abilities_to_frame,
    summarize_by_category,
)
from judgecheck.report import generate_html_report  # noqa: E402
from judgecheck.viz import (  # noqa: E402
    plot_category_discrimination,
    plot_discrimination_comparison,
    plot_item_discrimination,
    plot_judge_abilities,
)


def main() -> None:
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    print("=" * 60)
    print("JudgeCheck — MT-Bench IRT analysis")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading data...")
    catalog = build_item_catalog()
    human_df = load_mt_bench_judgments("human")
    gpt4_df = load_mt_bench_judgments("gpt4_pair")

    print(summarize_dataset(human_df, "human").to_string(index=False))
    print(summarize_dataset(gpt4_df, "gpt4_pair").to_string(index=False))

    # --- Build matrices ---
    print("\n[2/5] Building response matrices...")
    human_matrix, human_items, human_judges = prepare_grm_matrix(human_df)
    gpt4_matrix, gpt4_items, gpt4_slots = prepare_gpt4_comparison_matrix(gpt4_df)
    print(f"  Human: {human_matrix.shape[0]} items x {human_matrix.shape[1]} judges")
    print(f"  GPT-4: {gpt4_matrix.shape[0]} items x {gpt4_matrix.shape[1]} comparison slots")

    # --- Fit models ---
    print("\n[3/5] Fitting Graded Response Models...")
    print("  (Human model also estimates judge ability theta — may take 1–2 min)")
    human_results = fit_grm(
        human_matrix,
        human_items,
        human_judges,
        judge_label="human_experts",
        estimate_abilities=True,
    )
    gpt4_results = fit_grm(
        gpt4_matrix,
        gpt4_items,
        gpt4_slots,
        judge_label="gpt4_judge",
    )

    human_params = enrich_with_labels(grm_results_to_frame(human_results), catalog)
    gpt4_params = enrich_with_labels(grm_results_to_frame(gpt4_results), catalog)
    judge_params = judge_abilities_to_frame(human_results)
    comparison = compare_judges(human_results, gpt4_results)
    human_cat = summarize_by_category(human_params)
    gpt4_cat = summarize_by_category(gpt4_params)

    human_params.to_csv(out / "human_item_parameters.csv", index=False)
    gpt4_params.to_csv(out / "gpt4_item_parameters.csv", index=False)
    judge_params.to_csv(out / "human_judge_abilities.csv", index=False)
    comparison.to_csv(out / "judge_comparison.csv", index=False)
    human_cat.to_csv(out / "human_category_summary.csv", index=False)
    gpt4_cat.to_csv(out / "gpt4_category_summary.csv", index=False)

    print("\n  Reliability summary:")
    print(comparison.to_string(index=False))
    if "discrimination_spearman_r" in comparison.attrs:
        print(
            f"\n  Human vs GPT-4 discrimination correlation (Spearman): "
            f"{comparison.attrs['discrimination_spearman_r']:.3f}"
        )

    print("\n  Top 3 sharpest questions (human):")
    for _, row in human_params.head(3).iterrows():
        print(f"    [{row['discrimination']:.2f}] {row['short_label']}")

    # --- Visualize ---
    print("\n[4/5] Saving charts...")
    plot_item_discrimination(
        human_results,
        top_n=15,
        label_frame=catalog,
        save_path=out / "human_top_discrimination.png",
    )
    plot_item_discrimination(
        gpt4_results,
        top_n=15,
        title="Top 15 sharpest benchmark questions (GPT-4 judge)",
        label_frame=catalog,
        save_path=out / "gpt4_top_discrimination.png",
    )
    plot_discrimination_comparison(
        human_results,
        gpt4_results,
        label_frame=catalog,
        save_path=out / "human_vs_gpt4_discrimination.png",
    )
    plot_judge_abilities(human_results, top_n=15, save_path=out / "human_judge_abilities.png")
    plot_category_discrimination(
        human_cat,
        judge_label="human experts",
        save_path=out / "human_category_discrimination.png",
    )

    # --- HTML report ---
    print("\n[5/5] Writing plain-language HTML report...")
    report_path = generate_html_report(
        human_results=human_results,
        gpt4_results=gpt4_results,
        human_items=human_params,
        gpt4_items=gpt4_params,
        human_judges=judge_params,
        human_categories=human_cat,
        gpt4_categories=gpt4_cat,
        comparison=comparison,
        output_path=out / "report.html",
    )

    print("\nDone!")
    print(f"  Open in browser: {report_path.resolve()}")
    print(f"  All outputs:     {out.resolve()}")


if __name__ == "__main__":
    main()
