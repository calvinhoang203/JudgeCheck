"""
Central analysis pipeline for JudgeCheck.

All scripts should call functions here so future updates stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from judgecheck.data import (
    build_item_catalog,
    enrich_with_labels,
    load_gpt4_single_scores,
    load_mt_bench_judgments,
    prepare_gpt4_comparison_matrix,
    prepare_grm_matrix,
    prepare_score_grm_matrix,
    summarize_dataset,
)
from judgecheck.grm import (
    compare_judges,
    compare_item_discriminations,
    fit_grm,
    grm_results_to_frame,
    judge_abilities_to_frame,
    summarize_by_category,
    test_information_curve,
)
from judgecheck.report import generate_html_report
from judgecheck.viz import (
    plot_category_discrimination,
    plot_discrimination_comparison,
    plot_item_discrimination,
    plot_judge_abilities,
    plot_method_discrimination_comparison,
    plot_test_information,
)


@dataclass
class PairwiseOutputs:
    human_results: object
    gpt4_results: object
    human_params: pd.DataFrame
    gpt4_params: pd.DataFrame
    judge_params: pd.DataFrame
    comparison: pd.DataFrame
    human_categories: pd.DataFrame
    gpt4_categories: pd.DataFrame


@dataclass
class ScoreOutputs:
    results: object
    item_params: pd.DataFrame
    categories: pd.DataFrame
    information: pd.DataFrame
    method_comparison: pd.DataFrame | None


def run_pairwise_analysis(
    output_dir: Path,
    catalog: pd.DataFrame | None = None,
) -> PairwiseOutputs:
    """Pairwise A/B human + GPT-4 GRM (3-level ordinal)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if catalog is None:
        catalog = build_item_catalog()

    human_df = load_mt_bench_judgments("human")
    gpt4_df = load_mt_bench_judgments("gpt4_pair")

    human_matrix, human_items, human_judges = prepare_grm_matrix(human_df)
    gpt4_matrix, gpt4_items, gpt4_slots = prepare_gpt4_comparison_matrix(gpt4_df)

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
    human_categories = summarize_by_category(human_params)
    gpt4_categories = summarize_by_category(gpt4_params)

    human_params.to_csv(output_dir / "human_item_parameters.csv", index=False)
    gpt4_params.to_csv(output_dir / "gpt4_item_parameters.csv", index=False)
    judge_params.to_csv(output_dir / "human_judge_abilities.csv", index=False)
    comparison.to_csv(output_dir / "judge_comparison.csv", index=False)
    human_categories.to_csv(output_dir / "human_category_summary.csv", index=False)
    gpt4_categories.to_csv(output_dir / "gpt4_category_summary.csv", index=False)

    plot_item_discrimination(
        human_results,
        top_n=15,
        label_frame=catalog,
        save_path=output_dir / "human_top_discrimination.png",
    )
    plot_item_discrimination(
        gpt4_results,
        top_n=15,
        title="Top 15 sharpest benchmark questions (GPT-4 pairwise judge)",
        label_frame=catalog,
        save_path=output_dir / "gpt4_top_discrimination.png",
    )
    plot_discrimination_comparison(
        human_results,
        gpt4_results,
        label_frame=catalog,
        save_path=output_dir / "human_vs_gpt4_discrimination.png",
    )
    plot_judge_abilities(
        human_results, top_n=15, save_path=output_dir / "human_judge_abilities.png"
    )
    plot_category_discrimination(
        human_categories,
        judge_label="human experts",
        save_path=output_dir / "human_category_discrimination.png",
    )

    return PairwiseOutputs(
        human_results=human_results,
        gpt4_results=gpt4_results,
        human_params=human_params,
        gpt4_params=gpt4_params,
        judge_params=judge_params,
        comparison=comparison,
        human_categories=human_categories,
        gpt4_categories=gpt4_categories,
    )


def run_score_analysis(
    output_dir: Path,
    catalog: pd.DataFrame | None = None,
    pairwise_human_params: pd.DataFrame | None = None,
) -> ScoreOutputs:
    """GRM on GPT-4 single-answer scores (1–10) across 34 models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if catalog is None:
        catalog = build_item_catalog()

    data_dir = output_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    score_df = load_gpt4_single_scores(cache_path=data_dir / "gpt4_single.jsonl")
    matrix, item_ids, model_ids = prepare_score_grm_matrix(score_df)

    results = fit_grm(
        matrix,
        item_ids,
        model_ids,
        judge_label="gpt4_single_scores",
        valid_responses=list(range(1, 11)),
    )

    item_params = enrich_with_labels(grm_results_to_frame(results), catalog)
    categories = summarize_by_category(item_params)
    information = test_information_curve(results)

    item_params.to_csv(output_dir / "score_item_parameters.csv", index=False)
    categories.to_csv(output_dir / "score_category_summary.csv", index=False)
    information.to_csv(output_dir / "score_test_information.csv", index=False)

    plot_item_discrimination(
        results,
        top_n=15,
        title="Top 15 sharpest questions (GPT-4 score ratings 1–10)",
        label_frame=catalog,
        save_path=output_dir / "score_top_discrimination.png",
    )
    plot_category_discrimination(
        categories,
        judge_label="GPT-4 score ratings",
        save_path=output_dir / "score_category_discrimination.png",
    )
    plot_test_information(
        information,
        save_path=output_dir / "score_test_information.png",
    )

    method_comparison = None
    if pairwise_human_params is not None:
        method_comparison = compare_item_discriminations(
            pairwise_human_params,
            item_params,
            label_a="human_pairwise",
            label_b="gpt4_scores",
        )
        method_comparison.to_csv(output_dir / "method_discrimination_comparison.csv", index=False)
        plot_method_discrimination_comparison(
            pairwise_human_params,
            item_params,
            label_a="Human pairwise",
            label_b="GPT-4 scores (1–10)",
            save_path=output_dir / "pairwise_vs_score_discrimination.png",
        )

    return ScoreOutputs(
        results=results,
        item_params=item_params,
        categories=categories,
        information=information,
        method_comparison=method_comparison,
    )


def run_full_analysis(output_dir: Path) -> tuple[PairwiseOutputs, ScoreOutputs]:
    """Run both analysis tracks and write the combined HTML report."""
    catalog = build_item_catalog()
    pairwise = run_pairwise_analysis(output_dir, catalog=catalog)
    scores = run_score_analysis(
        output_dir,
        catalog=catalog,
        pairwise_human_params=pairwise.human_params,
    )

    generate_html_report(
        human_results=pairwise.human_results,
        gpt4_results=pairwise.gpt4_results,
        human_items=pairwise.human_params,
        gpt4_items=pairwise.gpt4_params,
        human_judges=pairwise.judge_params,
        human_categories=pairwise.human_categories,
        gpt4_categories=pairwise.gpt4_categories,
        comparison=pairwise.comparison,
        score_outputs=scores,
        output_path=output_dir / "report.html",
    )

    return pairwise, scores
