"""
Central analysis pipeline for JudgeCheck.

All scripts should call functions here so future updates stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from judgecheck.config import AnalysisConfig
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
    item_information_contributions,
    judge_abilities_to_frame,
    recommend_benchmark_items,
    summarize_by_category,
    test_information_curve,
)
from judgecheck.insights import pairwise_winner_agreement, select_weak_items
from judgecheck.report import generate_html_report
from judgecheck.summary import model_score_ranking, print_console_summary, write_text_summary
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
    weak_human_items: pd.DataFrame | None = None
    winner_agreement: pd.DataFrame | None = None
    winner_agreement_by_item: pd.DataFrame | None = None


@dataclass
class ScoreOutputs:
    results: object
    item_params: pd.DataFrame
    categories: pd.DataFrame
    information: pd.DataFrame
    method_comparison: pd.DataFrame | None
    item_contributions: pd.DataFrame | None = None
    recommended_items: pd.DataFrame | None = None
    model_ranking: pd.DataFrame | None = None
    score_df: pd.DataFrame | None = None


def run_pairwise_analysis(
    output_dir: Path,
    catalog: pd.DataFrame | None = None,
    config: AnalysisConfig | None = None,
) -> PairwiseOutputs:
    """Pairwise A/B human + GPT-4 GRM (3-level ordinal)."""
    config = config or AnalysisConfig()
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
    weak_human = enrich_with_labels(
        select_weak_items(human_params, n=config.weak_item_count), catalog
    )
    agree_overall, agree_by_item = pairwise_winner_agreement(human_df, gpt4_df)
    agree_by_item = enrich_with_labels(agree_by_item, catalog)

    human_params.to_csv(output_dir / "human_item_parameters.csv", index=False)
    gpt4_params.to_csv(output_dir / "gpt4_item_parameters.csv", index=False)
    judge_params.to_csv(output_dir / "human_judge_abilities.csv", index=False)
    comparison.to_csv(output_dir / "judge_comparison.csv", index=False)
    human_categories.to_csv(output_dir / "human_category_summary.csv", index=False)
    gpt4_categories.to_csv(output_dir / "gpt4_category_summary.csv", index=False)
    weak_human.to_csv(output_dir / "weak_benchmark_items.csv", index=False)
    agree_overall.to_csv(output_dir / "pairwise_winner_agreement.csv", index=False)
    agree_by_item.to_csv(output_dir / "pairwise_agreement_by_item.csv", index=False)

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

    generate_html_report(
        human_results=human_results,
        gpt4_results=gpt4_results,
        human_items=human_params,
        gpt4_items=gpt4_params,
        human_judges=judge_params,
        human_categories=human_categories,
        gpt4_categories=gpt4_categories,
        comparison=comparison,
        score_outputs=None,
        winner_agreement_rate=(
            float(pairwise.winner_agreement["agreement_rate"].iloc[0])
            if pairwise.winner_agreement is not None
            and not pairwise.winner_agreement.empty
            else None
        ),
        output_path=output_dir / "report.html",
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
        weak_human_items=weak_human,
        winner_agreement=agree_overall,
        winner_agreement_by_item=agree_by_item,
    )


def run_score_analysis(
    output_dir: Path,
    catalog: pd.DataFrame | None = None,
    pairwise_human_params: pd.DataFrame | None = None,
    config: AnalysisConfig | None = None,
) -> ScoreOutputs:
    """GRM on GPT-4 single-answer scores (1–10) across 34 models."""
    config = config or AnalysisConfig()
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
    contributions = item_information_contributions(results, information)
    recommended = recommend_benchmark_items(contributions, coverage=config.coverage)
    recommended = enrich_with_labels(recommended, catalog)
    weak_score = enrich_with_labels(
        select_weak_items(item_params, n=config.weak_item_count), catalog
    )
    ranking = model_score_ranking(score_df)

    item_params.to_csv(output_dir / "score_item_parameters.csv", index=False)
    categories.to_csv(output_dir / "score_category_summary.csv", index=False)
    information.to_csv(output_dir / "score_test_information.csv", index=False)
    contributions.to_csv(output_dir / "score_item_information.csv", index=False)
    recommended.to_csv(output_dir / "recommended_benchmark_items.csv", index=False)
    weak_score.to_csv(output_dir / "weak_score_items.csv", index=False)
    ranking.to_csv(output_dir / "model_score_ranking.csv", index=False)

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
        item_contributions=contributions,
        recommended_items=recommended,
        model_ranking=ranking,
        score_df=score_df,
    )


def _finalize_outputs(
    output_dir: Path,
    pairwise: PairwiseOutputs | None,
    scores: ScoreOutputs | None,
    config: AnalysisConfig | None = None,
) -> None:
    """Write cross-cutting summaries (text file + console)."""
    config = config or AnalysisConfig()
    human_top = None
    human_weak = None
    agree_rate = None
    comparison = None
    if pairwise is not None:
        comparison = pairwise.comparison
        if not pairwise.human_params.empty:
            row = pairwise.human_params.iloc[0]
            human_top = row.get("short_label", row["item_id"])
        if pairwise.weak_human_items is not None and not pairwise.weak_human_items.empty:
            w = pairwise.weak_human_items.iloc[0]
            human_weak = w.get("short_label", w["item_id"])
        if pairwise.winner_agreement is not None and not pairwise.winner_agreement.empty:
            agree_rate = float(pairwise.winner_agreement["agreement_rate"].iloc[0])

    peak_theta = None
    recommended = None
    ranking = None
    if scores is not None:
        peak_idx = scores.information["test_information"].idxmax()
        peak_theta = float(scores.information.loc[peak_idx, "theta"])
        recommended = scores.recommended_items
        ranking = scores.model_ranking

    write_text_summary(
        output_dir,
        pairwise_comparison=comparison,
        human_top_item=human_top,
        human_weak_item=human_weak,
        winner_agreement_rate=agree_rate,
        recommended_items=recommended,
        model_ranking=ranking,
        peak_theta=peak_theta,
        coverage_target=config.coverage,
    )
    print_console_summary(
        pairwise_comparison=comparison,
        recommended_items=recommended,
        model_ranking=ranking,
        winner_agreement_rate=agree_rate,
        coverage_target=config.coverage,
    )


def run_full_analysis(
    output_dir: Path,
    config: AnalysisConfig | None = None,
) -> tuple[PairwiseOutputs, ScoreOutputs]:
    """Run both analysis tracks and write the combined HTML report."""
    config = config or AnalysisConfig()
    catalog = build_item_catalog()
    pairwise = run_pairwise_analysis(output_dir, catalog=catalog, config=config)
    scores = run_score_analysis(
        output_dir,
        catalog=catalog,
        pairwise_human_params=pairwise.human_params,
        config=config,
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
        winner_agreement_rate=(
            float(pairwise.winner_agreement["agreement_rate"].iloc[0])
            if pairwise.winner_agreement is not None
            and not pairwise.winner_agreement.empty
            else None
        ),
        output_path=output_dir / "report.html",
    )

    _finalize_outputs(output_dir, pairwise, scores, config=config)

    return pairwise, scores
