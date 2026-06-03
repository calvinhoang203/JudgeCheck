"""JudgeCheck: IRT-based diagnostics for LLM-as-judge reliability."""

__version__ = "0.3.2"

from judgecheck.data import (
    CATEGORY_LABELS,
    WINNER_TO_ORDINAL,
    build_item_catalog,
    enrich_with_labels,
    load_gpt4_single_scores,
    load_mt_bench_judgments,
    load_mt_bench_questions,
    prepare_grm_matrix,
    prepare_score_grm_matrix,
    summarize_dataset,
    summarize_score_dataset,
)
from judgecheck.grm import (
    compare_item_discriminations,
    compare_judges,
    fit_grm,
    grm_results_to_frame,
    judge_abilities_to_frame,
    summarize_by_category,
    test_information_curve,
)
from judgecheck.pipeline import run_full_analysis, run_pairwise_analysis, run_score_analysis
from judgecheck.report import generate_html_report
from judgecheck.viz import (
    plot_category_discrimination,
    plot_discrimination_comparison,
    plot_item_discrimination,
    plot_judge_abilities,
    plot_method_discrimination_comparison,
    plot_test_information,
)

__all__ = [
    "CATEGORY_LABELS",
    "WINNER_TO_ORDINAL",
    "__version__",
    "build_item_catalog",
    "compare_item_discriminations",
    "compare_judges",
    "enrich_with_labels",
    "fit_grm",
    "generate_html_report",
    "grm_results_to_frame",
    "judge_abilities_to_frame",
    "load_gpt4_single_scores",
    "load_mt_bench_judgments",
    "load_mt_bench_questions",
    "plot_category_discrimination",
    "plot_discrimination_comparison",
    "plot_item_discrimination",
    "plot_judge_abilities",
    "plot_method_discrimination_comparison",
    "plot_test_information",
    "prepare_grm_matrix",
    "prepare_score_grm_matrix",
    "run_full_analysis",
    "run_pairwise_analysis",
    "run_score_analysis",
    "summarize_by_category",
    "summarize_dataset",
    "summarize_score_dataset",
    "test_information_curve",
]
