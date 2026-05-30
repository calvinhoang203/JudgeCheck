"""JudgeCheck: IRT-based diagnostics for LLM-as-judge reliability."""

__version__ = "0.2.0"

from judgecheck.data import (
    CATEGORY_LABELS,
    WINNER_TO_ORDINAL,
    build_item_catalog,
    enrich_with_labels,
    load_mt_bench_judgments,
    load_mt_bench_questions,
    prepare_grm_matrix,
    summarize_dataset,
)
from judgecheck.grm import (
    compare_judges,
    fit_grm,
    grm_results_to_frame,
    judge_abilities_to_frame,
    summarize_by_category,
)
from judgecheck.report import generate_html_report
from judgecheck.viz import (
    plot_category_discrimination,
    plot_discrimination_comparison,
    plot_item_discrimination,
    plot_judge_abilities,
)

__all__ = [
    "CATEGORY_LABELS",
    "WINNER_TO_ORDINAL",
    "build_item_catalog",
    "compare_judges",
    "enrich_with_labels",
    "fit_grm",
    "generate_html_report",
    "grm_results_to_frame",
    "judge_abilities_to_frame",
    "load_mt_bench_judgments",
    "load_mt_bench_questions",
    "plot_category_discrimination",
    "plot_discrimination_comparison",
    "plot_item_discrimination",
    "plot_judge_abilities",
    "prepare_grm_matrix",
    "summarize_by_category",
    "summarize_dataset",
]
