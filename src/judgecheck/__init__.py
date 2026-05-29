"""JudgeCheck: IRT-based diagnostics for LLM-as-judge reliability."""

__version__ = "0.1.0"

from judgecheck.data import (
    WINNER_TO_ORDINAL,
    load_mt_bench_judgments,
    prepare_grm_matrix,
    summarize_dataset,
)
from judgecheck.grm import compare_judges, fit_grm, grm_results_to_frame
from judgecheck.viz import plot_discrimination_comparison, plot_item_discrimination

__all__ = [
    "WINNER_TO_ORDINAL",
    "compare_judges",
    "fit_grm",
    "grm_results_to_frame",
    "load_mt_bench_judgments",
    "plot_discrimination_comparison",
    "plot_item_discrimination",
    "prepare_grm_matrix",
    "summarize_dataset",
]
