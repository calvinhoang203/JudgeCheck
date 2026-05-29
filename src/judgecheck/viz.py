"""Visualization helpers for JudgeCheck IRT diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from judgecheck.grm import GRMResults, grm_results_to_frame

PALETTE = {"human_experts": "#2E86AB", "gpt4_judge": "#E84855"}


def plot_item_discrimination(
    results: GRMResults,
    *,
    top_n: int = 20,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Bar chart of item discrimination (a) parameters.

    Higher values = the benchmark question better separates good vs bad responses
    *as perceived by this judge system*.
    """
    frame = grm_results_to_frame(results).head(top_n)
    label = title or f"Top {top_n} MT-Bench items by discrimination ({results.judge_label})"

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.25)))
    sns.barplot(
        data=frame,
        y="item_id",
        x="discrimination",
        hue="discrimination",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Discrimination (a)\n↑ better at separating response quality")
    ax.set_ylabel("Benchmark item (question + turn)")
    ax.set_title(label, fontsize=12, pad=12)
    ax.axvline(results.mean_discrimination, color="gray", ls="--", lw=1)
    ax.text(
        results.mean_discrimination,
        0.02,
        " mean",
        transform=ax.get_xaxis_transform(),
        color="gray",
        fontsize=9,
    )
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig


def plot_discrimination_comparison(
    human: GRMResults,
    gpt4: GRMResults,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot comparing item discrimination: human experts vs GPT-4 judge."""
    h = grm_results_to_frame(human).set_index("item_id")["discrimination"]
    g = grm_results_to_frame(gpt4).set_index("item_id")["discrimination"]
    merged = pd.concat([h, g], axis=1, keys=["human", "gpt4"]).dropna().reset_index()
    merged.columns = ["item_id", "human", "gpt4"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(merged["human"], merged["gpt4"], alpha=0.55, edgecolors="white", s=40)
    lims = [
        min(merged["human"].min(), merged["gpt4"].min()) * 0.9,
        max(merged["human"].max(), merged["gpt4"].max()) * 1.1,
    ]
    ax.plot(lims, lims, "k--", alpha=0.35, label="equal discrimination")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Human expert discrimination (a)")
    ax.set_ylabel("GPT-4 judge discrimination (a)")
    ax.set_title(
        "Do humans and GPT-4 agree on which questions are most discriminating?",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig
