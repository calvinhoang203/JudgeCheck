"""Visualization helpers for JudgeCheck IRT diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from judgecheck.grm import GRMResults, grm_results_to_frame

PALETTE = {"human_experts": "#2E86AB", "gpt4_judge": "#E84855"}


def _labeled_items(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    plot_frame = frame.head(top_n).copy()
    if "short_label" in plot_frame.columns:
        plot_frame["display_label"] = plot_frame["short_label"]
    else:
        plot_frame["display_label"] = plot_frame["item_id"]
    return plot_frame


def plot_item_discrimination(
    results: GRMResults,
    *,
    top_n: int = 20,
    title: str | None = None,
    save_path: str | Path | None = None,
    label_frame: pd.DataFrame | None = None,
) -> plt.Figure:
    """
    Bar chart of item discrimination (a) parameters.

    Higher values = the benchmark question better separates good vs bad responses
    *as perceived by this judge system*.
    """
    frame = grm_results_to_frame(results)
    if label_frame is not None:
        frame = frame.merge(
            label_frame[["item_id", "short_label", "category_label"]],
            on="item_id",
            how="left",
        )
    frame = _labeled_items(frame, top_n)
    label = title or f"Top {top_n} sharpest benchmark questions ({results.judge_label})"

    fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.32)))
    sns.barplot(
        data=frame,
        y="display_label",
        x="discrimination",
        hue="discrimination",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Discrimination — higher means the question separates quality better")
    ax.set_ylabel("")
    ax.set_title(label, fontsize=12, pad=12)
    ax.axvline(results.mean_discrimination, color="gray", ls="--", lw=1)
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
    label_frame: pd.DataFrame | None = None,
) -> plt.Figure:
    """Scatter plot comparing item discrimination: human experts vs GPT-4 judge."""
    h = grm_results_to_frame(human).set_index("item_id")["discrimination"]
    g = grm_results_to_frame(gpt4).set_index("item_id")["discrimination"]
    merged = pd.concat([h, g], axis=1, keys=["human", "gpt4"]).dropna().reset_index()
    merged.columns = ["item_id", "human", "gpt4"]

    if label_frame is not None:
        merged = merged.merge(label_frame[["item_id", "category_label"]], on="item_id", how="left")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    if "category_label" in merged.columns:
        sns.scatterplot(
            data=merged,
            x="human",
            y="gpt4",
            hue="category_label",
            alpha=0.75,
            s=55,
            ax=ax,
        )
        ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(merged["human"], merged["gpt4"], alpha=0.55, edgecolors="white", s=40)

    lims = [
        min(merged["human"].min(), merged["gpt4"].min()) * 0.9,
        max(merged["human"].max(), merged["gpt4"].max()) * 1.1,
    ]
    ax.plot(lims, lims, "k--", alpha=0.35, label="equal discrimination")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Human experts — discrimination")
    ax.set_ylabel("GPT-4 judge — discrimination")
    ax.set_title("Do humans and GPT-4 agree on which questions are sharpest?", fontsize=11)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig


def plot_judge_abilities(
    results: GRMResults,
    *,
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of estimated judge ability (θ) for human annotators."""
    from judgecheck.grm import judge_abilities_to_frame

    frame = judge_abilities_to_frame(results).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.28)))
    sns.barplot(
        data=frame,
        y="judge_id",
        x="ability_theta",
        hue="ability_theta",
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Judge ability (θ) — higher means more decisive / consistent ratings")
    ax.set_ylabel("Human annotator")
    ax.set_title(f"Top {top_n} most consistent human judges", fontsize=12)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig


def plot_category_discrimination(
    category_summary: pd.DataFrame,
    *,
    judge_label: str = "judge",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of mean discrimination by MT-Bench category."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=category_summary,
        x="category_label",
        y="mean_discrimination",
        hue="mean_discrimination",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Question category")
    ax.set_ylabel("Average discrimination")
    ax.set_title(f"Which topic areas produce the sharpest questions? ({judge_label})")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig
