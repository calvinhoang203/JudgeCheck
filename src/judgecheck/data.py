"""
Load and reshape MT-Bench human judgments for IRT analysis.

MT-Bench annotations are *pairwise* (model A vs model B), not direct 1–5 scores.
We map each judgment to a 3-level ordinal preference so a Graded Response Model
can estimate how well each benchmark question separates response quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from datasets import load_dataset

SplitName = Literal["human", "gpt4_pair"]

GPT4_SINGLE_URL = (
    "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
    "data/mt_bench/model_judgment/gpt-4_single.jsonl"
)

# Map pairwise outcomes to ordinal categories (1-indexed for girth).
WINNER_TO_ORDINAL: dict[str, int] = {
    "model_b": 1,  # prefers the second model (lower relative quality for A)
    "tie": 2,
    "tie (inconsistent)": 2,
    "model_a": 3,  # prefers the first model (higher relative quality for A)
}

# Friendly names for MT-Bench categories (for plots and reports).
CATEGORY_LABELS: dict[str, str] = {
    "writing": "Writing",
    "roleplay": "Role-play",
    "reasoning": "Reasoning",
    "math": "Math",
    "coding": "Coding",
    "extraction": "Information extraction",
    "stem": "STEM",
    "humanities": "Humanities",
}


def load_mt_bench_judgments(split: SplitName = "human") -> pd.DataFrame:
    """Load one split from lmsys/mt_bench_human_judgments on HuggingFace."""
    dataset = load_dataset("lmsys/mt_bench_human_judgments", split=split)
    df = dataset.to_pandas()
    df["item_id"] = (
        df["question_id"].astype(str) + "_t" + df["turn"].astype(str)
    )
    df["ordinal"] = df["winner"].map(WINNER_TO_ORDINAL)
    unknown = df["ordinal"].isna()
    if unknown.any():
        bad = df.loc[unknown, "winner"].unique().tolist()
        raise ValueError(f"Unmapped winner values: {bad}")
    return df


def load_mt_bench_questions() -> pd.DataFrame:
    """Load MT-Bench question text and categories from HuggingFace."""
    dataset = load_dataset("philschmid/mt-bench", split="train")
    return dataset.to_pandas()


def build_item_catalog(questions: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a lookup table: one row per benchmark item (question × turn).

    Adds human-readable labels so plots and reports show real prompts,
    not opaque IDs like ``129_t1``.
    """
    if questions is None:
        questions = load_mt_bench_questions()

    rows: list[dict] = []
    for _, row in questions.iterrows():
        category = row["category"]
        category_label = CATEGORY_LABELS.get(category, category.title())
        for turn_idx, prompt in enumerate(row["turns"], start=1):
            prompt_text = str(prompt).strip()
            short_prompt = prompt_text if len(prompt_text) <= 72 else prompt_text[:69] + "..."
            rows.append(
                {
                    "item_id": f"{row['question_id']}_t{turn_idx}",
                    "question_id": int(row["question_id"]),
                    "turn": turn_idx,
                    "category": category,
                    "category_label": category_label,
                    "prompt": prompt_text,
                    "short_label": (
                        f"Q{row['question_id']} [{category_label}] T{turn_idx}: "
                        f"{short_prompt}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def enrich_with_labels(frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Join question text and category labels onto an analysis DataFrame."""
    merged = frame.merge(catalog, on="item_id", how="left", suffixes=("", "_cat"))
    if "question_id_cat" in merged.columns:
        merged = merged.drop(columns=["question_id_cat", "turn_cat"], errors="ignore")
    return merged


def summarize_dataset(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """Return a one-row summary table suitable for display in notebooks."""
    return pd.DataFrame(
        [
            {
                "split": name,
                "n_judgments": len(df),
                "n_judges": df["judge"].nunique(),
                "n_items": df["item_id"].nunique(),
                "n_questions": df["question_id"].nunique(),
                "pct_model_a": (df["winner"] == "model_a").mean() * 100,
                "pct_model_b": (df["winner"] == "model_b").mean() * 100,
                "pct_tie": df["winner"].str.startswith("tie").mean() * 100,
            }
        ]
    )


def prepare_grm_matrix(
    df: pd.DataFrame,
    *,
    participant_col: str = "judge",
    item_col: str = "item_id",
    response_col: str = "ordinal",
    aggregate: Literal["median", "first"] = "median",
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a GRM response matrix for girth: shape (n_items, n_participants).

    Rows = benchmark items (questions), columns = judges (or comparison slots).
    Missing cells are filled with NaN and should be tagged before estimation.

    When a participant rates the same item more than once (different model pairs),
    responses are aggregated with the median ordinal score by default.
    """
    work = df[[participant_col, item_col, response_col]].copy()

    if aggregate == "median":
        work = (
            work.groupby([item_col, participant_col], as_index=False)[response_col]
            .median()
            .assign(**{response_col: lambda d: d[response_col].round().astype(int)})
        )
    elif aggregate == "first":
        work = work.drop_duplicates(subset=[item_col, participant_col], keep="first")
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}")

    pivot = work.pivot(index=item_col, columns=participant_col, values=response_col)
    item_ids = pivot.index.tolist()
    participant_ids = pivot.columns.tolist()
    matrix = pivot.to_numpy(dtype=float)
    return matrix, item_ids, participant_ids


def prepare_gpt4_comparison_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Reshape GPT-4 pairwise judgments into (n_items, 15) matrix.

    Each MT-Bench item has 15 model-pair comparisons judged by GPT-4.
    We treat each comparison slot as a pseudo-participant so GRM can estimate
    item discrimination for the automated judge.
    """
    work = df.copy()
    work["comparison_idx"] = work.groupby("item_id").cumcount()
    pivot = work.pivot(index="item_id", columns="comparison_idx", values="ordinal")
    item_ids = pivot.index.tolist()
    slot_ids = [f"pair_{i}" for i in pivot.columns.tolist()]
    return pivot.to_numpy(dtype=float), item_ids, slot_ids


def load_gpt4_single_scores(
    *,
    cache_path: Path | str | None = None,
    min_score: int = 1,
    max_score: int = 10,
) -> pd.DataFrame:
    """
    Load GPT-4 single-answer MT-Bench scores (1–10 per model response).

    Data source: FastChat ``gpt-4_single.jsonl`` on HuggingFace.
    Cached locally under ``data/gpt4_single.jsonl`` on first download.
    """
    if cache_path is None:
        cache_path = Path("data") / "gpt4_single.jsonl"
    else:
        cache_path = Path(cache_path)

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(GPT4_SINGLE_URL, cache_path)

    rows: list[dict] = []
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            score = record.get("score")
            if score is None or score < min_score or score > max_score:
                continue
            rows.append(
                {
                    "question_id": int(record["question_id"]),
                    "turn": int(record["turn"]),
                    "model": record["model"],
                    "score": int(score),
                    "item_id": f"{record['question_id']}_t{record['turn']}",
                }
            )

    return pd.DataFrame(rows)


def prepare_score_grm_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build GRM matrix from 1–10 scores: shape (n_items, n_models).

    Rows = benchmark items, columns = evaluated LLMs, values = GPT-4 scores.
    """
    pivot = df.pivot(index="item_id", columns="model", values="score")
    return pivot.to_numpy(dtype=float), pivot.index.tolist(), pivot.columns.tolist()


def summarize_score_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary for GPT-4 single-score data."""
    return pd.DataFrame(
        [
            {
                "split": "gpt4_single_scores",
                "n_scores": len(df),
                "n_models": df["model"].nunique(),
                "n_items": df["item_id"].nunique(),
                "mean_score": df["score"].mean(),
                "score_min": df["score"].min(),
                "score_max": df["score"].max(),
            }
        ]
    )
