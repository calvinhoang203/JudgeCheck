"""
Graded Response Model (GRM) fitting wrappers around girth.

In JudgeCheck framing:
  - Items  = MT-Bench questions (how discriminating is each benchmark item?)
  - People = human annotators or GPT-4 comparison slots (latent judging ability θ)
  - Ratings = ordinal preference from pairwise comparisons (1–3 scale here)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from girth import grm_mml, tag_missing_data
from scipy import stats
from scipy.special import expit

INVALID_FILL = -9999
VALID_RESPONSES = [1, 2, 3]


@dataclass
class GRMResults:
    """Container for a fitted Graded Response Model."""

    discrimination: np.ndarray
    difficulty: np.ndarray
    item_ids: list[str]
    participant_ids: list[str]
    judge_label: str
    aic: dict[str, float] | None
    bic: dict[str, float] | None
    n_items: int
    n_participants: int
    n_responses: int
    abilities: np.ndarray | None = field(default=None)
    valid_responses: list[int] = field(default_factory=lambda: [1, 2, 3])

    @property
    def mean_discrimination(self) -> float:
        return float(np.mean(self.discrimination))

    @property
    def reliability_proxy(self) -> float:
        """
        Simple reliability summary: mean item discrimination.

        Higher average discrimination suggests the judge/system more consistently
        separates strong vs weak responses across benchmark items.
        """
        return self.mean_discrimination


def _tag_matrix(matrix: np.ndarray, valid_responses: list[int]) -> np.ndarray:
    tagged = tag_missing_data(matrix, valid_responses)
    tagged = np.where(np.isnan(tagged), INVALID_FILL, tagged)
    return tagged.astype(int)


def _normalize_info_criteria(raw: dict | None) -> dict[str, float] | None:
    if not raw:
        return None
    if "final_model" in raw:
        return raw
    if "final" in raw:
        return {"final_model": raw["final"], "null_model": raw.get("null")}
    return raw


def _grm_response_prob(
    theta: np.ndarray,
    discrimination: float,
    thresholds: np.ndarray,
    category: int,
) -> np.ndarray:
    """Probability of a single GRM response category at theta grid points."""
    thresh = np.asarray(thresholds, dtype=float).copy()
    if np.isnan(thresh).any():
        # Rare items with unidentified upper threshold — extrapolate for EAP.
        base = thresh[~np.isnan(thresh)]
        if len(base) == 0:
            return np.full_like(theta, 1 / 3.0)
        if len(base) == 1:
            thresh = np.array([base[0], base[0] + 4.0])
        else:
            thresh = base

    k = int(category)

    def prob_at_least(level: int) -> np.ndarray:
        if level <= 1:
            return np.ones_like(theta)
        if level > len(thresh) + 1:
            return np.zeros_like(theta)
        return expit(discrimination * (theta - thresh[level - 2]))

    return np.clip(prob_at_least(k) - prob_at_least(k + 1), 1e-12, None)


def _category_probs_all(
    theta: np.ndarray,
    discrimination: float,
    thresholds: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """Stack of category probabilities shape (n_categories, len(theta))."""
    return np.vstack(
        [
            _grm_response_prob(theta, discrimination, thresholds, k)
            for k in range(1, n_categories + 1)
        ]
    )


def item_information(
    theta: np.ndarray,
    discrimination: float,
    thresholds: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """
    Samejima GRM item information I(θ) at grid points.

    Higher values = the item is more informative about latent ability at that θ.
    """
    probs = _category_probs_all(theta, discrimination, thresholds, n_categories)
    d_theta = theta[1] - theta[0] if len(theta) > 1 else 0.1
    d_prob = np.gradient(probs, d_theta, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        info = np.nansum((d_prob**2) / np.clip(probs, 1e-12, None), axis=0)
    return info


def test_information_curve(
    results: GRMResults,
    *,
    theta_min: float = -4.0,
    theta_max: float = 4.0,
    n_points: int = 81,
) -> pd.DataFrame:
    """Total test information T(θ) = sum of item information curves."""
    theta = np.linspace(theta_min, theta_max, n_points)
    n_categories = len(results.valid_responses)
    total = np.zeros(n_points)

    for i in range(results.n_items):
        total += item_information(
            theta,
            results.discrimination[i],
            results.difficulty[i],
            n_categories,
        )

    return pd.DataFrame(
        {
            "theta": theta,
            "test_information": total,
            "sem": 1 / np.sqrt(np.clip(total, 1e-12, None)),
        }
    )


def compare_item_discriminations(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    label_a: str = "method_a",
    label_b: str = "method_b",
) -> pd.DataFrame:
    """Align two item-parameter tables and compute rank correlation."""
    a = frame_a.set_index("item_id")[["discrimination"]].rename(columns={"discrimination": label_a})
    b = frame_b.set_index("item_id")[["discrimination"]].rename(columns={"discrimination": label_b})
    merged = a.join(b, how="inner").reset_index()
    if len(merged) >= 3:
        rho, pval = stats.spearmanr(merged[label_a], merged[label_b])
        merged.attrs["spearman_r"] = rho
        merged.attrs["spearman_p"] = pval
    return merged


def estimate_judge_abilities_eap(
    matrix: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    valid_responses: list[int],
    *,
    quadrature_n: int = 41,
    quadrature_bounds: tuple[float, float] = (-4.0, 4.0),
) -> np.ndarray:
    """
    Estimate latent judge ability (θ) via Expected A Posteriori (EAP).

    Uses item parameters from a fitted GRM and a standard normal prior.
    Works on Windows where girth's ``grm_mml_eap`` may fail (float128).
    """
    theta = np.linspace(quadrature_bounds[0], quadrature_bounds[1], quadrature_n)
    prior = stats.norm.pdf(theta)
    n_items, n_judges = matrix.shape
    abilities = np.zeros(n_judges)

    for j in range(n_judges):
        log_like = np.zeros(quadrature_n)
        for i in range(n_items):
            response = matrix[i, j]
            if np.isnan(response) or int(response) not in valid_responses:
                continue
            thresholds = difficulty[i]
            probs = _grm_response_prob(theta, discrimination[i], thresholds, int(response))
            log_like += np.log(np.clip(probs, 1e-12, None))

        posterior = np.exp(log_like - log_like.max()) * prior
        abilities[j] = np.sum(theta * posterior) / np.sum(posterior)

    return abilities


def fit_grm(
    matrix: np.ndarray,
    item_ids: list[str],
    participant_ids: list[str],
    *,
    judge_label: str = "judge",
    options: dict[str, Any] | None = None,
    estimate_abilities: bool = False,
    valid_responses: list[int] | None = None,
) -> GRMResults:
    """
    Fit a Graded Response Model via marginal maximum likelihood (girth).

    Parameters
    ----------
    matrix : array (n_items, n_participants)
        Ordinal responses; NaN for missing judgments.
    valid_responses : list[int], optional
        Allowed rating levels (default ``[1, 2, 3]`` for pairwise preferences).
    estimate_abilities : bool
        If True, estimate latent judge ability (θ) via EAP after fitting items.
    """
    responses = valid_responses or VALID_RESPONSES
    tagged = _tag_matrix(matrix, responses)
    estimates = grm_mml(tagged, options=options or {})
    abilities = None
    if estimate_abilities:
        abilities = estimate_judge_abilities_eap(
            matrix,
            np.asarray(estimates["Discrimination"]),
            np.asarray(estimates["Difficulty"]),
            responses,
        )

    n_observed = int(np.sum(np.isin(tagged, responses)))

    return GRMResults(
        discrimination=np.asarray(estimates["Discrimination"]),
        difficulty=np.asarray(estimates["Difficulty"]),
        item_ids=item_ids,
        participant_ids=participant_ids,
        judge_label=judge_label,
        aic=_normalize_info_criteria(estimates.get("AIC")),
        bic=_normalize_info_criteria(estimates.get("BIC")),
        n_items=len(item_ids),
        n_participants=len(participant_ids),
        n_responses=n_observed,
        abilities=abilities,
        valid_responses=responses,
    )


def grm_results_to_frame(results: GRMResults) -> pd.DataFrame:
    """Convert GRM item parameters to a tidy DataFrame."""
    n_thresholds = results.difficulty.shape[1]
    threshold_cols = [f"threshold_{k + 1}" for k in range(n_thresholds)]

    frame = pd.DataFrame(
        {
            "item_id": results.item_ids,
            "discrimination": results.discrimination,
            **{
                col: results.difficulty[:, i]
                for i, col in enumerate(threshold_cols)
            },
        }
    )
    frame["question_id"] = frame["item_id"].str.extract(r"^(\d+)_")[0].astype(int)
    frame["turn"] = frame["item_id"].str.extract(r"_t(\d+)$")[0].astype(int)
    return frame.sort_values("discrimination", ascending=False).reset_index(drop=True)


def judge_abilities_to_frame(results: GRMResults) -> pd.DataFrame:
    """
    Convert latent judge abilities (θ) to a ranked table.

    Higher θ → the judge tends to give stronger / more decisive preferences
    across benchmark items (in the model's latent scale).
    """
    if results.abilities is None:
        raise ValueError(
            "No abilities stored. Re-run fit_grm(..., estimate_abilities=True)."
        )

    frame = pd.DataFrame(
        {
            "judge_id": results.participant_ids,
            "ability_theta": results.abilities,
        }
    )
    frame["ability_rank"] = frame["ability_theta"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values("ability_theta", ascending=False).reset_index(drop=True)


def summarize_by_category(item_frame: pd.DataFrame) -> pd.DataFrame:
    """Average item discrimination by MT-Bench category."""
    required = {"category_label", "discrimination"}
    if not required.issubset(item_frame.columns):
        raise ValueError(f"item_frame must contain columns: {required}")

    summary = (
        item_frame.groupby(["category", "category_label"], as_index=False)
        .agg(
            n_items=("discrimination", "size"),
            mean_discrimination=("discrimination", "mean"),
            median_discrimination=("discrimination", "median"),
        )
        .sort_values("mean_discrimination", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def compare_judges(
    human: GRMResults,
    gpt4: GRMResults,
) -> pd.DataFrame:
    """Compare reliability metrics between two fitted judge systems."""
    rows = []
    for label, res in [("human_experts", human), ("gpt4_judge", gpt4)]:
        row = {
            "judge_system": label,
            "mean_discrimination": res.mean_discrimination,
            "median_discrimination": float(np.median(res.discrimination)),
            "n_items": res.n_items,
            "n_participants": res.n_participants,
            "n_responses": res.n_responses,
            "aic": (res.aic or {}).get("final_model") or (res.aic or {}).get("final"),
            "bic": (res.bic or {}).get("final_model") or (res.bic or {}).get("final"),
        }
        if res.abilities is not None:
            row["mean_judge_ability"] = float(np.mean(res.abilities))
            row["sd_judge_ability"] = float(np.std(res.abilities))
        rows.append(row)

    comparison = pd.DataFrame(rows)

    h = grm_results_to_frame(human).set_index("item_id")["discrimination"]
    g = grm_results_to_frame(gpt4).set_index("item_id")["discrimination"]
    aligned = pd.concat([h, g], axis=1, keys=["human", "gpt4"]).dropna()
    if len(aligned) >= 3:
        rho, pval = stats.spearmanr(aligned["human"], aligned["gpt4"])
        comparison.attrs["discrimination_spearman_r"] = rho
        comparison.attrs["discrimination_spearman_p"] = pval

    return comparison
