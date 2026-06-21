"""Machine-readable run summary for scripts and reproducibility."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from judgecheck import __version__

if TYPE_CHECKING:
    from judgecheck.config import AnalysisConfig
    from judgecheck.pipeline import PairwiseOutputs, ScoreOutputs


def _row_value(frame, column: str, default: Any = None) -> Any:
    if frame is None or frame.empty or column not in frame.columns:
        return default
    value = frame.iloc[0][column]
    if hasattr(value, "item"):
        return value.item()
    return value


def build_metrics_manifest(
    *,
    config: AnalysisConfig,
    pairwise: PairwiseOutputs | None = None,
    scores: ScoreOutputs | None = None,
    recommended_overlap_summary=None,
) -> dict[str, Any]:
    """Collect headline metrics from pipeline outputs."""
    manifest: dict[str, Any] = {
        "version": __version__,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "coverage": config.coverage,
            "sharp_item_pct": config.sharp_item_pct,
            "weak_item_count": config.weak_item_count,
            "export_pdf": config.export_pdf,
        },
    }

    if pairwise is not None:
        comparison = pairwise.comparison
        human_row = None
        gpt4_row = None
        if comparison is not None and not comparison.empty:
            human_rows = comparison.loc[comparison["judge_system"] == "human_experts"]
            gpt4_rows = comparison.loc[comparison["judge_system"] == "gpt4_judge"]
            if not human_rows.empty:
                human_row = human_rows.iloc[0]
            if not gpt4_rows.empty:
                gpt4_row = gpt4_rows.iloc[0]

        pairwise_block: dict[str, Any] = {}
        if human_row is not None:
            pairwise_block["human_mean_discrimination"] = float(
                human_row["mean_discrimination"]
            )
        if gpt4_row is not None:
            pairwise_block["gpt4_mean_discrimination"] = float(
                gpt4_row["mean_discrimination"]
            )
        if comparison is not None:
            rho = comparison.attrs.get("discrimination_spearman_r")
            if rho is not None:
                pairwise_block["discrimination_spearman_r"] = float(rho)

        pairwise_block["winner_agreement_rate"] = _row_value(
            pairwise.winner_agreement, "agreement_rate"
        )
        pairwise_block["human_tie_rate"] = None
        pairwise_block["gpt4_tie_rate"] = None
        if pairwise.tie_rates is not None and not pairwise.tie_rates.empty:
            for _, row in pairwise.tie_rates.iterrows():
                key = (
                    "human_tie_rate"
                    if row["judge_system"] == "human_experts"
                    else "gpt4_tie_rate"
                )
                pairwise_block[key] = float(row["tie_rate"])

        if pairwise.judge_summary is not None and not pairwise.judge_summary.empty:
            js = pairwise.judge_summary.iloc[0]
            pairwise_block["judge_count"] = int(js["n_judges"])
            pairwise_block["judge_mean_theta"] = float(js["mean_theta"])
            pairwise_block["judge_sd_theta"] = float(js["sd_theta"])

        if (
            pairwise.item_disc_agreement_summary is not None
            and not pairwise.item_disc_agreement_summary.empty
        ):
            ids = pairwise.item_disc_agreement_summary.iloc[0]
            pairwise_block["sharp_items_both"] = int(ids["n_both_sharp"])
            pairwise_block["sharp_jaccard"] = float(ids["sharp_jaccard"])

        if (
            pairwise.recommended_pairwise_items is not None
            and not pairwise.recommended_pairwise_items.empty
        ):
            rec = pairwise.recommended_pairwise_items
            pairwise_block["recommended_pairwise_count"] = len(rec)
            pairwise_block["recommended_pairwise_coverage_pct"] = float(
                rec["cumulative_pct"].iloc[-1]
            )

        if pairwise.human_information is not None and not pairwise.human_information.empty:
            peak_idx = pairwise.human_information["test_information"].idxmax()
            pairwise_block["human_peak_theta"] = float(
                pairwise.human_information.loc[peak_idx, "theta"]
            )

        manifest["pairwise"] = {k: v for k, v in pairwise_block.items() if v is not None}

    if scores is not None:
        score_block: dict[str, Any] = {}
        if scores.item_params is not None and not scores.item_params.empty:
            score_block["mean_discrimination"] = float(
                scores.item_params["discrimination"].mean()
            )
        if scores.information is not None and not scores.information.empty:
            peak_idx = scores.information["test_information"].idxmax()
            score_block["peak_theta"] = float(scores.information.loc[peak_idx, "theta"])
        if scores.recommended_items is not None and not scores.recommended_items.empty:
            rec = scores.recommended_items
            score_block["recommended_count"] = len(rec)
            score_block["recommended_coverage_pct"] = float(rec["cumulative_pct"].iloc[-1])
        if scores.model_ranking is not None and not scores.model_ranking.empty:
            top = scores.model_ranking.iloc[0]
            score_block["top_model"] = str(top["model"])
            score_block["top_model_mean_score"] = float(top["mean_score"])
        if scores.method_comparison is not None:
            rho = scores.method_comparison.attrs.get("spearman_r")
            if rho is not None:
                score_block["pairwise_vs_score_spearman_r"] = float(rho)
        manifest["scores"] = {k: v for k, v in score_block.items() if v is not None}

    if recommended_overlap_summary is not None and not recommended_overlap_summary.empty:
        row = recommended_overlap_summary.iloc[0]
        manifest["recommended_overlap"] = {
            "n_both": int(row["n_both"]),
            "jaccard": float(row["jaccard"]),
            "pct_pairwise_also_in_score": float(row["pct_pairwise_also_in_score"]),
        }

    return manifest


def write_metrics_json(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    filename: str = "metrics.json",
) -> Path:
    """Write ``outputs/metrics.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
