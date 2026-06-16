"""Build HTML report from pipeline outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from judgecheck.grm import GRMResults


def _benchmark_block(
    recommended: pd.DataFrame,
    *,
    csv_name: str,
    heading: str,
) -> str:
    n_rec = len(recommended)
    cov = recommended["cumulative_pct"].iloc[-1]
    rec_lines = "".join(
        f"<li>{escape(str(r.get('short_label', r['item_id'])))}</li>"
        for _, r in recommended.head(5).iterrows()
    )
    return f"""
    <h3>{heading} — {n_rec} items (~{cov:.0f}% information)</h3>
    <ul>{rec_lines}</ul>
    <p class="note">Full list: {csv_name}</p>
"""


def _score_section(score_outputs) -> str:
    if score_outputs is None:
        return ""

    items = score_outputs.item_params
    mean_disc = items["discrimination"].mean()
    peak = score_outputs.information.loc[score_outputs.information["test_information"].idxmax()]
    method_r = None
    if score_outputs.method_comparison is not None:
        method_r = score_outputs.method_comparison.attrs.get("spearman_r")

    method_metric = ""
    if method_r is not None:
        method_metric = (
            f"<div class='metric'><span>Pairwise vs score agreement</span>"
            f"<strong>{method_r:.2f}</strong></div>"
        )

    benchmark_block = ""
    if score_outputs.recommended_items is not None and not score_outputs.recommended_items.empty:
        benchmark_block = _benchmark_block(
            score_outputs.recommended_items,
            csv_name="recommended_benchmark_items.csv",
            heading="Benchmark designer",
        )

    return f"""
  <div class="card">
    <h2>Part B — GPT-4 score ratings (1–10)</h2>
    <p>
      MT-Bench also has <strong>direct 1–10 scores</strong> from GPT-4 (single-answer grading).
      We fit a full 10-level GRM across {score_outputs.results.n_participants} models.
    </p>
    <div class="metric">
      <span>Mean discrimination (scores)</span>
      <strong>{mean_disc:.2f}</strong>
    </div>
    <div class="metric">
      <span>Peak information θ</span>
      <strong>{peak['theta']:.1f}</strong>
    </div>
    {method_metric}
    {benchmark_block}
    <h3>Sharpest questions (score-based)</h3>
    <ul>
      {_item_rows(items, 5)}
    </ul>
    <img src="score_top_discrimination.png" alt="Score-based discrimination">
    <img src="score_test_information.png" alt="Test information curve">
    {"<img src='pairwise_vs_score_discrimination.png' alt='Pairwise vs score discrimination'>" if method_r is not None else ""}
    <p class="note">
      Test information peaks at θ ≈ {peak['theta']:.1f} — where this benchmark best separates model quality.
    </p>
  </div>
"""


def _table_html(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame if columns is None else frame[columns]
    return view.to_html(index=False, classes="data-table", border=0, escape=True)


def _item_rows(frame: pd.DataFrame, n: int = 5) -> str:
    cols = ["short_label", "discrimination"]
    if "category_label" in frame.columns:
        cols = ["category_label", "short_label", "discrimination"]
    rows = []
    for _, row in frame.head(n).iterrows():
        label = escape(str(row.get("short_label", row["item_id"])))
        cat = escape(str(row.get("category_label", "")))
        disc = f"{row['discrimination']:.2f}"
        rows.append(f"<li><strong>{disc}</strong> — {cat}: {label}</li>")
    return "\n".join(rows)


def generate_html_report(
    *,
    human_results: GRMResults,
    gpt4_results: GRMResults,
    human_items: pd.DataFrame,
    gpt4_items: pd.DataFrame,
    human_judges: pd.DataFrame | None,
    human_categories: pd.DataFrame,
    gpt4_categories: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: str | Path,
    score_outputs=None,
    winner_agreement_rate: float | None = None,
    winner_agreement_by_category: pd.DataFrame | None = None,
    category_discrimination_comparison: pd.DataFrame | None = None,
    tie_rates: pd.DataFrame | None = None,
    tie_rates_by_category: pd.DataFrame | None = None,
    recommended_pairwise_items: pd.DataFrame | None = None,
    human_peak_theta: float | None = None,
    recommended_overlap_summary: pd.DataFrame | None = None,
    recommended_overlap_detail: pd.DataFrame | None = None,
) -> Path:
    """Write ``outputs/report.html``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    human_row = comparison.loc[comparison["judge_system"] == "human_experts"].iloc[0]
    gpt4_row = comparison.loc[comparison["judge_system"] == "gpt4_judge"].iloc[0]
    spearman = comparison.attrs.get("discrimination_spearman_r")

    category_agreement_block = ""
    if (
        winner_agreement_by_category is not None
        and not winner_agreement_by_category.empty
    ):
        category_agreement_block = f"""
  <div class="card">
    <h2>Winner agreement by category</h2>
    {_table_html(
        winner_agreement_by_category,
        ["category_label", "n_comparisons", "pct_agreement"],
    )}
    <p class="note">Full table: pairwise_agreement_by_category.csv</p>
  </div>
"""

    category_discrimination_block = ""
    if (
        category_discrimination_comparison is not None
        and not category_discrimination_comparison.empty
    ):
        category_discrimination_block = f"""
  <div class="card">
    <h2>Discrimination by category (human vs GPT-4)</h2>
    {_table_html(
        category_discrimination_comparison,
        [
            "category_label",
            "mean_discrimination_human",
            "mean_discrimination_gpt4",
            "discrimination_gap",
        ],
    )}
    <img src="category_discrimination_comparison.png" alt="Category discrimination comparison">
    <p class="note">Full table: category_discrimination_comparison.csv</p>
  </div>
"""

    tie_block = ""
    if tie_rates is not None and not tie_rates.empty:
        human_tie = tie_rates.loc[
            tie_rates["judge_system"] == "human_experts", "pct_tie"
        ].iloc[0]
        gpt4_tie = tie_rates.loc[
            tie_rates["judge_system"] == "gpt4_judge", "pct_tie"
        ].iloc[0]
        tie_block = f"""
  <div class="card">
    <h2>Tie rates (indecisive judgments)</h2>
    <div class="metric">
      <span>Human tie rate</span>
      <strong>{human_tie:.1f}%</strong>
    </div>
    <div class="metric">
      <span>GPT-4 tie rate</span>
      <strong>{gpt4_tie:.1f}%</strong>
    </div>
    {"<img src='pairwise_tie_rates_by_category.png' alt='Tie rates by category'>" if tie_rates_by_category is not None and not tie_rates_by_category.empty else ""}
    {"<p class='note'>By category: pairwise_tie_rates_by_category.csv</p>" if tie_rates_by_category is not None and not tie_rates_by_category.empty else "<p class='note'>See pairwise_tie_rates.csv</p>"}
  </div>
"""

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pairwise_benchmark_block = ""
    if recommended_pairwise_items is not None and not recommended_pairwise_items.empty:
        pairwise_benchmark_block = _benchmark_block(
            recommended_pairwise_items,
            csv_name="recommended_pairwise_items.csv",
            heading="Benchmark designer (human pairwise)",
        )
        if human_peak_theta is not None:
            pairwise_benchmark_block += (
                f'<p class="note">Peak information θ ≈ {human_peak_theta:.1f}</p>'
            )
        pairwise_benchmark_block += (
            '<img src="human_test_information.png" alt="Human test information">'
        )

    overlap_block = ""
    if recommended_overlap_summary is not None and not recommended_overlap_summary.empty:
        row = recommended_overlap_summary.iloc[0]
        overlap_block = f"""
  <div class="card">
    <h2>Recommended sets overlap (pairwise vs scores)</h2>
    <div class="metric">
      <span>Items in both sets</span>
      <strong>{int(row['n_both'])}</strong>
    </div>
    <div class="metric">
      <span>Jaccard similarity</span>
      <strong>{row['jaccard']:.2f}</strong>
    </div>
    <div class="metric">
      <span>Pairwise rec. also in score set</span>
      <strong>{row['pct_pairwise_also_in_score']:.0f}%</strong>
    </div>
    {"<p class='note'>Only pairwise: " + str(int(row['n_only_pairwise'])) + "; only scores: " + str(int(row['n_only_score'])) + "</p>"}
    {"<ul>" + "".join(f"<li>{escape(str(r.get('short_label', r['item_id'])))} ({r['overlap_group']})</li>" for _, r in recommended_overlap_detail.head(8).iterrows()) + "</ul>" if recommended_overlap_detail is not None and not recommended_overlap_detail.empty else ""}
    <p class="note">Full tables: recommended_items_overlap_summary.csv, recommended_items_overlap_detail.csv</p>
  </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JudgeCheck Report — MT-Bench</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.55;
      color: #1a1a1a;
      max-width: 920px;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
      background: #fafafa;
    }}
    h1, h2, h3 {{ font-family: "Segoe UI", Arial, sans-serif; color: #0f2d52; }}
    .card {{
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin: 1.25rem 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .metric {{
      display: inline-block;
      min-width: 180px;
      margin: 0.5rem 1rem 0.5rem 0;
      padding: 0.75rem 1rem;
      background: #f0f7ff;
      border-radius: 8px;
      vertical-align: top;
    }}
    .metric strong {{ display: block; font-size: 1.5rem; color: #2E86AB; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #eee; padding: 0.45rem 0.5rem; text-align: left; }}
    .note {{ color: #555; font-size: 0.95rem; }}
    ul {{ padding-left: 1.2rem; }}
  </style>
</head>
<body>
  <h1>JudgeCheck Report</h1>
  <p class="note">Generated {generated}. Figures are PNGs in the same folder.</p>

  <div class="card">
    <h2>Overview</h2>
    <p>GRM item discrimination on MT-Bench. <strong>Discrimination</strong> = how sharply a question separates response quality for each judge system.</p>
  </div>

  <div class="card">
    <h2>Part A — Pairwise</h2>
    <div class="metric">
      <span>Human mean discrimination</span>
      <strong>{human_row['mean_discrimination']:.2f}</strong>
    </div>
    <div class="metric">
      <span>GPT-4 mean discrimination</span>
      <strong>{gpt4_row['mean_discrimination']:.2f}</strong>
    </div>
    {"<div class='metric'><span>Human vs GPT-4 agreement (Spearman)</span><strong>" + f"{spearman:.2f}" + "</strong></div>" if spearman is not None else ""}
    {"<div class='metric'><span>Same-comparison winner agreement</span><strong>" + f"{winner_agreement_rate * 100:.1f}%" + "</strong></div>" if winner_agreement_rate is not None else ""}
    <p class="note">
      Higher discrimination = sharper benchmark questions for that judge system.
      Human experts: {int(human_row['n_participants'])} annotators.
      GPT-4: {int(gpt4_row['n_responses'])} pairwise judgments.
    </p>
    {pairwise_benchmark_block}
  </div>

  <div class="card">
    <h2>Sharpest questions (human experts)</h2>
    <ul>
      {_item_rows(human_items, 5)}
    </ul>
    <img src="human_top_discrimination.png" alt="Top discriminating human items">
  </div>

  <div class="card">
    <h2>Weakest questions (human experts)</h2>
    <ul>
      {_item_rows(human_items.sort_values("discrimination").reset_index(drop=True), 5)}
    </ul>
    <p class="note">Low discrimination — see weak_benchmark_items.csv</p>
  </div>

  {"<div class='card'><h2>Human judge ability (θ)</h2>" + _table_html(human_judges.head(10), ["judge_id", "ability_theta", "ability_rank"]) + "<img src='human_judge_abilities.png' alt='Judge abilities'></div>" if human_judges is not None else ""}

  <div class="card">
    <h2>By category</h2>
    {_table_html(human_categories, ["category_label", "n_items", "mean_discrimination"])}
    <img src="human_category_discrimination.png" alt="Category discrimination">
  </div>

  <div class="card">
    <h2>Discrimination scatter (human vs GPT-4)</h2>
    <img src="human_vs_gpt4_discrimination.png" alt="Human vs GPT-4 discrimination scatter">
  </div>

  {category_agreement_block}

  {category_discrimination_block}

  {tie_block}

  {overlap_block}

  {_score_section(score_outputs)}

  <p class="note">Metric definitions: docs/GUIDE.md in the repository.</p>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
