"""
Generate a plain-language HTML report for non-expert audiences.

Open ``outputs/report.html`` in any browser after running the analysis script.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from judgecheck.grm import GRMResults


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
        rec = score_outputs.recommended_items
        n_rec = len(rec)
        cov = rec["cumulative_pct"].iloc[-1]
        rec_lines = "".join(
            f"<li>{escape(str(r.get('short_label', r['item_id'])))}</li>"
            for _, r in rec.head(5).iterrows()
        )
        benchmark_block = f"""
    <h3>Benchmark designer — keep these {n_rec} questions (~{cov:.0f}% of information)</h3>
    <ul>{rec_lines}</ul>
    <p class="note">Full list: recommended_benchmark_items.csv</p>
"""

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
) -> Path:
    """Write a self-contained HTML summary with charts and plain-language captions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    human_row = comparison.loc[comparison["judge_system"] == "human_experts"].iloc[0]
    gpt4_row = comparison.loc[comparison["judge_system"] == "gpt4_judge"].iloc[0]
    spearman = comparison.attrs.get("discrimination_spearman_r")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
  <p class="note">Generated {generated}. Open this file in any web browser — no Python required.</p>

  <div class="card">
    <h2>What is this?</h2>
    <p>
      <strong>JudgeCheck</strong> checks whether LLM judges and human experts are reliable graders.
      We analyzed <strong>MT-Bench</strong> — a standard benchmark where judges pick the better of two AI answers.
    </p>
    <p>
      Each benchmark question gets a <strong>discrimination</strong> score:
      how sharply it separates good vs bad answers.
      Think of it like a test question that actually tells students apart vs one everyone gets right.
    </p>
  </div>

  <div class="card">
    <h2>Part A — Pairwise judgments</h2>
    <p>Human experts and GPT-4 compare two model answers (A vs B) on each question.</p>
  </div>

  <div class="card">
    <h2>Headline numbers (pairwise)</h2>
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
    <p class="note">These items rarely help judges tell models apart — candidates to revise or drop from benchmarks.</p>
  </div>

  {"<div class='card'><h2>Most consistent human judges</h2><p>Ability (θ) estimates how decisively and consistently each annotator rates across questions.</p>" + _table_html(human_judges.head(10), ["judge_id", "ability_theta", "ability_rank"]) + "<img src='human_judge_abilities.png' alt='Judge abilities'></div>" if human_judges is not None else ""}

  <div class="card">
    <h2>By topic category</h2>
    <p>Which MT-Bench areas produce the most informative questions?</p>
    {_table_html(human_categories, ["category_label", "n_items", "mean_discrimination"])}
    <img src="human_category_discrimination.png" alt="Category discrimination">
  </div>

  <div class="card">
    <h2>Human vs GPT-4</h2>
    <p>Do automated and human judges agree on which questions are sharpest?</p>
    <img src="human_vs_gpt4_discrimination.png" alt="Human vs GPT-4 discrimination scatter">
  </div>

  {_score_section(score_outputs)}

  <div class="card">
    <h2>How to read this</h2>
    <ul>
      <li><strong>Discrimination</strong> — item quality. High = good benchmark question.</li>
      <li><strong>Judge ability (θ)</strong> — judge skill/consistency. Only estimated for human annotators.</li>
      <li><strong>Pairwise data</strong> — Part A uses A-vs-B preferences (1–3).</li>
      <li><strong>Score data</strong> — Part B uses GPT-4 ratings on a 1–10 scale.</li>
      <li><strong>Test information</strong> — shows where on the quality scale the benchmark is most informative.</li>
    </ul>
    <p class="note">
      Method: Graded Response Model (GRM) via the
      <a href="https://github.com/eribean/girth">girth</a> Python package.
      Data:
      <a href="https://huggingface.co/datasets/lmsys/mt_bench_human_judgments">lmsys/mt_bench_human_judgments</a>.
    </p>
  </div>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
