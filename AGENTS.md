# JudgeCheck — developer notes

Use this file when continuing work in a new chat or onboarding to the codebase.

## Purpose

JudgeCheck fits **Item Response Theory (IRT)** models to **MT-Bench** judge data. Main output: **item discrimination** — how well each benchmark question separates response quality for a given judge system.

PhD context: LLM-as-judge reliability; GRM via [`girth`](https://github.com/eribean/girth).

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_analysis.py              # full (~5 min)
python scripts/run_analysis.py --pairwise-only
python scripts/run_analysis.py --scores-only
python scripts/run_analysis.py --coverage 0.9
```

Outputs go to `outputs/`. Start with `outputs/SUMMARY.txt`, `outputs/report.html`, or `outputs/report.pdf`.

## Architecture

```
scripts/run_analysis.py     → CLI
scripts/fit_grm.py          → alias for full pipeline
src/judgecheck/
  pipeline.py               → orchestration (add new steps here)
  data.py                   → HuggingFace loaders, matrix prep
  grm.py                    → GRM fit, EAP abilities, test information
  insights.py               → weak items, pairwise winner agreement
  config.py                 → AnalysisConfig(coverage, weak_item_count)
  summary.py                → SUMMARY.txt + console bullets
  viz.py                    → matplotlib plots
  report.py                 → HTML report
  pdf_report.py             → PDF export (summary + figures)
  metrics.py                → metrics.json manifest
```

**Rule for new features:** implement in a module → call from `pipeline.py` → expose CLI flag if needed → update `CHANGELOG.md` and output table in `README.md`.

## Data sources

| Source | Loader | Use |
|--------|--------|-----|
| `lmsys/mt_bench_human_judgments` | `load_mt_bench_judgments()` | Human + GPT-4 pairwise (splits: `human`, `gpt4_pair`) |
| `philschmid/mt-bench` | `load_mt_bench_questions()` | Question text, category |
| FastChat `gpt-4_single.jsonl` | `load_gpt4_single_scores()` | Cached in `data/gpt4_single.jsonl` |

## Modeling choices (do not “fix” without intent)

1. **Pairwise → ordinal 1–3:** `model_b=1`, `tie=2`, `model_a=3` for GRM (not true Likert).
2. **Score track:** 1–10 GPT-4 ratings; matrix is `(n_items × n_models)`.
3. **Judge θ:** custom EAP in `grm.py` (not `grm_mml_eap` — fails on Windows float128).
4. **GPT-4 pairwise:** 15 comparison slots per item as pseudo-participants.

## Key outputs

| File | Module |
|------|--------|
| `human_item_parameters.csv` | `run_pairwise_analysis` |
| `recommended_benchmark_items.csv` | `run_score_analysis` + `recommend_benchmark_items` |
| `recommended_pairwise_items.csv` | `run_pairwise_analysis` + `recommend_benchmark_items` |
| `recommended_items_overlap_*.csv` | `insights.compare_recommended_sets` (full run) |
| `recommended_category_comparison.csv` | `insights.compare_recommended_categories` (full run) |
| `pairwise_winner_agreement.csv` | `insights.pairwise_winner_agreement` |
| `pairwise_agreement_by_category.csv` | `insights.pairwise_agreement_by_category` |
| `pairwise_tie_rates_by_category.csv` | `insights.pairwise_tie_rates` |
| `category_discrimination_comparison.csv` | `grm.compare_category_discrimination` |
| `weak_benchmark_items.csv` | `insights.select_weak_items` |
| `human_judge_summary.csv` | `insights.summarize_judge_abilities` |
| `item_discrimination_agreement_detail.csv` | `insights.item_discrimination_agreement` |

## Version

Current: `judgecheck.__version__` in `src/judgecheck/__init__.py`. Log changes in `CHANGELOG.md`.

## Roadmap (not done)

- Additional LLM judges (need dataset with non–GPT-4 pairwise labels)

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python scripts/run_analysis.py --pairwise-only   # integration smoke test (~minutes)
```

CI: `.github/workflows/tests.yml` runs unit tests on push/PR to `main`.

## Citation

MT-Bench: [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). See `README.md` for BibTeX.
