# JudgeCheck

IRT diagnostics for LLM-as-judge evaluation on MT-Bench. Estimates **item discrimination** (which benchmark questions separate good vs bad responses) using a **Graded Response Model (GRM)** via [`girth`](https://github.com/eribean/girth).

**Continuing development:** see [AGENTS.md](AGENTS.md).  
**Reading results:** see [docs/GUIDE.md](docs/GUIDE.md).  
**Version history:** [CHANGELOG.md](CHANGELOG.md).

## Setup

```powershell
cd JudgeCheck
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python scripts/run_analysis.py
```

Options: `--pairwise-only`, `--scores-only`, `--coverage 0.9` (default 0.8), `--no-pdf`.

Outputs: `outputs/SUMMARY.txt`, `outputs/report.html`, `outputs/report.pdf`, `outputs/metrics.json`, CSVs and PNGs.

Rebuild PDF only (no re-analysis):

```powershell
python scripts/export_pdf.py
```

Compare two runs:

```powershell
python scripts/compare_metrics.py outputs/run_a/metrics.json outputs/run_b/metrics.json
```

## Two analysis tracks

| Track | Data | Response scale |
|-------|------|----------------|
| **Pairwise** | Human experts + GPT-4 A/B judgments | 1–3 (B / tie / A) |
| **Scores** | GPT-4 single-answer ratings, 34 models | 1–10 |

## Project layout

```
src/judgecheck/     library
scripts/            run_analysis.py (main), fit_grm.py (alias)
notebooks/          01 explore pairwise; 02 score GRM
outputs/            generated (gitignored)
data/               cached gpt4_single.jsonl (gitignored)
```

## Main output files

| File | Description |
|------|-------------|
| `SUMMARY.txt` | Text headline results |
| `report.html` | HTML report with figures |
| `report.pdf` | PDF summary + figures (shareable) |
| `metrics.json` | Machine-readable headline metrics |
| `human_item_parameters.csv` | Item discrimination (human pairwise) |
| `human_judge_abilities.csv` | Annotator θ estimates + workload |
| `human_judge_summary.csv` | Aggregate human judge θ statistics |
| `score_item_parameters.csv` | Item discrimination (1–10 scores) |
| `recommended_benchmark_items.csv` | High-information question subset (score track) |
| `recommended_pairwise_items.csv` | High-information question subset (human pairwise) |
| `recommended_items_overlap_detail.csv` | Pairwise vs score recommended-set overlap (full run) |
| `recommended_category_comparison.csv` | Recommended-set topic mix: pairwise vs scores (full run) |
| `weak_benchmark_items.csv` | Low discrimination (pairwise) |
| `pairwise_winner_agreement.csv` | Human vs GPT-4 winner agreement |
| `pairwise_tie_rates_by_category.csv` | Human vs GPT-4 tie rates by category |
| `item_discrimination_agreement_detail.csv` | Sharp-item flags (human vs GPT-4 top tier) |
| `pairwise_agreement_by_category.csv` | Winner agreement by MT-Bench category |
| `category_discrimination_comparison.csv` | Human vs GPT-4 discrimination by category |

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

GitHub Actions runs the same tests on push/PR to `main`.

## IRT mapping

| IRT | JudgeCheck |
|-----|------------|
| Item | MT-Bench question + turn |
| Person | Judge (human) or model (score track) |
| Discrimination *a* | How sharply the item separates quality |

Pairwise MT-Bench labels are comparisons, not 1–5 stars; see [docs/GUIDE.md](docs/GUIDE.md).

## License

MIT (code). MT-Bench data: [CC-BY-4.0](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments).
