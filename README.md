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

Options: `--pairwise-only`, `--scores-only`, `--coverage 0.9` (default 0.8).

Outputs: `outputs/SUMMARY.txt`, `outputs/report.html`, CSVs and PNGs.

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
| `human_item_parameters.csv` | Item discrimination (human pairwise) |
| `human_judge_abilities.csv` | Annotator θ estimates |
| `score_item_parameters.csv` | Item discrimination (1–10 scores) |
| `recommended_benchmark_items.csv` | High-information question subset |
| `weak_benchmark_items.csv` | Low discrimination (pairwise) |
| `pairwise_winner_agreement.csv` | Human vs GPT-4 winner agreement |

## IRT mapping

| IRT | JudgeCheck |
|-----|------------|
| Item | MT-Bench question + turn |
| Person | Judge (human) or model (score track) |
| Discrimination *a* | How sharply the item separates quality |

Pairwise MT-Bench labels are comparisons, not 1–5 stars; see [docs/GUIDE.md](docs/GUIDE.md).

## License

MIT (code). MT-Bench data: [CC-BY-4.0](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments).
