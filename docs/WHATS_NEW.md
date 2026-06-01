# What's New in JudgeCheck

Quick, plain-language release notes. For technical details see [CHANGELOG.md](../CHANGELOG.md).

---

## v0.3.1 — Benchmark designer & plain-text summary

**No new commands** — just run as usual:
```powershell
python scripts/run_analysis.py
```

### What's new?

1. **`outputs/SUMMARY.txt`** — open in Notepad for headline results in plain English.
2. **`recommended_benchmark_items.csv`** — which ~N questions carry 80% of the benchmark's diagnostic power (benchmark designer lite).
3. **`model_score_ranking.csv`** — simple leaderboard of models by mean GPT-4 score.
4. **Quick findings in the terminal** — three bullets at the end of each run.

---

## v0.3 — Score ratings & test information

**Run everything:**
```powershell
python scripts/run_analysis.py
```

### What's new?

1. **True 1–10 star-style GRM**  
   MT-Bench also has GPT-4 giving each model a score from 1 to 10. We now fit a full 10-level GRM on 34 models × 160 questions — closer to classic psychometrics.

2. **Test information curve**  
   Answers: *"Is this benchmark better at telling apart mediocre vs good models, or good vs excellent ones?"*  
   Look at `outputs/score_test_information.png` — the peak is where the test works hardest.

3. **Two methods, one report**  
   Part A = human pairwise (A vs B). Part B = GPT-4 scores (1–10).  
   We compare whether they agree on which questions are sharpest.

4. **Cleaner project structure**  
   All analysis logic lives in `src/judgecheck/pipeline.py`. Scripts just call the pipeline — easier to extend in the next update.

### New files to look at

| File | What it is |
|------|------------|
| `score_item_parameters.csv` | Discrimination per question (score-based) |
| `score_test_information.csv` | Information curve data |
| `method_discrimination_comparison.csv` | Pairwise vs score discrimination per item |
| `score_test_information.png` | Visual: where the benchmark is most informative |

### Options

```powershell
python scripts/run_analysis.py --pairwise-only   # Part A only (faster)
python scripts/run_analysis.py --scores-only     # Part B only (~2 min for GRM)
```

---

## v0.2 — Labels, judge ability, HTML report

- Real question text on charts (not `129_t1`)
- Human annotator consistency rankings (θ)
- Open `outputs/report.html` in any browser

---

## v0.1 — First GRM fit

- Pairwise human vs GPT-4 analysis on MT-Bench
- Basic discrimination charts and CSVs

---

## Coming next (v0.4 ideas)

- Additional LLM judges (Claude pairwise, etc.)
- Benchmark designer: "top 20 items carry 80% of information"
- Optional PDF export from the HTML report
