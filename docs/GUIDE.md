# Reading JudgeCheck outputs

## Run

```powershell
python scripts/run_analysis.py
```

Then open `outputs/SUMMARY.txt` or `outputs/report.html`.

## Discrimination (*a*)

| Range | Interpretation |
|-------|----------------|
| High (~2+) | Question separates responses well — keep in benchmark |
| Low (~0.2) | Judges rarely distinguish answers — consider revising |

## Judge ability (θ)

Estimated for human annotators only. Higher θ = more decisive/consistent on the latent scale (not “correctness” vs ground truth).

## Human vs GPT-4

- **Mean discrimination** — overall sharpness of each judge system
- **Spearman *r*** (in `judge_comparison.csv`) — correlation of item discriminations across questions
- **Winner agreement** (in `pairwise_winner_agreement.csv`) — % same A/B winner on shared comparisons
- **By category** (in `pairwise_agreement_by_category.csv`) — which topic areas humans and GPT-4 disagree most

## Score track (1–10)

GPT-4 rates each model per question. `score_test_information.png` shows where on the quality scale (θ) the benchmark is most informative (peak θ).

## Pairwise coding

MT-Bench human data is A-vs-B, not star ratings:

| Outcome | Code |
|---------|------|
| model_b wins | 1 |
| tie | 2 |
| model_a wins | 3 |

## FAQ

**GRM?** Graded Response Model — IRT for ordered categories.

**Own data?** Needs a matrix `(items × judges)` of integer ordinal scores; wire through `prepare_grm_matrix` / `fit_grm`.
