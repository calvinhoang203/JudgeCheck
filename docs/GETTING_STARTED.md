# Getting Started with JudgeCheck

**No statistics or LLM background required.** This guide explains what JudgeCheck does and how to read the results.

---

## The one-sentence summary

JudgeCheck tells you **which benchmark questions actually help judges tell good AI answers from bad ones** — and **how reliable each judge is**.

---

## Step 1: Run the analysis

```powershell
cd JudgeCheck
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_analysis.py
```

Wait a few minutes for the full analysis (pairwise + 1–10 scores). Faster: `--pairwise-only` or `--scores-only`.

See [WHATS_NEW.md](WHATS_NEW.md) for what each part does.

---

## Step 2: Read the summary (fastest path)

Open either:

```
outputs/SUMMARY.txt     ← plain text, works everywhere
outputs/report.html     ← visual report with charts
```

Double-click it, or drag it into a browser window. You will see:

- Plain-language explanations
- Headline numbers (human vs GPT-4)
- Lists of the **sharpest** and **weakest** benchmark questions (with real prompt text)
- Charts you can screenshot for slides or papers

**No Python needed to read the report.**

---

## Step 3: What the numbers mean

### Discrimination (the most important number)

| Value | Plain meaning |
|-------|---------------|
| **High** (e.g. 3+) | This question sharply separates good vs bad answers. Keep it in your benchmark. |
| **Medium** (~0.5–1.5) | Moderately useful. |
| **Low** (~0.2) | Judges often can't tell answers apart on this question. Consider revising or removing it. |

**Analogy:** A math question where everyone scores 100% is a bad test question. A question where scores spread out is a good one. Discrimination measures that for LLM benchmarks.

### Judge ability (θ) — human experts only

| Value | Plain meaning |
|-------|---------------|
| **Higher θ** | This annotator gives more decisive, consistent preferences across questions. |
| **Lower θ** | More neutral or inconsistent ratings. |

This does **not** mean "correct" vs "incorrect" — it measures consistency on the latent scale, not agreement with ground truth.

### Human vs GPT-4 comparison

- **Mean discrimination** — which judge system produces sharper separations overall?
- **Spearman correlation** — do humans and GPT-4 agree on *which* questions are sharp? (1 = perfect agreement, 0 = none)

---

## Step 4: Output files (if you want the raw data)

| File | What's inside |
|------|---------------|
| `report.html` | Visual summary for anyone |
| `human_item_parameters.csv` | Every question's discrimination + thresholds |
| `human_judge_abilities.csv` | Ranked list of human annotators by θ |
| `human_category_summary.csv` | Average discrimination by topic (Writing, Math, …) |
| `judge_comparison.csv` | Human vs GPT-4 side-by-side |
| `*.png` | Charts embedded in the report |

### Test information (Part B)

The **test information curve** (`score_test_information.png`) shows where on the quality scale your benchmark works hardest. The peak θ is where the test best separates models.

---

## FAQ

**Q: Why 1–3 scores instead of 1–5 stars?**  
A: MT-Bench human data is pairwise (pick A or B), not star ratings. We map: B wins → 1, tie → 2, A wins → 3.

**Q: What is a GRM?**  
A: Graded Response Model — a standard statistical model for ordered categories (like Likert scales). It estimates item quality and judge ability jointly.

**Q: Can I use my own judge data?**  
A: Not yet out of the box, but the library expects a table with: judge ID, item ID, ordinal response. Pairwise or absolute scores both work with minor preprocessing.

**Q: I'm writing a paper — what do I cite?**  
A: MT-Bench paper (see README) + mention IRT/GRM via the `girth` package.

---

## Next steps

- Open `notebooks/01_explore_and_fit_grm.ipynb` for an interactive walkthrough
- Read `README.md` for project structure and roadmap
- Star the repo: [github.com/calvinhoang203/JudgeCheck](https://github.com/calvinhoang203/JudgeCheck)
