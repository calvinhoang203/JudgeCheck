# JudgeCheck

**Diagnostic IRT tooling for LLM-as-judge systems** — built for researchers who want rigorous psychometrics, explained clearly enough for anyone evaluating benchmark pipelines.

> *Is your LLM judge actually measuring what you think it is? JudgeCheck uses Item Response Theory (IRT) to find out.*

---

## What problem does this solve?

When we use one LLM to score another (LLM-as-judge), we usually report aggregate accuracy or agreement with humans. That tells us **how much** the judge agrees, but not **where** it fails or **which benchmark questions** are doing real work.

**JudgeCheck** reframes evaluation through [Item Response Theory](https://en.wikipedia.org/wiki/Item_response_theory):

| Classic IRT (education) | JudgeCheck (LLM evaluation) |
|------------------------|----------------------------|
| Test questions (items) | MT-Bench benchmark prompts |
| Student ability (θ)    | Judge skill / consistency  |
| Response (correct/incorrect or Likert) | Pairwise preference → ordinal score |

The **Graded Response Model (GRM)** extends IRT to multi-level ratings (here, 3-level pairwise preferences). Its key output is **item discrimination (a)**: how strongly each benchmark question separates high- vs low-quality responses *as seen by the judge*.

- **High discrimination** → the question is a sharp probe; judges (or models) consistently rank responses differently on it.
- **Low discrimination** → the question is a weak probe; most responses look similar to the judge.

---

## Quick start

### 1. Environment

```powershell
cd JudgeCheck
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Library choice:** We use [`girth`](https://github.com/eribean/girth) for GRM estimation (marginal maximum likelihood). [`irtorch`](https://irtorch.readthedocs.io/) offers PyTorch-based GRM but adds heavy dependencies; [`catsim`](https://douglasrizzo.com.br/catsim/) focuses on adaptive testing simulation and dichotomous logistic models, not polytomous GRM fitting.

### 2. Run the first analysis

```powershell
python scripts/fit_grm.py
```

This will:

1. Download [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) (~3.3K human + 2.4K GPT-4 pairwise judgments across 80 MT-Bench questions × 2 turns).
2. Map pairwise outcomes to ordinal scores: `model_b → 1`, `tie → 2`, `model_a → 3`.
3. Fit GRM models for **human expert judges** and **GPT-4 as judge**.
4. Write CSVs and plots to `outputs/`.

### 3. Interactive notebook

```powershell
jupyter notebook notebooks/01_explore_and_fit_grm.ipynb
```

---

## Data note (important for interpretation)

MT-Bench human annotations are **pairwise comparisons**, not direct 1–5 quality ratings. We convert each judgment to a 3-level ordinal preference so GRM machinery applies cleanly. This is a deliberate modeling choice documented in the notebook — future versions can incorporate absolute score scales when available.

---

## Project layout

```
JudgeCheck/
├── src/judgecheck/       # Core library
│   ├── data.py           # Load & reshape MT-Bench judgments
│   ├── grm.py            # GRM fitting & comparison
│   └── viz.py            # Discrimination plots
├── scripts/fit_grm.py    # One-command first analysis
├── notebooks/            # Learning-oriented walkthroughs
├── outputs/              # Generated figures & CSVs (gitignored)
└── requirements.txt
```

---

## Reading the outputs

| File | Meaning |
|------|---------|
| `human_item_parameters.csv` | Per-question discrimination & threshold parameters (human judges) |
| `gpt4_item_parameters.csv` | Same for GPT-4 judge |
| `judge_comparison.csv` | Mean discrimination, AIC/BIC across judge systems |
| `human_top_discrimination.png` | Which questions human experts use most to separate models |
| `human_vs_gpt4_discrimination.png` | Do GPT-4 and humans agree on “hard” vs “easy” items? |

---

## Roadmap

- [ ] Absolute 1–5 / 1–10 score GRM when score-labeled data is available
- [ ] Judge ability (θ) estimation and calibration plots
- [ ] Test information functions per judge system
- [ ] CLI / web dashboard for non-Python users

---

## Citation

If you use MT-Bench data, cite:

```bibtex
@misc{zheng2023judging,
  title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
  author={Lianmin Zheng and Wei-Lin Chiang and others},
  year={2023},
  eprint={2306.05685},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

## License

MIT (code). MT-Bench judgment data: [CC-BY-4.0](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments).
