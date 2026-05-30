# JudgeCheck

**Diagnostic IRT tooling for LLM-as-judge systems** — built for researchers who want rigorous psychometrics, explained clearly enough for anyone evaluating benchmark pipelines.

> *Is your LLM judge actually measuring what you think it is? JudgeCheck uses Item Response Theory (IRT) to find out.*

**New here?** Read **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — no statistics background required.

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

- **High discrimination** → the question is a sharp probe; judges consistently rank responses differently on it.
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

### 2. Run the analysis

```powershell
python scripts/fit_grm.py
```

This will:

1. Download MT-Bench judgments and question text from HuggingFace.
2. Fit GRM models for **human experts** (with judge ability θ) and **GPT-4**.
3. Write labeled CSVs, charts, and **`outputs/report.html`** — open in any browser.

### 3. Read the report (no Python needed)

```
outputs/report.html
```

Plain-language summary with real question text, sharpest/weakest items, and human vs GPT-4 comparison.

### 4. Interactive notebook

```powershell
jupyter notebook notebooks/01_explore_and_fit_grm.ipynb
```

---

## What's included now

| Feature | Description |
|---------|-------------|
| **Labeled questions** | Plots show real prompts, not opaque IDs like `129_t1` |
| **Judge ability (θ)** | Rank human annotators by consistency |
| **Category summaries** | Which MT-Bench topics (Writing, Math, …) produce sharpest questions |
| **HTML report** | Shareable summary for advisors, collaborators, or blog readers |
| **Human vs GPT-4** | Side-by-side reliability and discrimination correlation |

---

## Project layout

```
JudgeCheck/
├── docs/GETTING_STARTED.md   # Plain-language guide
├── src/judgecheck/
│   ├── data.py               # Load judgments + question labels
│   ├── grm.py                # GRM fitting, judge ability
│   ├── viz.py                # Charts
│   └── report.py             # HTML report generator
├── scripts/fit_grm.py        # One-command analysis
├── notebooks/                # Interactive tutorial
└── outputs/                  # report.html, CSVs, PNGs (generated)
```

---

## Reading the outputs

| File | Meaning |
|------|---------|
| `report.html` | **Start here** — visual summary for any audience |
| `human_item_parameters.csv` | Per-question discrimination with prompt text |
| `human_judge_abilities.csv` | Human annotators ranked by θ |
| `human_category_summary.csv` | Mean discrimination by topic area |
| `judge_comparison.csv` | Human vs GPT-4 reliability metrics |
| `*.png` | Charts (embedded in the report) |

---

## Roadmap

- [x] Question labels on plots and CSVs
- [x] Judge ability (θ) estimation
- [x] HTML report for non-Python users
- [ ] Absolute 1–10 score GRM when score-labeled data is available
- [ ] Test information functions per judge system
- [ ] Additional LLM judges (Claude, Llama, …)

---

## Citation

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
