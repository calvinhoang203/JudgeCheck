# Changelog

All notable JudgeCheck updates are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.6] — 2026-05-30

### Added
- **Category discrimination comparison** — human vs GPT-4 mean discrimination by MT-Bench topic (`category_discrimination_comparison.csv`, plot, report table)

## [0.3.5] — 2026-05-30

### Added
- **`report.pdf`** — summary text + key figures, written after each run (matplotlib)
- **`scripts/export_pdf.py`** — rebuild PDF from existing `outputs/` without re-running analysis
- **`--no-pdf`** CLI flag to skip PDF export

## [0.3.4] — 2026-05-30

### Added
- **Category-level winner agreement** — `pairwise_agreement_by_category.csv` and report table
- **`tests/`** — unit tests for insights and data matrix prep (`python -m unittest discover -s tests`)

### Fixed
- **`run_pairwise_analysis`** — HTML report no longer referenced undefined `pairwise` when building winner agreement rate

## [0.3.3] — 2026-05-30

### Changed
- Added **AGENTS.md** for continuing work in new sessions (architecture, data, conventions)
- Replaced verbose docs with **docs/GUIDE.md**; removed redundant `GETTING_STARTED.md` and `WHATS_NEW.md`
- Trimmed README, HTML report, and SUMMARY marketing copy

## [0.3.2] — 2026-05-30

### Added
- **`--coverage` flag** — tune how much information the recommended question set must cover (default 80%)
- **`weak_benchmark_items.csv` / `weak_score_items.csv`** — lowest-discrimination questions to revise or drop
- **Pairwise winner agreement** — human majority vs GPT-4 on the same A/B comparisons (`pairwise_winner_agreement.csv`)
- **`AnalysisConfig`** — shared settings object for consistent pipeline updates

## [0.3.1] — 2026-05-30

### Added
- **Benchmark designer (lite)** — `recommended_benchmark_items.csv` lists the smallest question set covering ~80% of diagnostic information
- **`SUMMARY.txt`** — plain-text headline findings (no browser or Python needed)
- **`model_score_ranking.csv`** — models ranked by mean GPT-4 score
- **Console quick findings** — three bullet points printed after each run

## [0.3.0] — 2026-05-30

### Added
- **GPT-4 score GRM (1–10)** — full polytomous model on single-answer MT-Bench ratings across 34 models
- **Test information curve** — shows where the benchmark best separates model quality (θ)
- **Method comparison** — correlates pairwise vs score-based item discrimination
- **`judgecheck.pipeline`** — central orchestration module for consistent future updates
- **`scripts/run_analysis.py`** — main CLI with `--pairwise-only` / `--scores-only` flags
- **`docs/WHATS_NEW.md`** — release notes (later removed; use CHANGELOG)

### Changed
- `fit_grm.py` now delegates to the full pipeline (pairwise + scores)
- HTML report split into **Part A** (pairwise) and **Part B** (1–10 scores)
- `fit_grm()` accepts custom `valid_responses` for different rating scales

## [0.2.0] — 2026-05-29

### Added
- Question text and category labels on all outputs
- Human judge ability (θ) estimates via EAP
- Category-level discrimination summaries
- Browser-readable `outputs/report.html`
- `docs/GETTING_STARTED.md`

## [0.1.0] — 2026-05-29

### Added
- Initial MT-Bench pairwise GRM analysis (human + GPT-4)
- Core library, notebook, and README
