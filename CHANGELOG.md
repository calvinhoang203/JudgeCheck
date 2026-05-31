# Changelog

All notable JudgeCheck updates are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] — 2026-05-30

### Added
- **GPT-4 score GRM (1–10)** — full polytomous model on single-answer MT-Bench ratings across 34 models
- **Test information curve** — shows where the benchmark best separates model quality (θ)
- **Method comparison** — correlates pairwise vs score-based item discrimination
- **`judgecheck.pipeline`** — central orchestration module for consistent future updates
- **`scripts/run_analysis.py`** — main CLI with `--pairwise-only` / `--scores-only` flags
- **`docs/WHATS_NEW.md`** — plain-language summary of each release

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
