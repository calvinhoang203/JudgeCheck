"""Build PDF report from SUMMARY.txt and pipeline figures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from judgecheck import __version__

ReportMode = Literal["full", "pairwise", "scores"]

PAIRWISE_FIGURES: list[tuple[str, str]] = [
    ("Sharpest questions (human)", "human_top_discrimination.png"),
    ("Human vs GPT-4 discrimination", "human_vs_gpt4_discrimination.png"),
    ("By category (human)", "human_category_discrimination.png"),
    ("Human judge ability", "human_judge_abilities.png"),
]

SCORE_FIGURES: list[tuple[str, str]] = [
    ("Sharpest questions (scores)", "score_top_discrimination.png"),
    ("Test information curve", "score_test_information.png"),
    ("Pairwise vs score discrimination", "pairwise_vs_score_discrimination.png"),
]

_LINES_PER_PAGE = 52


def _figure_list(mode: ReportMode) -> list[tuple[str, str]]:
    figures: list[tuple[str, str]] = []
    if mode in ("full", "pairwise"):
        figures.extend(PAIRWISE_FIGURES)
    if mode in ("full", "scores"):
        figures.extend(SCORE_FIGURES)
    return figures


def _text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(
        0.5,
        0.96,
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.90,
        body,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        linespacing=1.35,
    )
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.98)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_pdf_report(
    output_dir: Path,
    *,
    mode: ReportMode = "full",
    output_path: Path | None = None,
) -> Path | None:
    """
    Write ``report.pdf`` from ``SUMMARY.txt`` and PNG figures in ``output_dir``.

    Returns the PDF path, or ``None`` if there is nothing to export.
    """
    output_dir = Path(output_dir)
    output_path = output_path or output_dir / "report.pdf"
    summary_path = output_dir / "SUMMARY.txt"

    existing = [
        (title, output_dir / name)
        for title, name in _figure_list(mode)
        if (output_dir / name).exists()
    ]
    if not summary_path.exists() and not existing:
        return None

    report_title = f"JudgeCheck Report (v{__version__})"

    with PdfPages(output_path) as pdf:
        if summary_path.exists():
            lines = summary_path.read_text(encoding="utf-8").splitlines()
            for page_idx in range(0, len(lines), _LINES_PER_PAGE):
                chunk = lines[page_idx : page_idx + _LINES_PER_PAGE]
                page_title = report_title if page_idx == 0 else f"{report_title} (cont.)"
                _text_page(pdf, page_title, "\n".join(chunk))
        for fig_title, path in existing:
            _image_page(pdf, fig_title, path)

    return output_path
