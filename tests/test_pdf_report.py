"""Unit tests for PDF report export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from judgecheck.pdf_report import generate_pdf_report


class TestPdfReport(unittest.TestCase):
    def test_generate_pdf_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "SUMMARY.txt").write_text(
                "JudgeCheck Run Summary\nVersion: 0.3.5\nTest line\n",
                encoding="utf-8",
            )
            plt.figure(figsize=(4, 3))
            plt.plot([1, 2, 3], [1, 4, 2])
            plt.savefig(output_dir / "human_top_discrimination.png")
            plt.close()

            path = generate_pdf_report(output_dir, mode="pairwise")
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 500)

    def test_generate_pdf_report_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_pdf_report(Path(tmp), mode="pairwise")
            self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
