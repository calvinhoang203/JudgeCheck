"""Shared settings for JudgeCheck pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Tunable parameters — passed through pipeline functions."""

    coverage: float = 0.8
    weak_item_count: int = 15
    export_pdf: bool = True

    def __post_init__(self) -> None:
        if not 0.5 <= self.coverage <= 0.99:
            raise ValueError("coverage must be between 0.5 and 0.99")
