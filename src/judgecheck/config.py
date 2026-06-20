"""Shared settings for JudgeCheck pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Tunable parameters — passed through pipeline functions."""

    coverage: float = 0.8
    weak_item_count: int = 15
    export_pdf: bool = True
    sharp_item_pct: float = 0.25

    def __post_init__(self) -> None:
        if not 0.5 <= self.coverage <= 0.99:
            raise ValueError("coverage must be between 0.5 and 0.99")
        if not 0.05 <= self.sharp_item_pct <= 0.5:
            raise ValueError("sharp_item_pct must be between 0.05 and 0.5")
