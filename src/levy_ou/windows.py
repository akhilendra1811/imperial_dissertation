"""Rolling-window Metadata shared by runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairWindow:
    """Minimal metadata for one pair-window experiment."""

    window_id: int
    ticker_a: str
    ticker_b: str
    formation_start: str
    formation_end: str
    trading_start: str
    trading_end: str

