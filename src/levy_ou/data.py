"""Data loading utilities for model-ready LOBSTER panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_LOBSTER_DATA = (
    "data/processed_lobster_all_tickers/lobster_minute_prices_model_ready.csv.gz"
)


def load_pair_prices_from_lobster(
    ticker_a: str,
    ticker_b: str,
    formation_start: str,
    formation_end: str,
    data_path: str | Path = DEFAULT_LOBSTER_DATA,
    price_col: str = "model_price_close",
    min_observations: int = 100,
) -> dict[str, Any]:
    """Load aligned model-ready prices for one pair and formation period."""
    ticker_a = ticker_a.upper()
    ticker_b = ticker_b.upper()

    usecols = ["timestamp_utc", "trade_date", "ticker", price_col, "model_ready_price"]
    raw = pd.read_csv(Path(data_path), compression="infer", usecols=usecols)
    raw["model_ready_price"] = raw["model_ready_price"].astype(str).str.lower().eq("true")
    raw = raw[
        raw["model_ready_price"]
        & raw["ticker"].isin([ticker_a, ticker_b])
        & raw["trade_date"].between(formation_start, formation_end)
    ].copy()

    metadata: dict[str, Any] = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "formation_start": formation_start,
        "formation_end": formation_end,
        "spread_definition": "log(M_A(t)/M_A(0)) - log(M_B(t)/M_B(0))",
        "price_column": price_col,
    }

    if raw.empty:
        return {
            "valid": False,
            **metadata,
            "reason": "No model-ready rows found for this pair and formation period.",
        }

    prices = (
        raw.pivot_table(
            index="timestamp_utc",
            columns="ticker",
            values=price_col,
            aggfunc="last",
        )
        .sort_index()
        .dropna(subset=[ticker_a, ticker_b])
    )

    observations = int(len(prices))
    if observations < min_observations:
        return {
            "valid": False,
            **metadata,
            "observations": observations,
            "reason": f"Too few overlapping observations; need at least {min_observations}.",
        }

    price_a = prices[ticker_a].astype(float)
    price_b = prices[ticker_b].astype(float)
    if (price_a <= 0).any() or (price_b <= 0).any():
        return {
            "valid": False,
            **metadata,
            "observations": observations,
            "reason": "Non-positive prices found; log spread cannot be constructed.",
        }

    return {
        "valid": True,
        **metadata,
        "observations": observations,
        "price_a": price_a.to_numpy(dtype=float),
        "price_b": price_b.to_numpy(dtype=float),
    }

__all__ = ["DEFAULT_LOBSTER_DATA", "load_pair_prices_from_lobster"]
