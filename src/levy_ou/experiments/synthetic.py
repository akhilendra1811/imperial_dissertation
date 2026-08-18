"""Synthetic data helpers used by smoke and example runners."""

from __future__ import annotations

import numpy as np
import pandas as pd

from levy_ou.spreads import build_spread


def synthetic_ou_spread(
    n: int,
    seed: int,
    rho: float = 0.97,
    innovation_sigma: float = 0.002,
    mean: float = 0.0,
) -> np.ndarray:
    """Generate a simple AR(1)/OU-like spread path."""

    rng = np.random.default_rng(seed)
    x = np.zeros(int(n), dtype=float)
    x[0] = float(mean)
    for idx in range(1, len(x)):
        x[idx] = float(mean) + float(rho) * (x[idx - 1] - float(mean)) + rng.normal(scale=innovation_sigma)
    return x


def synthetic_prices_from_spread(spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct positive prices whose normalized-log spread equals ``spread``."""

    x = np.asarray(spread, dtype=float)
    centered = x - x[0]
    price_b = np.full(len(centered), 50.0, dtype=float)
    price_a = 100.0 * np.exp(centered)
    return price_a, price_b


def synthetic_trading_frame(spread: np.ndarray, seed: int = 123) -> pd.DataFrame:
    """Return a model-ready synthetic trading frame for trade replay."""

    price_a, price_b = synthetic_prices_from_spread(spread)
    rebuilt_spread = build_spread(price_a, price_b)
    rng = np.random.default_rng(seed)
    spread_bps = rng.uniform(2.0, 8.0, size=len(rebuilt_spread)) / 10000.0
    bid_a = price_a * (1.0 - spread_bps / 2.0)
    ask_a = price_a * (1.0 + spread_bps / 2.0)
    bid_b = price_b * (1.0 - spread_bps / 2.0)
    ask_b = price_b * (1.0 + spread_bps / 2.0)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01 14:30", periods=len(rebuilt_spread), freq="min"),
            "trade_date": ["2024-01-01"] * len(rebuilt_spread),
            "market_time": [f"t{idx}" for idx in range(len(rebuilt_spread))],
            "spread": rebuilt_spread,
            "mid_a": price_a,
            "mid_b": price_b,
            "bid_a": bid_a,
            "ask_a": ask_a,
            "bid_b": bid_b,
            "ask_b": ask_b,
        }
    )


__all__ = [
    "synthetic_ou_spread",
    "synthetic_prices_from_spread",
    "synthetic_trading_frame",
]
