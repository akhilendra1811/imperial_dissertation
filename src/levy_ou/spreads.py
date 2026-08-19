"""Spread construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_spread(
    price_a: pd.Series | np.ndarray,
    price_b: pd.Series | np.ndarray,
    method: str = "normalized_log",
    beta: float | None = None,
) -> np.ndarray:
    """
    Build a pair spread from two aligned price series.

    The default method follows the Endres-Stuebinger normalised log spread:

        X_t = log(M_A(t) / M_A(0)) - log(M_B(t) / M_B(0))

    Levy-OU estimators reuse this function so the Brownian and
    Levy models are fitted to the exact same spread definition.
    """
    a = np.asarray(price_a, dtype=float)
    b = np.asarray(price_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a = a[valid]
    b = b[valid]
    if len(a) == 0:
        return np.array([], dtype=float)
    if len(b) == 0:
        return np.array([], dtype=float)

    log_a = np.log(a)
    log_b = np.log(b)
    if method == "normalized_log":
        return (log_a - log_a[0]) - (log_b - log_b[0])
    if method == "hedge_ratio_log":
        if beta is None:
            raise ValueError("beta is required when method='hedge_ratio_log'.")
        return log_a - beta * log_b
    raise ValueError(f"Unknown spread method: {method}")


def build_spread_with_anchor(
    price_a: pd.Series | np.ndarray,
    price_b: pd.Series | np.ndarray,
    anchor_a: float,
    anchor_b: float,
) -> np.ndarray:
    """Build the normalized log spread using explicit price anchors.

    This is the rolling-backtest version of the Endres-Stuebinger spread. Use
    it when formation and trading windows must share the same coordinate
    system, typically the first valid formation observation.
    """

    anchor_a = float(anchor_a)
    anchor_b = float(anchor_b)
    if not (np.isfinite(anchor_a) and np.isfinite(anchor_b) and anchor_a > 0.0 and anchor_b > 0.0):
        raise ValueError("anchor_a and anchor_b must be finite positive prices.")

    a = np.asarray(price_a, dtype=float)
    b = np.asarray(price_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a = a[valid]
    b = b[valid]
    if len(a) == 0:
        return np.array([], dtype=float)
    return np.log(a / anchor_a) - np.log(b / anchor_b)


__all__ = ["build_spread", "build_spread_with_anchor"]
