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

    The default method follows the Endres-Stuebinger normalized log spread:

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

__all__ = ["build_spread"]
