"""Gaussian/Brownian OU estimator."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..data import DEFAULT_LOBSTER_DATA, load_pair_prices_from_lobster
from ..spreads import build_spread


def ar1_to_brownian_ou_result(
    a: float,
    phi: float,
    eps_var: float,
    delta: float,
    metadata: dict[str, Any],
    fit_seconds: float | None = None,
) -> dict[str, Any]:
    """Convert AR(1) estimates into continuous-time Brownian OU parameters."""
    metadata = {key: value for key, value in metadata.items() if key not in {"valid", "spread"}}
    result: dict[str, Any] = {
        "valid": True,
        **metadata,
        "delta": float(delta),
        "estimation_method": "numpy_lstsq",
        "a": float(a),
        "phi": float(phi),
        "eps_var": float(eps_var),
    }
    if fit_seconds is not None:
        result["fit_seconds"] = float(fit_seconds)

    if not (0 < phi < 1):
        result.update(
            {
                "valid": False,
                "reason": "Reject: AR(1) phi must be between 0 and 1 for mean reversion.",
            }
        )
        return result

    theta = float(-np.log(phi) / delta)
    mu = float(a / (1 - phi))
    sigma = float(np.sqrt(eps_var * 2 * theta / (1 - phi**2)))
    sigma_eq = float(np.sqrt(eps_var / (1 - phi**2)))
    half_life = float(np.log(2) / theta)

    result.update(
        {
            "theta": theta,
            "kappa": theta,
            "mu": mu,
            "sigma": sigma,
            "sigma_eq": sigma_eq,
            "half_life_minutes": half_life,
        }
    )
    return result


def fit_brownian_ou_from_spread(
    spread: np.ndarray,
    metadata: dict[str, Any] | None = None,
    delta: float = 1.0,
) -> dict[str, Any]:
    """Fast Brownian OU fit from an already-built spread using NumPy OLS."""
    metadata = {key: value for key, value in (metadata or {}).items() if key not in {"valid", "spread"}}
    x = np.asarray(spread, dtype=float)
    x = x[np.isfinite(x)]
    x_lag = x[:-1]
    x_next = x[1:]

    started = perf_counter()
    design = np.column_stack([np.ones(len(x_lag)), x_lag])
    a, phi = np.linalg.lstsq(design, x_next, rcond=None)[0]
    residuals = x_next - (a + phi * x_lag)
    eps_var = float(np.var(residuals, ddof=1))
    elapsed = perf_counter() - started

    return ar1_to_brownian_ou_result(
        a=float(a),
        phi=float(phi),
        eps_var=eps_var,
        delta=delta,
        metadata=metadata,
        fit_seconds=elapsed,
)

def estimate_brownian_ou(
    ticker_a: str,
    ticker_b: str,
    formation_start: str,
    formation_end: str,
    data_path: str | Path = DEFAULT_LOBSTER_DATA,
    price_col: str = "model_price_close",
    delta: float = 1.0,
    min_observations: int = 100,
) -> dict[str, Any]:
    """
    Estimate the Brownian/Gaussian OU benchmark for one pair and formation period.
    The spread is:
        X_t = log(M_A(t) / M_A(0)) - log(M_B(t) / M_B(0)) 
        
        and the discrete AR(1) fit is:
        X_{t+1} = a + phi X_t + eps_t
    """
    price_data = load_pair_prices_from_lobster(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        formation_start=formation_start,
        formation_end=formation_end,
        data_path=data_path,
        price_col=price_col,
        min_observations=min_observations,
    )
    if not price_data["valid"]:
        return price_data

    price_a = price_data.pop("price_a")
    price_b = price_data.pop("price_b")
    spread = build_spread(price_a, price_b, method="normalized_log")
    metadata = {
        **price_data,
        "spread_start": float(spread[0]),
        "spread_end": float(spread[-1]),
        "spread_mean": float(np.mean(spread)),
        "spread_std": float(np.std(spread, ddof=1)),
    }
    return fit_brownian_ou_from_spread(
        spread=spread,
        metadata=metadata,
        delta=delta,
    )


# Backwards-compatible private alias used by older code.
_fit_brownian_ou_from_spread = fit_brownian_ou_from_spread

__all__ = [
    "ar1_to_brownian_ou_result",
    "estimate_brownian_ou",
    "fit_brownian_ou_from_spread",
]
