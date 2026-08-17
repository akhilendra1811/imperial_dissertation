"""Bertram Gaussian/Brownian OU threshold benchmark.

This is the Bertram-style symmetric band-to-opposite-band benchmark . 
 The fitted Brownian OU is

    dX_t = alpha * (mu - X_t) dt + eta dW_t

and the threshold optimiser chooses a symmetric distance ``b`` so that the
strategy enters at ``mu +/- b`` and exits at the opposite band.

The cost ``c`` is a completed pair round-trip cost in spread/PnL units.  For an
Endres-style 5 bps per stock leg per half-turn convention and one dollar on
each leg, this is approximately ``4 * 0.0005 = 0.002``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from levy_ou.backtesting.execution import ENDRES_HALF_TURN_RATE
from levy_ou.backtesting.gaussian_ou_numerics import erfi, maximize_bounded
from levy_ou.backtesting.trade_replay import trade_band_window


def pair_round_trip_c_from_half_turn_bps(
    half_turn_bps: float = 5.0,
    legs: int = 2,
    half_turns: int = 2,
) -> float:
    """Map per-leg half-turn bps to pair round-trip spread/PnL cost ``c``.

    With one dollar on each leg, 5 bps per leg at entry and again at exit gives
    ``2 legs * 2 half-turns * 0.0005 = 0.002``.
    """

    rate = float(half_turn_bps) / 10000.0
    return float(int(legs) * int(half_turns) * rate)


def endres_5bps_round_trip_c() -> float:
    """Return the Endres-style 5 bps per leg per half-turn round-trip cost."""

    return pair_round_trip_c_from_half_turn_bps(half_turn_bps=ENDRES_HALF_TURN_RATE * 10000.0)


def bertram_objective(b: float, alpha: float, eta: float, c: float) -> float:
    """Expected net log-spread return per time step for symmetric boundary ``b``."""

    if not (np.isfinite(b) and np.isfinite(alpha) and np.isfinite(eta) and np.isfinite(c)):
        return 0.0
    if b <= 0.0 or alpha <= 0.0 or eta <= 0.0 or c < 0.0 or 2.0 * b <= c:
        return 0.0
    z = float(b) * math.sqrt(float(alpha)) / float(eta)
    denominator = 2.0 * math.pi * float(erfi(z))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(float(alpha) * (2.0 * float(b) - float(c)) / denominator)


def solve_optimal_gaussian_ou_boundary(alpha: float, eta: float, c: float = 0.0) -> dict[str, Any]:
    """Solve the optimal symmetric Gaussian OU band distance.

    Parameters
    ----------
    alpha:
        OU mean-reversion speed. In this repo this is `gaussian_fit["theta"]`.
    eta:
        Brownian diffusion coefficient. In this repo this is
        `gaussian_fit["sigma"]`, not `sigma_eq`.
    c:
        Pair round-trip transaction cost in spread/PnL units.
    """

    if not (np.isfinite(alpha) and float(alpha) > 0.0):
        return {"threshold_valid": False, "threshold_reason": "invalid alpha"}
    if not (np.isfinite(eta) and float(eta) > 0.0):
        return {"threshold_valid": False, "threshold_reason": "invalid eta"}
    if not (np.isfinite(c) and float(c) >= 0.0):
        return {"threshold_valid": False, "threshold_reason": "invalid transaction cost c"}

    alpha = float(alpha)
    eta = float(eta)
    c = float(c)
    scale_b = eta / math.sqrt(alpha)
    z_min = c / (2.0 * scale_b)
    lower = max(z_min * (1.0 + 1e-10), z_min + 1e-12, 1e-12)
    upper = lower + 8.0

    result = maximize_bounded(
        lambda z: bertram_objective(float(z) * scale_b, alpha=alpha, eta=eta, c=c),
        lower=lower,
        upper=upper,
        xatol=1e-12,
        maxiter=1000,
    )
    z_star = float(result["x"])
    b_star = float(z_star * scale_b)
    objective = bertram_objective(b_star, alpha=alpha, eta=eta, c=c)
    erfi_value = float(erfi(z_star))
    expected_cycle_steps = float(2.0 * math.pi * erfi_value / alpha)
    lhs = float(math.exp(z_star * z_star) * (2.0 * b_star - c))
    rhs = float(scale_b * math.sqrt(math.pi) * erfi_value)
    foc_relative_error = float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300))
    sigma_eq = eta / math.sqrt(2.0 * alpha)

    valid = bool(
        result["success"]
        and np.isfinite(b_star)
        and b_star > c / 2.0
        and objective > 0.0
        and np.isfinite(expected_cycle_steps)
    )
    return {
        "threshold_valid": valid,
        "threshold_reason": "" if valid else str(result["message"]),
        "threshold_family": "bertram_gaussian",
        "alpha": alpha,
        "eta": eta,
        "sigma_eq": float(sigma_eq),
        "c": c,
        "c_bps_approx": float(c * 10000.0),
        "b_star": b_star,
        "b_star_sigma_eq": float(b_star / sigma_eq),
        "d_plus": b_star,
        "d_minus": b_star,
        "upper_entry_distance": b_star,
        "lower_entry_distance": b_star,
        "net_log_spread_per_completed_cycle": float(2.0 * b_star - c),
        "bertram_objective_per_step": objective,
        "expected_cycle_steps": expected_cycle_steps,
        "erfi_argument": z_star,
        "foc_relative_error": foc_relative_error,
        "optimizer_success": bool(result["success"]),
        "optimizer_iterations": int(result["nfev"]),
    }


def solve_optimal_gaussian_ou_from_fit(fit: dict[str, Any], c: float = 0.0) -> dict[str, Any]:
    """Solve the optimal boundary from a Gaussian OU fit dictionary."""

    if not bool(fit.get("valid", False)):
        return {
            "threshold_valid": False,
            "threshold_reason": f"invalid Gaussian fit: {fit.get('reason', '')}",
        }
    result = solve_optimal_gaussian_ou_boundary(
        alpha=float(fit["theta"]),
        eta=float(fit["sigma"]),
        c=float(c),
    )
    return {
        "mu": float(fit.get("mu", np.nan)),
        "theta": float(fit.get("theta", np.nan)),
        "sigma": float(fit.get("sigma", np.nan)),
        "fit_sigma_eq": float(fit.get("sigma_eq", np.nan)),
        **result,
    }


def trade_optimal_gaussian_ou_window(
    trading: pd.DataFrame,
    mu: float,
    b_star: float,
    ticker_a: str | None = None,
    ticker_b: str | None = None,
) -> list[dict[str, Any]]:
    """Replay the Bertram Gaussian OU band-to-opposite-band trading rule."""

    return trade_band_window(
        frame=trading,
        mean=float(mu),
        d_plus=float(b_star),
        d_minus=float(b_star),
        exit_rule="opposite_band",
        mean_mode="constant",
        ticker_a=ticker_a,
        ticker_b=ticker_b,
    )


__all__ = [
    "bertram_objective",
    "endres_5bps_round_trip_c",
    "pair_round_trip_c_from_half_turn_bps",
    "solve_optimal_gaussian_ou_boundary",
    "solve_optimal_gaussian_ou_from_fit",
    "trade_optimal_gaussian_ou_window",
]
