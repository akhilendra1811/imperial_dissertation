"""Zeng-Lee Gaussian OU optimal threshold rules."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from levy_ou.backtesting.gaussian_ou_numerics import erfi, maximize_bounded
from levy_ou.backtesting.optimal_gaussian import (
    endres_5bps_round_trip_c,
    pair_round_trip_c_from_half_turn_bps,
)
from levy_ou.backtesting.trade_replay import trade_band_window


ZENG_LEE_FOC_TOLERANCE = 1e-6


def zeng_lee_objective(
    b: float,
    alpha: float,
    eta: float,
    c: float,
    rule: str = "new",
) -> float:
    """Expected net return per time step for one Zeng-Lee rule."""

    if not all(np.isfinite(x) for x in (b, alpha, eta, c)):
        return 0.0
    if b <= 0.0 or alpha <= 0.0 or eta <= 0.0 or c < 0.0:
        return 0.0

    z = float(b) * math.sqrt(float(alpha)) / float(eta)
    denominator = math.pi * erfi(z)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0

    if rule == "conventional":
        if b <= c:
            return 0.0
        return float(2.0 * alpha * (b - c) / denominator)
    if rule == "new":
        if 2.0 * b <= c:
            return 0.0
        return float(alpha * (2.0 * b - c) / denominator)
    raise ValueError("rule must be 'conventional' or 'new'")


def solve_zeng_lee_gaussian_ou_boundary(
    alpha: float,
    eta: float,
    c: float = 0.0,
    rule: str = "new",
    foc_tolerance: float = ZENG_LEE_FOC_TOLERANCE,
) -> dict[str, Any]:
    """Solve the Zeng-Lee optimal Gaussian OU threshold."""

    if rule not in {"conventional", "new"}:
        return {"threshold_valid": False, "threshold_reason": "rule must be 'conventional' or 'new'"}
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
    minimum_b = c if rule == "conventional" else c / 2.0
    lower = max(minimum_b / scale_b + 1e-12, 1e-12)
    upper = lower + 8.0

    result = maximize_bounded(
        lambda z: zeng_lee_objective(float(z) * scale_b, alpha=alpha, eta=eta, c=c, rule=rule),
        lower=lower,
        upper=upper,
        xatol=1e-12,
        maxiter=1000,
    )

    z_star = float(result["x"])
    b_star = float(z_star * scale_b)
    objective = zeng_lee_objective(b_star, alpha=alpha, eta=eta, c=c, rule=rule)
    sigma_eq = eta / math.sqrt(2.0 * alpha)
    erfi_value = erfi(z_star)

    try:
        exp_z2 = math.exp(z_star * z_star)
    except OverflowError:
        exp_z2 = math.inf

    if rule == "conventional":
        gross_spread_move = b_star
        net_spread_move = b_star - c
        expected_cycle_steps = float(math.pi * erfi_value / (2.0 * alpha))
        lhs = float(2.0 * exp_z2 * (b_star - c))
    else:
        gross_spread_move = 2.0 * b_star
        net_spread_move = 2.0 * b_star - c
        expected_cycle_steps = float(math.pi * erfi_value / alpha)
        lhs = float(exp_z2 * (2.0 * b_star - c))

    rhs = float(scale_b * math.sqrt(math.pi) * erfi_value)
    foc_relative_error = float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300))
    success = bool(result["success"])
    valid = bool(
        success
        and np.isfinite(b_star)
        and b_star > minimum_b
        and objective > 0.0
        and np.isfinite(expected_cycle_steps)
        and expected_cycle_steps > 0.0
        and np.isfinite(foc_relative_error)
        and foc_relative_error <= float(foc_tolerance)
    )
    reason = "" if valid else str(result["message"] or "")
    if success and not valid and (not np.isfinite(objective) or objective <= 0.0):
        reason = "non-positive or non-finite Zeng-Lee objective"
    if success and not valid and np.isfinite(foc_relative_error) and foc_relative_error > float(foc_tolerance):
        reason = f"FOC relative error {foc_relative_error:.3g} exceeds tolerance {float(foc_tolerance):.3g}"
    if success and not valid and not np.isfinite(foc_relative_error):
        reason = "non-finite Zeng-Lee FOC relative error"

    return {
        "threshold_valid": valid,
        "threshold_reason": reason,
        "threshold_rule": rule,
        "threshold_family": "zeng_lee_gaussian",
        "zeng_lee_foc_tolerance": float(foc_tolerance),
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
        "gross_spread_move": float(gross_spread_move),
        "net_spread_move": float(net_spread_move),
        "expected_cycle_steps": expected_cycle_steps,
        "erfi_argument": z_star,
        "foc_relative_error": foc_relative_error,
        "equation_relative_error": foc_relative_error,
        "objective_per_step": float(objective),
        "zeng_lee_objective_per_step": float(objective),
        "optimizer_success": success,
        "optimizer_iterations": int(result["nfev"]),
    }


def solve_zeng_lee_gaussian_ou_from_fit(
    fit: dict[str, Any],
    c: float = 0.0,
    rule: str = "new",
    foc_tolerance: float = ZENG_LEE_FOC_TOLERANCE,
) -> dict[str, Any]:
    """Solve the Zeng-Lee threshold from a Gaussian OU fit."""

    if not bool(fit.get("valid", False)):
        return {
            "threshold_valid": False,
            "threshold_reason": f"invalid Gaussian fit: {fit.get('reason', '')}",
        }
    result = solve_zeng_lee_gaussian_ou_boundary(
        alpha=float(fit["theta"]),
        eta=float(fit["sigma"]),
        c=float(c),
        rule=rule,
        foc_tolerance=foc_tolerance,
    )
    return {
        "mu": float(fit.get("mu", np.nan)),
        "theta": float(fit.get("theta", np.nan)),
        "sigma": float(fit.get("sigma", np.nan)),
        "fit_sigma_eq": float(fit.get("sigma_eq", np.nan)),
        **result,
    }


def trade_zeng_lee_gaussian_ou_window(
    trading: pd.DataFrame,
    mu: float,
    b_star: float,
    ticker_a: str | None = None,
    ticker_b: str | None = None,
) -> list[dict[str, Any]]:
    """Replay the Zeng-Lee band-to-opposite-band trading rule."""

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
    "ZENG_LEE_FOC_TOLERANCE",
    "endres_5bps_round_trip_c",
    "pair_round_trip_c_from_half_turn_bps",
    "solve_zeng_lee_gaussian_ou_boundary",
    "solve_zeng_lee_gaussian_ou_from_fit",
    "trade_zeng_lee_gaussian_ou_window",
    "zeng_lee_objective",
]
