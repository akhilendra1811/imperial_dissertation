"""Double-exponential compound-Poisson OU estimators."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def endres_stubinger_mu_t(spread: pd.Series) -> pd.Series:
    """
    Build the Endres-Stuebinger/Liu-style time-varying mean level.

    The mean level is a step function. At each minute, it is the average of the
    last two available daily opening/closing spread values. For example, after
    today's open and before today's close, this is usually:

        mu_t = (yesterday_close_spread + today_open_spread) / 2

    The function requires a timestamped spread series so it can identify daily
    opening and closing observations.
    """
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series with a timestamp index.")

    x = pd.to_numeric(spread, errors="coerce").dropna().sort_index()
    if x.empty:
        return pd.Series(dtype=float, name="mu_t")

    timestamps = pd.to_datetime(x.index, errors="coerce")
    if pd.isna(timestamps).any():
        raise ValueError("spread index must be convertible to timestamps.")

    frame = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps),
            "spread": x.to_numpy(dtype=float),
        },
        index=x.index,
    )
    frame["trade_date"] = frame["timestamp"].dt.date

    daily_open = frame.groupby("trade_date", sort=True).first()[["timestamp", "spread"]]
    daily_close = frame.groupby("trade_date", sort=True).last()[["timestamp", "spread"]]
    daily_open["event_order"] = 0
    daily_close["event_order"] = 1

    events = (
        pd.concat([daily_open, daily_close], ignore_index=True)
        .sort_values(["timestamp", "event_order"])
        .reset_index(drop=True)
    )
    event_times = events["timestamp"].astype("int64").to_numpy()
    event_values = events["spread"].to_numpy(dtype=float)
    observation_times = frame["timestamp"].astype("int64").to_numpy()

    event_counts = np.searchsorted(event_times, observation_times, side="right")
    mu_values = np.empty(len(frame), dtype=float)
    for i, count in enumerate(event_counts):
        if count >= 2:
            mu_values[i] = 0.5 * (event_values[count - 1] + event_values[count - 2])
        elif count == 1:
            # At the very start of the first available day, only the opening
            # spread is known. Use it until a second open/close value exists.
            mu_values[i] = event_values[0]
        else:
            mu_values[i] = np.nan

    return pd.Series(mu_values, index=x.index, name="mu_t")


def estimate_double_exp_cp_ou(
    spread: pd.Series,
    beta: float = 0.4999,
    delta_mode: str = "nu",
    trading_days_per_year: int = 250,
    minutes_per_day: int = 391,
) -> dict[str, Any]:
    """
    Estimate the double-exponential compound Poisson jump-diffusion OU model, with moving mean.

    The model is:

        dX_t = theta * (mu_t - X_t) dt + sigma dW_t + dJ_t

    where jumps are detected by the threshold nu_n = dt ** beta. The mean level
    mu_t is the Endres-Stuebinger-style step function based on the last two
    available daily opening/closing spread values.

    Parameters
    ----------
    spread:
        Minute-level spread series with a timestamp index.
    beta:
        Jump-threshold exponent. The paper uses 0.4999.
    delta_mode:
        "nu" uses delta = nu_n for the shifted double-exponential jump-size
        distribution. "zero" uses delta = 0.
    trading_days_per_year, minutes_per_day:
        Used to define dt. The paper's minute-data convention is
        dt = 1 / (250 * 391). 
    """
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series with a timestamp index.")
    if not (0 < beta < 0.5):
        raise ValueError("beta must be in (0, 0.5).")

    x = pd.to_numeric(spread, errors="coerce").dropna().sort_index()
    if len(x) < 3:
        return {
            "valid": False,
            "reason": "Need at least three finite spread observations.",
        }

    started = perf_counter()
    dt = 1.0 / float(trading_days_per_year * minutes_per_day)
    nu_n = float(dt**beta)

    mu_t = endres_stubinger_mu_t(x).reindex(x.index)
    x_values = x.to_numpy(dtype=float)
    mu_values = mu_t.to_numpy(dtype=float)

    # Increments are from t_i to t_{i+1}. The drift term uses X_i and mu_i.
    dx = np.diff(x_values)
    x_lag = x_values[:-1]
    mu_lag = mu_values[:-1]
    increment_index = x.index[1:]

    finite = np.isfinite(dx) & np.isfinite(x_lag) & np.isfinite(mu_lag)
    dx = dx[finite]
    x_lag = x_lag[finite]
    mu_lag = mu_lag[finite]
    increment_index = increment_index[finite]

    if len(dx) < 2:
        return {
            "valid": False,
            "reason": "Need at least two finite increments after aligning mu_t.",
            "mu_t": mu_t,
            "dt": dt,
            "nu_n": nu_n,
        }

    jump_mask = np.abs(dx) > nu_n
    continuous_mask = ~jump_mask
    jump_increments = pd.Series(dx[jump_mask], index=increment_index[jump_mask], name="jump_increment")
    continuous_increments = pd.Series(
        dx[continuous_mask],
        index=increment_index[continuous_mask],
        name="continuous_increment",
    )

    # Paper-style theta estimator: remove jump increments from the numerator,
    # while the denominator is the integrated squared distance from mu_t.
    distance_to_mean = mu_lag - x_lag
    denominator = float(np.sum(distance_to_mean**2) * dt)
    numerator = float(np.sum(distance_to_mean[continuous_mask] * dx[continuous_mask]))
    if denominator <= 0 or not np.isfinite(denominator):
        return {
            "valid": False,
            "reason": "Cannot estimate theta because the denominator is zero or non-finite.",
            "mu_t": mu_t,
            "dt": dt,
            "nu_n": nu_n,
            "jump_increments": jump_increments,
            "continuous_increments": continuous_increments,
        }

    theta = numerator / denominator

    # Estimate Brownian volatility from the non-jump residuals.
    continuous_residuals = (
        dx[continuous_mask]
        - theta * distance_to_mean[continuous_mask] * dt
    )
    if len(continuous_residuals) >= 2:
        sigma = float(np.std(continuous_residuals, ddof=1) / np.sqrt(dt))
    else:
        sigma = np.nan

    num_jumps = int(jump_mask.sum())
    total_time = float(len(dx) * dt)
    lambda_jump = float(num_jumps / total_time) if total_time > 0 else np.nan

    if delta_mode == "nu":
        delta = nu_n
    elif delta_mode == "zero":
        delta = 0.0
    else:
        raise ValueError("delta_mode must be 'nu' or 'zero'.")

    if num_jumps > 0:
        jump_excess = np.abs(jump_increments.to_numpy(dtype=float)) - delta
        jump_excess = jump_excess[jump_excess > 0]
        eta = float(len(jump_excess) / np.sum(jump_excess)) if len(jump_excess) else np.nan
    else:
        eta = np.nan

    mean_reverting = bool(np.isfinite(theta) and theta > 0)
    valid = bool(mean_reverting and np.isfinite(sigma))
    reason = None
    if not mean_reverting:
        reason = "Estimated theta is not positive, so the spread is not mean-reverting to mu_t."
    elif not np.isfinite(sigma):
        reason = "Estimated sigma is not finite."

    elapsed = perf_counter() - started
    return {
        "valid": valid,
        "reason": reason,
        "mean_reverting": mean_reverting,
        "estimation_method": "double_exponential_compound_poisson_ou_threshold",
        "mu": float(np.nanmean(mu_lag)),
        "mu_t": mu_t,
        "theta": float(theta),
        "kappa": float(theta),
        "sigma": sigma,
        "lambda_jump": lambda_jump,
        "eta": eta,
        "delta": float(delta),
        "delta_mode": delta_mode,
        "nu_n": nu_n,
        "beta": float(beta),
        "dt": dt,
        "trading_days_per_year": int(trading_days_per_year),
        "minutes_per_day": int(minutes_per_day),
        "observations": int(len(x)),
        "increments": int(len(dx)),
        "num_jumps": num_jumps,
        "num_continuous": int(continuous_mask.sum()),
        "jump_fraction": float(num_jumps / len(dx)),
        "jump_increments": jump_increments,
        "continuous_increments": continuous_increments,
        "fit_seconds": float(elapsed),
    }

def estimate_compound_poisson_ou(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for the double-exponential compound Poisson OU estimator."""
    return estimate_double_exp_cp_ou(*args, **kwargs)


def estimate_cp_ou_fixed_mean(
    spread: pd.Series | np.ndarray,
    u_form: float,
    beta: float = 0.4999,
    delta_mode: str = "nu",
    trading_days_per_year: int = 250,
    minutes_per_day: int = 390,
) -> dict[str, Any]:
    """Estimate CP-OU with the OU mean fixed to supplied formation mean u_form.

    This is the dissertation-facing corrected CP-OU API. It uses the same jump
    threshold and diffusion/jump estimators as the legacy CP-OU implementation,
    but replaces the Endres/Stuebinger moving `mu_t` with a constant
    Gaussian-formation mean.
    """

    if not (0 < beta < 0.5):
        raise ValueError("beta must be in (0, 0.5).")
    if not np.isfinite(float(u_form)):
        return {"valid": False, "reason": "u_form must be finite.", "u_form": float(u_form)}

    if isinstance(spread, pd.Series):
        x = pd.to_numeric(spread, errors="coerce").dropna().sort_index()
    else:
        values = np.asarray(spread, dtype=float)
        values = values[np.isfinite(values)]
        x = pd.Series(values)
    if len(x) < 3:
        return {"valid": False, "reason": "Need at least three finite spread observations.", "u_form": float(u_form)}

    started = perf_counter()
    dt = 1.0 / float(trading_days_per_year * minutes_per_day)
    nu_n = float(dt**beta)
    x_values = x.to_numpy(dtype=float)

    dx = np.diff(x_values)
    x_lag = x_values[:-1]
    mu_lag = np.full_like(x_lag, float(u_form), dtype=float)
    increment_index = x.index[1:]

    finite = np.isfinite(dx) & np.isfinite(x_lag)
    dx = dx[finite]
    x_lag = x_lag[finite]
    mu_lag = mu_lag[finite]
    increment_index = increment_index[finite]
    if len(dx) < 2:
        return {
            "valid": False,
            "reason": "Need at least two finite increments.",
            "u_form": float(u_form),
            "dt": dt,
            "nu_n": nu_n,
        }

    jump_mask = np.abs(dx) > nu_n
    continuous_mask = ~jump_mask
    jump_increments = pd.Series(dx[jump_mask], index=increment_index[jump_mask], name="jump_increment")
    continuous_increments = pd.Series(
        dx[continuous_mask],
        index=increment_index[continuous_mask],
        name="continuous_increment",
    )

    distance_to_mean = mu_lag - x_lag
    denominator = float(np.sum(distance_to_mean**2) * dt)
    numerator = float(np.sum(distance_to_mean[continuous_mask] * dx[continuous_mask]))
    if denominator <= 0 or not np.isfinite(denominator):
        return {
            "valid": False,
            "reason": "Cannot estimate theta because the fixed-mean denominator is zero or non-finite.",
            "u_form": float(u_form),
            "dt": dt,
            "nu_n": nu_n,
            "jump_increments": jump_increments,
            "continuous_increments": continuous_increments,
        }

    theta = numerator / denominator
    continuous_residuals = dx[continuous_mask] - theta * distance_to_mean[continuous_mask] * dt
    sigma = float(np.std(continuous_residuals, ddof=1) / np.sqrt(dt)) if len(continuous_residuals) >= 2 else np.nan

    num_jumps = int(jump_mask.sum())
    total_time = float(len(dx) * dt)
    lambda_jump = float(num_jumps / total_time) if total_time > 0 else np.nan

    if delta_mode == "nu":
        delta = nu_n
    elif delta_mode == "zero":
        delta = 0.0
    else:
        raise ValueError("delta_mode must be 'nu' or 'zero'.")

    if num_jumps > 0:
        jump_excess = np.abs(jump_increments.to_numpy(dtype=float)) - delta
        jump_excess = jump_excess[jump_excess > 0]
        eta = float(len(jump_excess) / np.sum(jump_excess)) if len(jump_excess) else np.nan
    else:
        eta = np.nan

    mean_reverting = bool(np.isfinite(theta) and theta > 0)
    valid = bool(mean_reverting and np.isfinite(sigma))
    reason = None
    if not mean_reverting:
        reason = "Estimated theta is not positive, so the spread is not mean-reverting to u_form."
    elif not np.isfinite(sigma):
        reason = "Estimated sigma is not finite."

    elapsed = perf_counter() - started
    return {
        "valid": valid,
        "reason": reason,
        "mean_reverting": mean_reverting,
        "estimation_method": "double_exponential_cp_ou_fixed_mean_threshold",
        "mean_mode": "fixed_u_form",
        "u_form": float(u_form),
        "mu": float(u_form),
        "theta": float(theta),
        "kappa": float(theta),
        "sigma": sigma,
        "lambda_jump": lambda_jump,
        "eta": eta,
        "delta": float(delta),
        "delta_mode": delta_mode,
        "nu_n": nu_n,
        "beta": float(beta),
        "dt": dt,
        "trading_days_per_year": int(trading_days_per_year),
        "minutes_per_day": int(minutes_per_day),
        "observations": int(len(x)),
        "increments": int(len(dx)),
        "num_jumps": num_jumps,
        "num_continuous": int(continuous_mask.sum()),
        "jump_fraction": float(num_jumps / len(dx)),
        "jump_increments": jump_increments,
        "continuous_increments": continuous_increments,
        "fit_seconds": float(elapsed),
    }

__all__ = [
    "endres_stubinger_mu_t",
    "estimate_cp_ou_fixed_mean",
    "estimate_compound_poisson_ou",
    "estimate_double_exp_cp_ou",
]
