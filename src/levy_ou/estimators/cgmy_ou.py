#!/usr/bin/env python3
"""Valdivieso-style finite-variation asymmetric CGMY-OU estimator.

This provides the repository's main CGMY-OU estimator:

1. Asymmetric CGMY-OU: stationary CGMY(C, G, M, Y).

It uses the same modelling convention as the no-mean NIG-OU estimator:

    gaussian_mean = AR(1) long-run mean estimated before Levy fitting
    Y_t = X_t - gaussian_mean
    E[Y_t] = 0
    Y_i = exp(-lambda*dt) * (Y_{i-1} + Z_i^*)
    Z_i^* = exp(lambda*dt) * Y_i - Y_{i-1}

The stationary CGMY characteristic function contains a deterministic location
term chosen so that its stationary mean is exactly zero. The transition
likelihood is evaluated for Z_i^* and includes the Valdivieso Jacobian
n * lambda * dt.

The multistart structure intentionally mirrors the NIG-OU implementation:

* multi-lag ACF lambda start and lambda perturbations;
* Valdivieso innovation-cumulant starts;
* Wu-style stationary innovation-cumulant starts;
* direct stationary-sample cumulant starts;
* conservative fallback starts;
* optional ECF start;
* both cumulant-based and sample-based FFT intervals;
* optional stationary density contribution for Y_0;
* likelihood scoring followed by true multistart optimisation;
* PIT, self-PIT, ACF, scale and tail diagnostics.

The code assumes 0 < Y < 1, so the CGMY process has infinite activity but
finite variation.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from ..fourier.fft import _fft_density_from_cf_real_line

DT = 1.0 / 390.0
TAIL_PROBS = np.array([0.001, 0.005, 0.01, 0.05, 0.50, 0.95, 0.99, 0.995, 0.999])


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _finite(values: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def _empirical_cumulants(values: np.ndarray) -> dict[str, float]:
    x = _finite(values)
    if len(x) < 4:
        return {f"c{k}": np.nan for k in range(1, 5)}
    mean = float(np.mean(x))
    centered = x - mean
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    return {
        "c1": mean,
        "c2": m2,
        "c3": m3,
        "c4": m4 - 3.0 * m2**2,
    }


def acf_1d(values: np.ndarray, max_lag: int = 20) -> np.ndarray:
    x = _finite(values)
    if len(x) <= max_lag + 1:
        return np.full(max_lag, np.nan)
    x = x - float(np.mean(x))
    denom = float(x @ x)
    if denom <= 0.0:
        return np.full(max_lag, np.nan)
    return np.asarray([float(x[:-lag] @ x[lag:] / denom) for lag in range(1, max_lag + 1)])


def gaussian_ar1_start(x: np.ndarray, dt: float) -> dict[str, float]:
    """Gaussian AR(1) estimate used only for centring and the initial lambda."""
    x = _finite(x)
    if len(x) < 3:
        raise ValueError("Need at least three finite observations.")
    response = x[1:]
    lagged = x[:-1]
    intercept, rho = np.linalg.lstsq(
        np.column_stack([np.ones_like(lagged), lagged]),
        response,
        rcond=None,
    )[0]
    rho = float(np.clip(rho, 1e-6, 0.999999))
    mean = float(intercept / (1.0 - rho))
    lambda_ou = float(-np.log(rho) / dt)
    residuals = response - intercept - rho * lagged
    return {
        "mu": mean,
        "rho": rho,
        "lambda": lambda_ou,
        "resid_std": float(np.std(residuals, ddof=2)),
        "acf1": float(np.corrcoef(response, lagged)[0, 1]),
    }


def _lambda_start_multilag(x: np.ndarray, dt: float, max_lag: int = 20) -> tuple[float, float]:
    """Same positive multi-lag ACF start used by the NIG-OU estimator."""
    x = _finite(x)
    if len(x) < 10:
        return 0.1 / dt, np.nan
    acf1 = float(np.corrcoef(x[1:], x[:-1])[0, 1])
    lambda1 = float(-np.log(acf1) / dt) if np.isfinite(acf1) and 0.0 < acf1 < 1.0 else 0.1 / dt
    centered = x - float(np.mean(x))
    denom = float(centered @ centered)
    if not np.isfinite(denom) or denom <= 0.0:
        return lambda1, acf1

    max_lag = max(1, min(int(max_lag), len(x) // 4))
    lags: list[int] = []
    empirical: list[float] = []
    for lag in range(1, max_lag + 1):
        value = float(centered[:-lag] @ centered[lag:] / denom)
        if np.isfinite(value) and value > 0.0:
            lags.append(lag)
            empirical.append(value)
    if not lags:
        return lambda1, acf1

    lag_array = np.asarray(lags, dtype=float)
    empirical_array = np.asarray(empirical, dtype=float)

    def score(lambda_ou: float) -> float:
        fitted = np.exp(-float(lambda_ou) * dt * lag_array)
        return float(np.sum((empirical_array - fitted) ** 2))

    upper = max(20.0 / max(dt, 1e-12), lambda1 * 20.0)
    grid = np.geomspace(1e-8, upper, 500)
    scores = np.asarray([score(value) for value in grid])
    index = int(np.nanargmin(scores))
    lo = float(grid[max(0, index - 1)])
    hi = float(grid[min(len(grid) - 1, index + 1)])
    if lo >= hi:
        return float(grid[index]), acf1
    result = optimize.minimize_scalar(score, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return (float(result.x) if result.success else float(grid[index])), acf1


# -----------------------------------------------------------------------------
# CGMY stationary law, zero-mean location correction and Valdivieso transition
# -----------------------------------------------------------------------------


def cgmy_jump_cumulants(C: float, G: float, M: float, Y: float) -> dict[str, float]:
    """Cumulants of the zero-location finite-variation CGMY law."""
    if not (C > 0.0 and G > 0.0 and M > 0.0 and 0.0 < Y < 1.0):
        return {f"c{k}": np.nan for k in range(1, 5)}
    return {
        "c1": float(C * special.gamma(1.0 - Y) * (M ** (Y - 1.0) - G ** (Y - 1.0))),
        "c2": float(C * special.gamma(2.0 - Y) * (M ** (Y - 2.0) + G ** (Y - 2.0))),
        "c3": float(C * special.gamma(3.0 - Y) * (M ** (Y - 3.0) - G ** (Y - 3.0))),
        "c4": float(C * special.gamma(4.0 - Y) * (M ** (Y - 4.0) + G ** (Y - 4.0))),
    }


def cgmy_zero_mean_location(C: float, G: float, M: float, Y: float) -> float:
    """Location added to the stationary CGMY law so that E[Y_t] = 0."""
    return -float(cgmy_jump_cumulants(C, G, M, Y)["c1"])


def stationary_cumulants_zero_mean(C: float, G: float, M: float, Y: float) -> dict[str, float]:
    cumulants = cgmy_jump_cumulants(C, G, M, Y)
    cumulants["c1"] = 0.0
    return cumulants


def stationary_log_cf_zero_mean(
    u: np.ndarray | complex,
    C: float,
    G: float,
    M: float,
    Y: float,
) -> np.ndarray | complex:
    """Log CF of the stationary CGMY law after exact mean-zero correction."""
    u_arr = np.asarray(u, dtype=complex)
    location = cgmy_zero_mean_location(C, G, M, Y)
    jump = C * special.gamma(-Y) * (
        (M - 1j * u_arr) ** Y
        - M**Y
        + (G + 1j * u_arr) ** Y
        - G**Y
    )
    return 1j * location * u_arr + jump


def stationary_cf_zero_mean(
    u: np.ndarray | complex,
    C: float,
    G: float,
    M: float,
    Y: float,
) -> np.ndarray | complex:
    return np.exp(stationary_log_cf_zero_mean(u, C=C, G=G, M=M, Y=Y))


def zstar_cf(
    u: np.ndarray | complex,
    C: float,
    G: float,
    M: float,
    Y: float,
    lambda_ou: float,
    dt: float,
) -> np.ndarray | complex:
    """Valdivieso transition CF for Z* = exp(lambda dt)Y_i - Y_{i-1}."""
    scale = float(np.exp(lambda_ou * dt))
    return np.exp(
        stationary_log_cf_zero_mean(scale * np.asarray(u), C, G, M, Y)
        - stationary_log_cf_zero_mean(np.asarray(u), C, G, M, Y)
    )


def zstar_cumulants(
    C: float,
    G: float,
    M: float,
    Y: float,
    lambda_ou: float,
    dt: float,
) -> dict[str, float]:
    stationary = stationary_cumulants_zero_mean(C, G, M, Y)
    return {
        f"c{order}": float(np.expm1(order * lambda_ou * dt)) * stationary[f"c{order}"]
        for order in range(1, 5)
    }


def valdivieso_innovations(centered_x: np.ndarray, lambda_ou: float, dt: float) -> np.ndarray:
    x = _finite(centered_x)
    return np.exp(lambda_ou * dt) * x[1:] - x[:-1]


# -----------------------------------------------------------------------------
# FFT intervals and densities
# -----------------------------------------------------------------------------


def _cumulant_interval(cumulants: dict[str, float], truncation_l: float) -> tuple[float, float, dict[str, float]]:
    c1 = float(cumulants["c1"])
    c2 = float(cumulants["c2"])
    c4 = float(cumulants["c4"])
    width_scale = c2 + np.sqrt(max(c4, 0.0))
    width = float(truncation_l * np.sqrt(max(width_scale, 1e-16)))
    left = c1 - width
    right = c1 + width
    return left, right, {
        "interval_method": "cumulant",
        "interval_left": left,
        "interval_right": right,
        "c1": c1,
        "c2": c2,
        "c4": c4,
    }


def _sample_interval(values: np.ndarray, truncation_l: float) -> tuple[float, float, dict[str, float]]:
    x = _finite(values)
    if len(x) == 0:
        raise ValueError("Cannot construct sample FFT interval from an empty sample.")
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    iqr_scale = float(np.subtract(*np.percentile(x, [75, 25])) / 1.349) if len(x) > 1 else 0.0
    scale = max(std, iqr_scale, 1e-8)
    left = float(np.min(x) - truncation_l * scale)
    right = float(np.max(x) + truncation_l * scale)
    return left, right, {
        "interval_method": "sample",
        "interval_left": left,
        "interval_right": right,
        "sample_scale": scale,
        "sample_min": float(np.min(x)),
        "sample_max": float(np.max(x)),
    }


def fft_density(
    values: np.ndarray,
    cf_func: Callable[[np.ndarray], np.ndarray],
    cumulants: dict[str, float],
    fft_grid_size: int,
    truncation_l: float,
    density_floor: float,
    fft_interval: Literal["cumulant", "sample"],
    sample_values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if fft_interval == "cumulant":
        left, right, diagnostics = _cumulant_interval(cumulants, truncation_l)
    elif fft_interval == "sample":
        left, right, diagnostics = _sample_interval(values if sample_values is None else sample_values, truncation_l)
    else:
        raise ValueError("fft_interval must be 'cumulant' or 'sample'.")
    return _fft_density_from_cf_real_line(
        np.asarray(values, dtype=float),
        cf_func,
        left=left,
        right=right,
        fft_grid_size=int(fft_grid_size),
        density_floor=float(density_floor),
        interval_diagnostics=diagnostics,
    )


def zstar_density_fft(
    values: np.ndarray,
    C: float,
    G: float,
    M: float,
    Y: float,
    lambda_ou: float,
    dt: float,
    fft_grid_size: int,
    truncation_l: float,
    density_floor: float,
    fft_interval: Literal["cumulant", "sample"],
) -> tuple[np.ndarray, dict[str, float]]:
    return fft_density(
        values,
        lambda u: zstar_cf(u, C, G, M, Y, lambda_ou, dt),
        zstar_cumulants(C, G, M, Y, lambda_ou, dt),
        fft_grid_size,
        truncation_l,
        density_floor,
        fft_interval,
    )


def stationary_density_fft(
    values: np.ndarray,
    C: float,
    G: float,
    M: float,
    Y: float,
    fft_grid_size: int,
    truncation_l: float,
    density_floor: float,
    fft_interval: Literal["cumulant", "sample"],
    sample_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    return fft_density(
        values,
        lambda u: stationary_cf_zero_mean(u, C, G, M, Y),
        stationary_cumulants_zero_mean(C, G, M, Y),
        fft_grid_size,
        truncation_l,
        density_floor,
        fft_interval,
        sample_values=sample_values,
    )


# -----------------------------------------------------------------------------
# Parameter transformations
# -----------------------------------------------------------------------------


def symmetric_params_to_raw(C: float, eta: float, Y: float, lambda_ou: float) -> np.ndarray:
    return np.asarray([
        np.log(max(C, 1e-16)),
        np.log(max(eta, 1e-16)),
        _logit((Y - 0.02) / 0.96),
        np.log(max(lambda_ou, 1e-16)),
    ])


def symmetric_raw_to_params(raw: np.ndarray) -> tuple[float, float, float, float]:
    C = float(np.exp(raw[0]))
    eta = float(np.exp(raw[1]))
    Y = float(0.02 + 0.96 * _sigmoid(raw[2]))
    lambda_ou = float(np.exp(raw[3]))
    return C, eta, Y, lambda_ou


def asymmetric_params_to_raw(C: float, G: float, M: float, Y: float, lambda_ou: float) -> np.ndarray:
    return np.asarray([
        np.log(max(C, 1e-16)),
        np.log(max(G, 1e-16)),
        np.log(max(M, 1e-16)),
        _logit((Y - 0.02) / 0.96),
        np.log(max(lambda_ou, 1e-16)),
    ])


def asymmetric_raw_to_params(raw: np.ndarray) -> tuple[float, float, float, float, float]:
    C = float(np.exp(raw[0]))
    G = float(np.exp(raw[1]))
    M = float(np.exp(raw[2]))
    Y = float(0.02 + 0.96 * _sigmoid(raw[3]))
    lambda_ou = float(np.exp(raw[4]))
    return C, G, M, Y, lambda_ou


def _raw_in_bounds(raw: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(raw)) and np.all(raw > -40.0) and np.all(raw < 40.0))


# -----------------------------------------------------------------------------
# Moment and ECF starting mechanisms
# -----------------------------------------------------------------------------


def _relative_cumulant_residual(model: dict[str, float], target: dict[str, float], orders: tuple[int, ...]) -> np.ndarray:
    residuals = []
    for order in orders:
        key = f"c{order}"
        scale = max(abs(float(target[key])), 1e-10)
        residuals.append((float(model[key]) - float(target[key])) / scale)
    return np.asarray(residuals, dtype=float)


def _symmetric_stationary_moment_start(target: dict[str, float], method: str) -> dict[str, Any]:
    c2, c4 = float(target["c2"]), float(target["c4"])
    if not (np.isfinite(c2) and np.isfinite(c4) and c2 > 0.0 and c4 > 0.0):
        return {"valid": False, "method": method, "reason": "Need positive c2 and c4."}
    ratio = c4 / c2
    best: tuple[float, float, float, float] | None = None
    for Y in np.linspace(0.05, 0.95, 91):
        eta = float(np.sqrt(max((3.0 - Y) * (2.0 - Y) / ratio, 1e-16)))
        C = float(c2 / max(2.0 * special.gamma(2.0 - Y) * eta ** (Y - 2.0), 1e-16))
        predicted = stationary_cumulants_zero_mean(C, eta, eta, Y)
        error = float(np.sum(_relative_cumulant_residual(predicted, target, (2, 4)) ** 2))
        if best is None or error < best[0]:
            best = (error, C, eta, Y)
    if best is None:
        return {"valid": False, "method": method, "reason": "No feasible symmetric match."}
    error, C, eta, Y = best
    return {"valid": True, "method": method, "C": C, "G": eta, "M": eta, "Y": Y, "moment_objective": error}


def _asymmetric_stationary_moment_start(target: dict[str, float], method: str) -> dict[str, Any]:
    """Match stationary c2-c4. c1 is zero by construction through location."""
    if not all(np.isfinite(target[f"c{k}"]) for k in (2, 3, 4)):
        return {"valid": False, "method": method, "reason": "Non-finite target cumulants."}
    if target["c2"] <= 0.0 or target["c4"] <= 0.0:
        return {"valid": False, "method": method, "reason": "Need positive c2 and c4."}

    scale = np.sqrt(max(float(target["c2"]), 1e-16))
    starts = []
    for Y0 in (0.20, 0.40, 0.60, 0.80):
        for g_mult in (0.75, 1.5, 3.0):
            for m_mult in (0.75, 1.5, 3.0):
                G0 = g_mult / scale
                M0 = m_mult / scale
                denom = special.gamma(2.0 - Y0) * (M0 ** (Y0 - 2.0) + G0 ** (Y0 - 2.0))
                C0 = float(target["c2"] / max(denom, 1e-16))
                starts.append(np.asarray([np.log(C0), np.log(G0), np.log(M0), _logit((Y0 - 0.02) / 0.96)]))

    def unpack(raw: np.ndarray) -> tuple[float, float, float, float]:
        return float(np.exp(raw[0])), float(np.exp(raw[1])), float(np.exp(raw[2])), float(0.02 + 0.96 * _sigmoid(raw[3]))

    def residual(raw: np.ndarray) -> np.ndarray:
        if not _raw_in_bounds(raw):
            return np.full(3, 1e6)
        C, G, M, Y = unpack(raw)
        return _relative_cumulant_residual(stationary_cumulants_zero_mean(C, G, M, Y), target, (2, 3, 4))

    best: tuple[float, Any] | None = None
    for raw0 in starts:
        result = optimize.least_squares(residual, raw0, max_nfev=500)
        objective = float(np.sum(result.fun**2))
        if best is None or objective < best[0]:
            best = (objective, result)
    if best is None:
        return {"valid": False, "method": method, "reason": "No asymmetric moment optimiser completed."}
    C, G, M, Y = unpack(best[1].x)
    return {
        "valid": bool(C > 0.0 and G > 0.0 and M > 0.0 and 0.0 < Y < 1.0),
        "method": method,
        "C": C,
        "G": G,
        "M": M,
        "Y": Y,
        "moment_objective": float(best[0]),
        "reason": str(best[1].message),
    }


def _stationary_target_from_zstar(z_cumulants: dict[str, float], lambda_ou: float, dt: float) -> dict[str, float]:
    target = {"c1": 0.0}
    for order in range(2, 5):
        denominator = float(np.expm1(order * lambda_ou * dt))
        target[f"c{order}"] = float(z_cumulants[f"c{order}"]) / denominator if denominator > 0.0 else np.nan
    return target


def _ecf_start(
    centered_x: np.ndarray,
    lambda_ou: float,
    dt: float,
    symmetric: bool,
    n_u: int = 80,
    maxiter: int = 500,
) -> dict[str, Any]:
    z = valdivieso_innovations(centered_x, lambda_ou, dt)
    scale = max(float(np.std(z, ddof=1)), 1e-8)
    u = np.linspace(0.05 / scale, 4.0 / scale, int(n_u))
    empirical = np.exp(1j * np.outer(u, z)).mean(axis=1)
    weights = 1.0 / np.sqrt(1.0 + (u * scale) ** 2)
    stationary_target = _stationary_target_from_zstar(_empirical_cumulants(z), lambda_ou, dt)
    moment = _symmetric_stationary_moment_start(stationary_target, "ecf_moment_seed") if symmetric else _asymmetric_stationary_moment_start(stationary_target, "ecf_moment_seed")
    if not moment.get("valid"):
        return {"valid": False, "method": "ecf_start", "reason": "No valid ECF moment seed."}

    if symmetric:
        raw0 = symmetric_params_to_raw(moment["C"], moment["G"], moment["Y"], lambda_ou)[:3]

        def unpack(raw: np.ndarray) -> tuple[float, float, float, float]:
            C, eta, Y, _ = symmetric_raw_to_params(np.r_[raw, np.log(lambda_ou)])
            return C, eta, eta, Y
    else:
        raw0 = asymmetric_params_to_raw(moment["C"], moment["G"], moment["M"], moment["Y"], lambda_ou)[:4]

        def unpack(raw: np.ndarray) -> tuple[float, float, float, float]:
            C, G, M, Y, _ = asymmetric_raw_to_params(np.r_[raw, np.log(lambda_ou)])
            return C, G, M, Y

    def objective(raw: np.ndarray) -> float:
        if not _raw_in_bounds(raw):
            return 1e100
        C, G, M, Y = unpack(raw)
        model = zstar_cf(u, C, G, M, Y, lambda_ou, dt)
        difference = empirical - model
        return float(np.mean(weights * (difference.real**2 + difference.imag**2)))

    result = optimize.minimize(objective, raw0, method="Nelder-Mead", options={"maxiter": int(maxiter), "xatol": 1e-5, "fatol": 1e-7})
    C, G, M, Y = unpack(result.x)
    return {
        "valid": True,
        "method": "ecf_start",
        "C": C,
        "G": G,
        "M": M,
        "Y": Y,
        "lambda_ou": lambda_ou,
        "ecf_objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


# -----------------------------------------------------------------------------
# NIG-style candidate generation
# -----------------------------------------------------------------------------


def _build_candidates(
    centered_x: np.ndarray,
    dt: float,
    lambda0: float,
    lambda_factors: tuple[float, ...],
    symmetric: bool,
    include_ecf_start: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    def add(C: float, G: float, M: float, Y: float, lambda_ou: float, method: str, reason: str = "") -> None:
        if not all(np.isfinite(v) for v in (C, G, M, Y, lambda_ou)):
            return
        if not (C > 0.0 and G > 0.0 and M > 0.0 and 0.0 < Y < 1.0 and lambda_ou > 0.0 and lambda_ou * dt < 20.0):
            return
        raw = symmetric_params_to_raw(C, G, Y, lambda_ou) if symmetric else asymmetric_params_to_raw(C, G, M, Y, lambda_ou)
        if not _raw_in_bounds(raw):
            return
        key = tuple(int(round(value * 1000)) for value in raw)
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "C": float(C), "G": float(G), "M": float(M), "Y": float(Y),
            "lambda_ou": float(lambda_ou), "raw": raw, "method": method, "reason": reason,
        })

    sample_target = _empirical_cumulants(centered_x)
    sample_target["c1"] = 0.0
    sample_start = _symmetric_stationary_moment_start(sample_target, "sample_stationary_cumulant_match") if symmetric else _asymmetric_stationary_moment_start(sample_target, "sample_stationary_cumulant_match")

    for factor in lambda_factors:
        lambda_candidate = max(lambda0 * float(factor), 1e-8)
        if lambda_candidate * dt >= 20.0:
            continue
        z = valdivieso_innovations(centered_x, lambda_candidate, dt)
        z_cumulants = _empirical_cumulants(z)
        stationary_target = _stationary_target_from_zstar(z_cumulants, lambda_candidate, dt)

        # Valdivieso start: fit the actual Z* cumulants through their exact scaling.
        valdivieso_start = _symmetric_stationary_moment_start(stationary_target, f"valdivieso_moment_factor_{factor:g}") if symmetric else _asymmetric_stationary_moment_start(stationary_target, f"valdivieso_moment_factor_{factor:g}")
        if valdivieso_start.get("valid"):
            add(valdivieso_start["C"], valdivieso_start["G"], valdivieso_start["M"], valdivieso_start["Y"], lambda_candidate, valdivieso_start["method"], str(valdivieso_start.get("reason", "")))

        # Wu-style stationary innovation-cumulant start. Kept separately for provenance.
        wu_start = _symmetric_stationary_moment_start(stationary_target, f"wu_stationary_innovation_cumulant_match_factor_{factor:g}") if symmetric else _asymmetric_stationary_moment_start(stationary_target, f"wu_stationary_innovation_cumulant_match_factor_{factor:g}")
        if wu_start.get("valid"):
            add(wu_start["C"], wu_start["G"], wu_start["M"], wu_start["Y"], lambda_candidate, wu_start["method"], str(wu_start.get("reason", "")))

        if sample_start.get("valid"):
            add(sample_start["C"], sample_start["G"], sample_start["M"], sample_start["Y"], lambda_candidate, f"sample_stationary_cumulant_match_factor_{factor:g}")

        scale = max(float(np.std(z, ddof=1)), 1e-8)
        for Y0 in (0.25, 0.50, 0.75):
            if symmetric:
                for eta_mult in (1.0, 2.0, 4.0):
                    eta0 = eta_mult / scale
                    denominator = float(np.expm1(2.0 * lambda_candidate * dt)) * 2.0 * special.gamma(2.0 - Y0) * eta0 ** (Y0 - 2.0)
                    C0 = float(np.var(z, ddof=1) / max(denominator, 1e-16))
                    add(C0, eta0, eta0, Y0, lambda_candidate, f"symmetric_fallback_factor_{factor:g}_eta_{eta_mult:g}")
            else:
                for g_mult, m_mult in ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 4.0), (4.0, 2.0)):
                    G0, M0 = g_mult / scale, m_mult / scale
                    denominator = float(np.expm1(2.0 * lambda_candidate * dt)) * special.gamma(2.0 - Y0) * (M0 ** (Y0 - 2.0) + G0 ** (Y0 - 2.0))
                    C0 = float(np.var(z, ddof=1) / max(denominator, 1e-16))
                    add(C0, G0, M0, Y0, lambda_candidate, f"asymmetric_fallback_factor_{factor:g}_G_{g_mult:g}_M_{m_mult:g}")

    if include_ecf_start:
        ecf = _ecf_start(centered_x, lambda0, dt, symmetric=symmetric)
        if ecf.get("valid"):
            add(ecf["C"], ecf["G"], ecf["M"], ecf["Y"], ecf["lambda_ou"], "ecf_start")
    return candidates


def _limit_candidates(candidates: list[dict[str, Any]], max_candidates: int | None, lambda0: float) -> list[dict[str, Any]]:
    if max_candidates is None or max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates

    def source(method: str) -> str:
        for prefix in ("ecf", "wu_stationary", "valdivieso", "sample_stationary", "symmetric", "asymmetric"):
            if method.startswith(prefix):
                return prefix
        return "other"

    priority = ["ecf", "wu_stationary", "valdivieso", "sample_stationary", "asymmetric", "symmetric", "other"]
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in priority}
    for candidate in candidates:
        groups.setdefault(source(str(candidate["method"])), []).append(candidate)
    for group in groups.values():
        group.sort(key=lambda row: abs(np.log(max(row["lambda_ou"], 1e-12) / max(lambda0, 1e-12))))

    selected: list[dict[str, Any]] = []
    while len(selected) < max_candidates:
        added = False
        for key in priority:
            if groups.get(key):
                selected.append(groups[key].pop(0))
                added = True
                if len(selected) >= max_candidates:
                    break
        if not added:
            break
    return selected


# -----------------------------------------------------------------------------
# Unified Valdivieso FFT-MLE
# -----------------------------------------------------------------------------


def estimate_cgmy_ou_valdivieso_fft_multistart(
    spread: np.ndarray | pd.Series,
    variant: Literal["asymmetric"] = "asymmetric",
    dt: float = DT,
    fft_grid_size: int = 8192,
    truncation_l: float = 10.0,
    maxiter: int = 220,
    optimizer_method: str = "Powell",
    density_floor: float = 1e-300,
    lambda_factors: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
    lambda_max_lag: int = 20,
    max_optimizer_starts: int = 8,
    max_candidate_starts_to_score: int | None = 32,
    fft_intervals: tuple[str, ...] = ("cumulant", "sample"),
    include_stationary_density: bool = True,
    include_ecf_start: bool = True,
    wu_first: bool = True,
) -> dict[str, Any]:
    """Estimate the asymmetric zero-mean CGMY-OU model by Valdivieso FFT-MLE.

    The ``variant`` keyword is retained only to make accidental old calls fail
    clearly. The repo-supported CGMY estimator is asymmetric CGMY with
    independent left and right tempering parameters ``G`` and ``M``.
    """
    x = _finite(spread)
    if len(x) < 50:
        return {"valid": False, "reason": "Need at least 50 finite spread observations."}
    if variant != "asymmetric":
        raise ValueError("Only variant='asymmetric' is supported by the repo CGMY estimator.")

    started = perf_counter()
    gaussian = gaussian_ar1_start(x, dt)
    gaussian_mean = float(gaussian["mu"])
    centered_x = x - gaussian_mean
    lambda0, acf1 = _lambda_start_multilag(centered_x, dt, lambda_max_lag)
    symmetric = False

    candidates = _build_candidates(centered_x, dt, lambda0, lambda_factors, symmetric, include_ecf_start)
    generated_candidate_starts = len(candidates)
    candidates = _limit_candidates(candidates, max_candidate_starts_to_score, lambda0)
    if not candidates:
        return {"valid": False, "reason": "No feasible CGMY starts.", "variant": variant}

    def unpack(raw: np.ndarray) -> tuple[float, float, float, float, float]:
        if symmetric:
            C, eta, Y, lambda_ou = symmetric_raw_to_params(raw)
            return C, eta, eta, Y, lambda_ou
        return asymmetric_raw_to_params(raw)

    def neg_loglik(raw: np.ndarray, fft_interval: str) -> float:
        if not _raw_in_bounds(raw):
            return 1e100
        C, G, M, Y, lambda_ou = unpack(raw)
        if not (C > 0.0 and G > 0.0 and M > 0.0 and 0.0 < Y < 1.0 and lambda_ou > 0.0 and lambda_ou * dt < 20.0):
            return 1e100
        z = valdivieso_innovations(centered_x, lambda_ou, dt)
        densities, _ = zstar_density_fft(z, C, G, M, Y, lambda_ou, dt, fft_grid_size, truncation_l, density_floor, fft_interval)  # type: ignore[arg-type]
        if len(densities) != len(z) or not np.all(np.isfinite(densities)):
            return 1e100
        loglik = len(z) * lambda_ou * dt + float(np.sum(np.log(densities)))
        if include_stationary_density:
            stationary_density, _ = stationary_density_fft(
                np.asarray([centered_x[0]]), C, G, M, Y,
                fft_grid_size, truncation_l, density_floor,
                fft_interval, centered_x,  # type: ignore[arg-type]
            )
            if len(stationary_density) != 1 or not np.isfinite(stationary_density[0]):
                return 1e100
            loglik += float(np.log(stationary_density[0]))
        return -loglik if np.isfinite(loglik) else 1e100

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        for interval in fft_intervals:
            objective = float(neg_loglik(candidate["raw"], str(interval)))
            if np.isfinite(objective):
                scored.append({**candidate, "fft_interval": str(interval), "start_neg_loglik": objective})
    scored.sort(key=lambda row: row["start_neg_loglik"])
    if not scored:
        return {"valid": False, "reason": "Every candidate had non-finite likelihood.", "variant": variant}

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, int]] = set()

    def add_selected(row: dict[str, Any]) -> None:
        source = str(row["method"]).split("_factor_")[0]
        key = (source, str(row["fft_interval"]), int(round(np.log(max(row["lambda_ou"], 1e-12)) * 1000)))
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)

    if wu_first:
        for row in scored:
            if str(row["method"]).startswith(("ecf", "wu_stationary", "valdivieso")):
                add_selected(row)
                if len(selected) >= max_optimizer_starts:
                    break
    for row in scored:
        if len(selected) >= max_optimizer_starts:
            break
        add_selected(row)
    if not selected:
        selected = scored[:max_optimizer_starts]

    options: dict[str, Any] = {"maxiter": int(maxiter)}
    if optimizer_method == "Powell":
        options.update({"xtol": 1e-4, "ftol": 1e-4})
    elif optimizer_method == "Nelder-Mead":
        options.update({"xatol": 1e-4, "fatol": 1e-4})

    optimizer_rows: list[dict[str, Any]] = []
    best: tuple[float, Any, dict[str, Any]] | None = None
    for start in selected:
        interval = str(start["fft_interval"])
        result = optimize.minimize(
            lambda raw, chosen_interval=interval: neg_loglik(raw, chosen_interval),
            np.asarray(start["raw"], dtype=float),
            method=optimizer_method,
            options=options,
        )
        objective = float(result.fun) if np.isfinite(result.fun) else 1e100
        C_i, G_i, M_i, Y_i, lambda_i = unpack(result.x)
        optimizer_rows.append({
            "start_method": start["method"], "fft_interval": interval,
            "start_neg_loglik": start["start_neg_loglik"],
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
            "optimizer_neg_loglik": objective,
            "C": C_i, "G": G_i, "M": M_i, "Y": Y_i, "lambda_ou": lambda_i,
            "nfev": int(getattr(result, "nfev", -1)), "nit": int(getattr(result, "nit", -1)),
        })
        if best is None or objective < best[0]:
            best = (objective, result, start)

    if best is None:
        return {"valid": False, "reason": "No optimizer completed.", "variant": variant}

    _, result, best_start = best
    C, G, M, Y, lambda_ou = unpack(result.x)
    z = valdivieso_innovations(centered_x, lambda_ou, dt)
    best_interval = str(best_start["fft_interval"])
    densities, fft_diag = zstar_density_fft(z, C, G, M, Y, lambda_ou, dt, fft_grid_size, truncation_l, density_floor, best_interval)  # type: ignore[arg-type]
    loglik = len(z) * lambda_ou * dt + float(np.sum(np.log(densities)))
    stationary_diag: dict[str, Any] = {}
    if include_stationary_density:
        stationary_density, stationary_diag = stationary_density_fft(
            np.asarray([centered_x[0]]), C, G, M, Y,
            fft_grid_size, truncation_l, density_floor,
            best_interval, centered_x,  # type: ignore[arg-type]
        )
        loglik += float(np.log(stationary_density[0]))

    table = pd.DataFrame(optimizer_rows).sort_values("optimizer_neg_loglik")
    best_row = table.iloc[0].to_dict() if not table.empty else {}
    location = cgmy_zero_mean_location(C, G, M, Y)
    valid = bool(np.isfinite(loglik) and C > 0.0 and G > 0.0 and M > 0.0 and 0.0 < Y < 1.0 and lambda_ou > 0.0)
    return {
        "valid": valid,
        "reason": None if valid else str(result.message),
        "estimation_method": f"{variant}_cgmy_ou_valdivieso_transition_mle_fft_multistart",
        "variant": variant,
        "gaussian_mean": gaussian_mean,
        "mu": gaussian_mean,
        "u_form": gaussian_mean,
        "mu_form": gaussian_mean,
        "process_mean": gaussian_mean,
        "centered_stationary_mean": 0.0,
        "cgmy_location": location,
        "C": float(C), "G": float(G), "M": float(M), "eta": float(G) if symmetric else np.nan,
        "Y": float(Y), "lambda_ou": float(lambda_ou), "lambda": float(lambda_ou),
        "rho": float(np.exp(-lambda_ou * dt)),
        "loglik": float(loglik), "neg_loglik": float(-loglik),
        "dt": float(dt), "observations": int(len(x)), "increments": int(len(z)),
        "acf1": float(acf1) if np.isfinite(acf1) else np.nan,
        "initial_lambda_ou": float(lambda0),
        "initial_gaussian_mean": gaussian_mean,
        "initial_C": float(best_start["C"]), "initial_G": float(best_start["G"]),
        "initial_M": float(best_start["M"]), "initial_Y": float(best_start["Y"]),
        "best_start_method": str(best_start["method"]), "best_fft_interval": best_interval,
        "best_start_neg_loglik": float(best_start["start_neg_loglik"]),
        "best_optimizer_start_method": str(best_row.get("start_method", best_start["method"])),
        "best_optimizer_fft_interval": str(best_row.get("fft_interval", best_interval)),
        "generated_candidate_starts": int(generated_candidate_starts),
        "candidate_starts": int(len(candidates)), "scored_starts": int(len(scored)),
        "optimizer_starts": int(len(selected)), "optimizer_method": optimizer_method,
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "optimizer_iterations": int(getattr(result, "nit", -1)),
        "optimizer_evaluations": int(getattr(result, "nfev", -1)),
        "fit_seconds": float(perf_counter() - started),
        "fft_grid_size": int(fft_grid_size), "truncation_l": float(truncation_l),
        "fft_intervals": ",".join(fft_intervals), "include_stationary_density": bool(include_stationary_density),
        "include_ecf_start": bool(include_ecf_start),
        "innovations": pd.Series(z, name="cgmy_ou_valdivieso_zstar"),
        "start_table": table,
        **{f"fft_{key}": value for key, value in fft_diag.items()},
        **{f"stationary_fft_{key}": value for key, value in stationary_diag.items()},
    }


def estimate_asymmetric_cgmy_ou_valdivieso_fft_multistart(spread: np.ndarray | pd.Series, **kwargs: Any) -> dict[str, Any]:
    """Estimate fully asymmetric CGMY-OU with independent ``G`` and ``M``."""
    return estimate_cgmy_ou_valdivieso_fft_multistart(spread, variant="asymmetric", **kwargs)


def estimate_cgmy_ou_fft_mle(spread: np.ndarray | pd.Series, **kwargs: Any) -> dict[str, Any]:
    """Main repo CGMY estimator: asymmetric Valdivieso transition FFT-MLE."""
    return estimate_asymmetric_cgmy_ou_valdivieso_fft_multistart(spread, **kwargs)


# -----------------------------------------------------------------------------
# Diagnostics retained from the earlier CGMY implementation
# -----------------------------------------------------------------------------


def _density_grid_for_fit(
    fit: dict[str, Any],
    dt: float,
    n_grid: int = 32768,
    truncation_l: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    C, G, M, Y = map(float, (fit["C"], fit["G"], fit["M"], fit["Y"]))
    lambda_ou = float(fit["lambda_ou"])
    cumulants = zstar_cumulants(C, G, M, Y, lambda_ou, dt)
    left, right, interval_diag = _cumulant_interval(cumulants, truncation_l)
    x_grid = np.linspace(left, right, int(n_grid))
    density, fft_diag = fft_density(
        x_grid,
        lambda u: zstar_cf(u, C, G, M, Y, lambda_ou, dt),
        cumulants,
        int(n_grid), truncation_l, 0.0, "cumulant",
    )
    density = np.maximum(np.asarray(density, dtype=float), 0.0)
    dx = float(x_grid[1] - x_grid[0])
    mass = float(np.trapz(density, x_grid))
    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("Diagnostic density has non-positive total mass.")
    density /= mass
    cdf = np.r_[0.0, np.cumsum((density[:-1] + density[1:]) * 0.5 * dx)]
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    cdf[-1] = 1.0
    keep = np.r_[True, np.diff(cdf) > 1e-14]
    return x_grid[keep], density[keep], cdf[keep], {**interval_diag, **fft_diag, "renormalized_mass": mass}


def diagnose_cgmy_ou_fit(
    spread: np.ndarray | pd.Series,
    fit: dict[str, Any],
    dt: float = DT,
    n_selftest: int = 100_000,
    seed: int = 123,
) -> dict[str, Any]:
    if not fit.get("valid", False):
        return {"diagnostic_success": False, "diagnostic_reason": str(fit.get("reason", "invalid fit"))}
    x = _finite(spread)
    centered = x - float(fit["gaussian_mean"])
    z_emp = valdivieso_innovations(centered, float(fit["lambda_ou"]), dt)
    try:
        x_grid, density_grid, cdf_grid, fft_diag = _density_grid_for_fit(fit, dt)
        rng = np.random.default_rng(seed)
        z_sim = np.interp(rng.random(int(n_selftest)), cdf_grid, x_grid)
        pit_emp = np.clip(np.interp(z_emp, x_grid, cdf_grid, left=0.0, right=1.0), 1e-12, 1.0 - 1e-12)
        pit_self = np.clip(np.interp(z_sim, x_grid, cdf_grid, left=0.0, right=1.0), 1e-12, 1.0 - 1e-12)
        ks_emp = stats.kstest(pit_emp, "uniform")
        ks_self = stats.kstest(pit_self, "uniform")
        rho = float(np.exp(-fit["lambda_ou"] * dt))
        acf_emp = acf_1d(centered, 20)
        acf_theory = rho ** np.arange(1, 21)
        q_emp = np.quantile(z_emp, TAIL_PROBS)
        q_sim = np.quantile(z_sim, TAIL_PROBS)
        emp_std = float(np.std(z_emp, ddof=1))
        sim_std = float(np.std(z_sim, ddof=1))
        emp_width = float(q_emp[-1] - q_emp[0])
        sim_width = float(q_sim[-1] - q_sim[0])
        return {
            "diagnostic_success": True,
            "diagnostic_reason": "",
            "formation_residual_ks_stat": float(ks_emp.statistic),
            "formation_residual_ks_p": float(ks_emp.pvalue),
            "simulator_self_pit_ks_stat": float(ks_self.statistic),
            "simulator_self_pit_ks_p": float(ks_self.pvalue),
            "acf_mae_lag20": float(np.nanmean(np.abs(acf_emp - acf_theory))),
            "transition_residual_mean": float(np.mean(z_emp)),
            "transition_residual_std": emp_std,
            "fitted_transition_mean": float(np.mean(z_sim)),
            "fitted_transition_std": sim_std,
            "std_ratio_sim_over_emp": float(sim_std / emp_std) if emp_std > 0.0 else np.nan,
            "tail_width_ratio_sim_over_emp": float(sim_width / emp_width) if emp_width > 0.0 else np.nan,
            "x_min_used": float(x_grid[0]), "x_max_used": float(x_grid[-1]),
            "density_grid_points": int(len(x_grid)),
            **{f"diagnostic_fft_{key}": value for key, value in fft_diag.items()},
        }
    except Exception as exc:
        return {"diagnostic_success": False, "diagnostic_reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "estimate_cgmy_ou_fft_mle",
    "estimate_asymmetric_cgmy_ou_valdivieso_fft_multistart",
    "estimate_cgmy_ou_valdivieso_fft_multistart",
    "diagnose_cgmy_ou_fit",
    "gaussian_ar1_start",
    "valdivieso_innovations",
    "cgmy_zero_mean_location",
    "stationary_cf_zero_mean",
    "zstar_cf",
]
