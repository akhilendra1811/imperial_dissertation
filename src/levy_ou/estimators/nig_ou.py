"""Maintained NIG-OU estimators.

The supported project API is deliberately narrow:

* ``ar1_formation_mean``
* ``estimate_centered_nig_residual_fft_multistart``
* ``estimate_nig_ou_fixed_mean_fft_multistart``

The estimator fixes the OU mean to the Gaussian/AR(1) formation
mean, centres the residual process, and fits the centred residuals by the
multistart FFT transition likelihood.  Some older COS and uncentred FFT helpers
remain in this module because ``nig_trials.py`` imports them for legacy
comparison estimators; they are not part of the public API exported by
``__all__``.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..config import DT_MINUTE
from ..fourier.fft import fft_density_from_cf_real_line as _fft_density_from_cf_real_line


def nig_ou_zstar_cf(
    u: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
) -> np.ndarray:
    """
    Characteristic function of the NIG-OU transition innovation Z*(dt).

    The transition innovation is:
        Y_k = exp(lambda_ou * dt) X_k - X_{k-1}
    and the paper's characteristic function is:
    
        exp(delta * [sqrt(alpha^2 - (beta + iu)^2)
            - sqrt(alpha^2 - (beta + iu exp(lambda dt))^2)])
    """
    u = np.asarray(u, dtype=float)
    e = np.exp(lambda_ou * dt)
    term1 = np.sqrt(alpha**2 - (beta + 1j * u) ** 2)
    term2 = np.sqrt(alpha**2 - (beta + 1j * u * e) ** 2)
    return np.exp(delta_nig * (term1 - term2))


def nig_ou_zstar_density_cos(
    y_values: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
    cos_terms: int = 256,
    truncation_l: float = 10.0,
    density_floor: float = 1e-300,
) -> np.ndarray:
    """
    Approximate the Z*(dt) density using the Fourier-COS expansion.

    The density on [a, b] is approximated by:

        f(y) = 2 / (b - a) * sum'_k Re(phi(u_k) exp(-iu_k a))
               * cos(u_k (y - a))

    where u_k = k*pi/(b-a), and the k=0 term has half weight.

    The truncation interval is chosen from the current innovation sample with a
    generous padding. This is a practical first implementation; for production
    speed/accuracy, the interval should be chosen from model cumulants.
    """
    y = np.asarray(y_values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.array([], dtype=float)

    scale = max(
        float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        float(np.subtract(*np.percentile(y, [75, 25])) / 1.349) if len(y) > 1 else 0.0,
        1e-8,
    )
    a = float(np.min(y) - truncation_l * scale)
    b = float(np.max(y) + truncation_l * scale)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return np.full(len(y), density_floor, dtype=float)

    k = np.arange(cos_terms, dtype=float)
    u = k * np.pi / (b - a)
    phi = nig_ou_zstar_cf(
        u,
        alpha=alpha,
        beta=beta,
        delta_nig=delta_nig,
        lambda_ou=lambda_ou,
        dt=dt,
    )
    coefficients = np.real(phi * np.exp(-1j * u * a))
    coefficients[0] *= 0.5

    cos_matrix = np.cos(np.outer(y - a, u))
    density = (2.0 / (b - a)) * cos_matrix.dot(coefficients)
    return np.maximum(density, density_floor)


def nig_stationary_cumulants(
    alpha: float,
    beta: float,
    delta_nig: float,
) -> dict[str, float]:
    """Cumulants 1-4 of a zero-location stationary NIG(alpha, beta, delta)."""
    gamma_sq = float(alpha) ** 2 - float(beta) ** 2
    if not np.isfinite(gamma_sq) or gamma_sq <= 0 or delta_nig <= 0:
        return {"c1": np.nan, "c2": np.nan, "c3": np.nan, "c4": np.nan}

    gamma = float(np.sqrt(gamma_sq))
    alpha_sq = float(alpha) ** 2
    beta_sq = float(beta) ** 2
    delta_nig = float(delta_nig)
    return {
        "c1": float(delta_nig * float(beta) / gamma),
        "c2": float(delta_nig * alpha_sq / gamma**3),
        "c3": float(3.0 * delta_nig * float(beta) * alpha_sq / gamma**5),
        "c4": float(
            3.0 * delta_nig * alpha_sq * (alpha_sq + 4.0 * beta_sq) / gamma**7
        ),
    }


def nig_ou_zstar_cumulants(
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
) -> dict[str, float]:
    """
    Cumulants 1-4 of Y = exp(lambda*dt) X_t - X_{t-dt}.

    For stationary NIG-OU, the transition innovation has cumulants:

        kappa_r(Y) = (exp(r * lambda * dt) - 1) * kappa_r(X)

    where X has stationary NIG(alpha, beta, delta) distribution.
    """
    stationary = nig_stationary_cumulants(alpha, beta, delta_nig)
    return {
        f"c{order}": float(np.expm1(order * float(lambda_ou) * float(dt)))
        * stationary[f"c{order}"]
        for order in range(1, 5)
    }


def nig_ou_cos_cumulant_interval(
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
    truncation_l: float = 10.0,
) -> tuple[float, float, dict[str, float]]:
    """
    COS truncation interval based on transition-innovation cumulants.

    Fang-Oosterlee style:

        [a,b] = [c1 - L * sqrt(c2 + sqrt(c4)),
                 c1 + L * sqrt(c2 + sqrt(c4))]
    """
    cumulants = nig_ou_zstar_cumulants(
        alpha=alpha,
        beta=beta,
        delta_nig=delta_nig,
        lambda_ou=lambda_ou,
        dt=dt,
    )
    c1 = cumulants["c1"]
    c2 = cumulants["c2"]
    c4 = cumulants["c4"]
    width_scale = c2 + np.sqrt(max(c4, 0.0))
    if not (
        np.isfinite(c1)
        and np.isfinite(c2)
        and np.isfinite(c4)
        and np.isfinite(width_scale)
        and width_scale > 0
    ):
        diagnostics = {
            **cumulants,
            "truncation_a": np.nan,
            "truncation_b": np.nan,
            "truncation_width": np.nan,
            "truncation_l": float(truncation_l),
        }
        return np.nan, np.nan, diagnostics

    half_width = float(truncation_l) * float(np.sqrt(width_scale))
    a = float(c1 - half_width)
    b = float(c1 + half_width)
    diagnostics = {
        **cumulants,
        "truncation_a": a,
        "truncation_b": b,
        "truncation_width": b - a,
        "truncation_l": float(truncation_l),
    }
    return a, b, diagnostics




def nig_stationary_cf(
    u: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
) -> np.ndarray:
    """Characteristic function of zero-location stationary NIG(alpha,beta,delta)."""
    u = np.asarray(u, dtype=float)
    gamma = np.sqrt(float(alpha) ** 2 - float(beta) ** 2)
    term = np.sqrt(float(alpha) ** 2 - (float(beta) + 1j * u) ** 2)
    return np.exp(float(delta_nig) * (gamma - term))


def _nig_sample_fft_interval(
    values: np.ndarray,
    truncation_l: float,
) -> tuple[float, float, dict[str, float]]:
    """Sample-based real-line FFT interval for concentrated NIG innovations."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    diagnostics = {
        "truncation_a": np.nan,
        "truncation_b": np.nan,
        "truncation_width": np.nan,
        "truncation_l": float(truncation_l),
        "interval_source": "sample",
    }
    if len(finite) <= 1:
        return np.nan, np.nan, diagnostics

    q75, q25 = np.percentile(finite, [75, 25])
    scale = max(
        float(np.std(finite, ddof=1)),
        float((q75 - q25) / 1.349),
        1e-8,
    )
    a = float(np.min(finite) - float(truncation_l) * scale)
    b = float(np.max(finite) + float(truncation_l) * scale)
    diagnostics.update(
        {
            "truncation_a": a,
            "truncation_b": b,
            "truncation_width": b - a,
            "sample_scale": scale,
        }
    )
    return a, b, diagnostics


def _nig_cumulant_interval_from_cumulants(
    cumulants: dict[str, float],
    truncation_l: float,
) -> tuple[float, float, dict[str, float]]:
    """Fang-Oosterlee-style real-line interval from cumulants."""
    c1 = float(cumulants.get("c1", np.nan))
    c2 = float(cumulants.get("c2", np.nan))
    c4 = float(cumulants.get("c4", np.nan))
    diagnostics = {
        **cumulants,
        "truncation_a": np.nan,
        "truncation_b": np.nan,
        "truncation_width": np.nan,
        "truncation_l": float(truncation_l),
        "interval_source": "cumulant",
    }
    width_scale = c2 + np.sqrt(max(c4, 0.0))
    if not (
        np.isfinite(c1)
        and np.isfinite(c2)
        and np.isfinite(c4)
        and np.isfinite(width_scale)
        and width_scale > 0
    ):
        return np.nan, np.nan, diagnostics
    half_width = float(truncation_l) * float(np.sqrt(width_scale))
    a = float(c1 - half_width)
    b = float(c1 + half_width)
    diagnostics.update(
        {
            "truncation_a": a,
            "truncation_b": b,
            "truncation_width": b - a,
        }
    )
    return a, b, diagnostics


def nig_ou_zstar_density_fft_multistart(
    y_values: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
    fft_grid_size: int,
    truncation_l: float,
    density_floor: float,
    fft_interval: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Transition innovation density with selectable FFT interval scheme."""
    y = np.asarray(y_values, dtype=float)
    y = y[np.isfinite(y)]
    if fft_interval == "cumulant":
        left, right, interval_diagnostics = nig_ou_cos_cumulant_interval(
            alpha=alpha,
            beta=beta,
            delta_nig=delta_nig,
            lambda_ou=lambda_ou,
            dt=dt,
            truncation_l=truncation_l,
        )
        interval_diagnostics["interval_source"] = "cumulant"
    elif fft_interval == "sample":
        left, right, interval_diagnostics = _nig_sample_fft_interval(
            y,
            truncation_l=truncation_l,
        )
    else:
        raise ValueError(f"Unknown NIG FFT interval: {fft_interval}")

    return _fft_density_from_cf_real_line(
        y,
        lambda u: nig_ou_zstar_cf(
            u,
            alpha=alpha,
            beta=beta,
            delta_nig=delta_nig,
            lambda_ou=lambda_ou,
            dt=dt,
        ),
        left=left,
        right=right,
        fft_grid_size=fft_grid_size,
        density_floor=density_floor,
        interval_diagnostics=interval_diagnostics,
    )


def nig_stationary_density_fft_multistart(
    x_values: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
    fft_grid_size: int,
    truncation_l: float,
    density_floor: float,
    fft_interval: str,
    sample_values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Stationary NIG density with selectable FFT interval scheme."""
    x = np.asarray(x_values, dtype=float)
    x = x[np.isfinite(x)]
    if fft_interval == "cumulant":
        left, right, interval_diagnostics = _nig_cumulant_interval_from_cumulants(
            nig_stationary_cumulants(alpha, beta, delta_nig),
            truncation_l=truncation_l,
        )
    elif fft_interval == "sample":
        interval_sample = x if sample_values is None else np.asarray(sample_values, dtype=float)
        left, right, interval_diagnostics = _nig_sample_fft_interval(
            interval_sample,
            truncation_l=truncation_l,
        )
    else:
        raise ValueError(f"Unknown NIG FFT interval: {fft_interval}")

    return _fft_density_from_cf_real_line(
        x,
        lambda u: nig_stationary_cf(
            u,
            alpha=alpha,
            beta=beta,
            delta_nig=delta_nig,
        ),
        left=left,
        right=right,
        fft_grid_size=fft_grid_size,
        density_floor=density_floor,
        interval_diagnostics=interval_diagnostics,
    )


def _nig_ou_lambda_start(x: np.ndarray, dt: float) -> tuple[float, float]:
    """Initial OU speed from lag-1 autocorrelation."""
    acf1 = float(np.corrcoef(x[1:], x[:-1])[0, 1])
    if not np.isfinite(acf1) or acf1 <= 0 or acf1 >= 1:
        # Fallback to a slow but positive mean-reversion speed.
        return 0.1 / dt, acf1
    return float(-np.log(acf1) / dt), acf1


def _nig_ou_lambda_start_multilag(
    x: np.ndarray,
    dt: float,
    max_lag: int = 20,
) -> tuple[float, float]:
    """
    Robust OU speed start by matching several empirical ACF lags.

    This mirrors the TS estimator idea: rather than trusting only lag 1, choose
    lambda to minimize the distance between empirical acf(k) and
    exp(-lambda * k * dt) over the first few lags.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lambda1, acf1 = _nig_ou_lambda_start(x, dt=dt)
    if len(x) < 10:
        return lambda1, acf1

    max_lag = max(1, min(int(max_lag), len(x) // 4))
    centered = x - float(np.mean(x))
    denom = float(centered @ centered)
    if not np.isfinite(denom) or denom <= 0:
        return lambda1, acf1

    lags: list[int] = []
    empirical: list[float] = []
    for lag in range(1, max_lag + 1):
        rho = float(centered[:-lag] @ centered[lag:] / denom)
        if np.isfinite(rho) and rho > 0:
            lags.append(lag)
            empirical.append(rho)
    if not lags:
        return lambda1, acf1

    lag_array = np.asarray(lags, dtype=float)
    empirical_array = np.asarray(empirical, dtype=float)

    def score(lambda_ou: float) -> float:
        fitted = np.exp(-float(lambda_ou) * float(dt) * lag_array)
        return float(np.sum((empirical_array - fitted) ** 2))

    lower = 1e-8
    upper_candidates = [20.0 / max(float(dt), 1e-12), lambda1 * 20.0]
    upper = max(value for value in upper_candidates if np.isfinite(value) and value > lower)
    grid = np.geomspace(lower, upper, 500)
    scores = np.asarray([score(value) for value in grid], dtype=float)
    best_idx = int(np.nanargmin(scores))
    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(len(grid) - 1, best_idx + 1)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float(grid[best_idx]), acf1

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    c_score = score(c)
    d_score = score(d)
    for _ in range(80):
        if c_score < d_score:
            hi = d
            d = c
            d_score = c_score
            c = hi - inv_phi * (hi - lo)
            c_score = score(c)
        else:
            lo = c
            c = d
            c_score = d_score
            d = lo + inv_phi * (hi - lo)
            d_score = score(d)
    return float((lo + hi) / 2.0), acf1


def _nig_ou_innovations(x: np.ndarray, lambda_ou: float, dt: float) -> np.ndarray:
    """Compute Y_k = exp(lambda dt) X_k - X_{k-1}."""
    return np.exp(lambda_ou * dt) * x[1:] - x[:-1]


def _nig_ou_moment_start(
    y: np.ndarray,
    lambda_ou: float,
    dt: float,
) -> dict[str, float | bool | str]:
    """
    Moment-based starting values for alpha, beta, and delta.

    If the skewness formula is numerically unusable, fall back to a symmetric
    NIG starting point. This fallback is deliberately conservative and only
    meant to give the optimiser a feasible place to start.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    ybar = float(np.mean(y))
    s2 = float(np.var(y, ddof=1))
    s = float(np.sqrt(s2))
    if not np.isfinite(s) or s <= 0:
        s = 1e-4
        s2 = s**2

    centered = y - ybar
    skew = float(np.mean(centered**3) / s**3) if s > 0 else np.nan
    a1 = float(np.exp(lambda_ou * dt) - 1.0)
    a2 = float(np.exp(2.0 * lambda_ou * dt) - 1.0)
    a3 = float(np.exp(3.0 * lambda_ou * dt) - 1.0)

    use_fallback = (
        not np.isfinite(skew)
        or abs(skew) < 1e-8
        or a1 <= 0
        or a2 <= 0
        or a3 <= 0
        or abs(ybar) < 1e-12
    )

    if not use_fallback:
        delta0_sq = (
            (3.0 * s * ybar * a3) / (skew * a2**2 * a1)
            - (ybar / a1) ** 2
        )
        use_fallback = not np.isfinite(delta0_sq) or delta0_sq <= 0
    else:
        delta0_sq = np.nan

    if use_fallback:
        delta0 = max(s, 1e-6)
        alpha0 = max(1.0 / max(s, 1e-6), 1e-6)
        beta0 = 0.0
        return {
            "alpha0": float(alpha0),
            "beta0": float(beta0),
            "delta0": float(delta0),
            "moment_start_used": False,
            "moment_start_reason": "Used symmetric fallback because moment start was not feasible.",
            "sample_mean_y": ybar,
            "sample_std_y": s,
            "sample_skew_y": skew,
        }

    delta0 = float(np.sqrt(delta0_sq))
    alpha0 = float(
        ((ybar**2 + (a1 * delta0) ** 2) ** 1.5 * a2)
        / (delta0**2 * s2 * a1**3)
    )
    beta0 = float(
        alpha0 * ybar / np.sqrt(ybar**2 + (a1 * delta0) ** 2)
    )

    if (
        not np.isfinite(alpha0)
        or not np.isfinite(beta0)
        or not np.isfinite(delta0)
        or alpha0 <= 0
        or delta0 <= 0
        or abs(beta0) >= alpha0
    ):
        delta0 = max(s, 1e-6)
        alpha0 = max(1.0 / max(s, 1e-6), 1e-6)
        beta0 = 0.0
        return {
            "alpha0": float(alpha0),
            "beta0": float(beta0),
            "delta0": float(delta0),
            "moment_start_used": False,
            "moment_start_reason": "Used symmetric fallback because moment start violated constraints.",
            "sample_mean_y": ybar,
            "sample_std_y": s,
            "sample_skew_y": skew,
        }

    return {
        "alpha0": alpha0,
        "beta0": beta0,
        "delta0": delta0,
        "moment_start_used": True,
        "moment_start_reason": "Moment start succeeded.",
        "sample_mean_y": ybar,
        "sample_std_y": s,
        "sample_skew_y": skew,
    }


def _nig_raw_to_params(params_raw: np.ndarray) -> tuple[float, float, float, float]:
    """Transform unconstrained raw parameters into valid NIG-OU parameters."""
    log_alpha, raw_rho, log_delta, log_lambda = params_raw
    alpha = float(np.exp(log_alpha))
    rho = float(np.tanh(raw_rho))
    beta = float(alpha * rho)
    delta_nig = float(np.exp(log_delta))
    lambda_ou = float(np.exp(log_lambda))
    return alpha, beta, delta_nig, lambda_ou


def _nig_params_to_raw(
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
) -> np.ndarray:
    """Pack valid NIG-OU parameters into the unconstrained optimizer scale."""
    alpha = max(float(alpha), 1e-12)
    delta_nig = max(float(delta_nig), 1e-12)
    lambda_ou = max(float(lambda_ou), 1e-12)
    rho = float(np.clip(float(beta) / alpha, -0.999999, 0.999999))
    return np.array(
        [
            np.log(alpha),
            np.arctanh(rho),
            np.log(delta_nig),
            np.log(lambda_ou),
        ],
        dtype=float,
    )


def _nig_gamma_raw_to_params(params_raw: np.ndarray) -> tuple[float, float, float, float]:
    """
    Transform gamma/rho raw parameters into NIG-OU parameters.

    gamma = sqrt(alpha^2 - beta^2), rho = beta / alpha.
    This is often less coupled than optimizing alpha and beta directly.
    """
    log_gamma, raw_rho, log_delta, log_lambda = params_raw
    gamma = float(np.exp(log_gamma))
    rho = float(np.tanh(raw_rho))
    alpha = float(gamma / np.sqrt(max(1.0 - rho**2, 1e-12)))
    beta = float(rho * alpha)
    delta_nig = float(np.exp(log_delta))
    lambda_ou = float(np.exp(log_lambda))
    return alpha, beta, delta_nig, lambda_ou


def _nig_params_to_gamma_raw(
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
) -> np.ndarray:
    """Pack NIG-OU parameters into gamma/rho optimizer coordinates."""
    alpha = max(float(alpha), 1e-12)
    rho = float(np.clip(float(beta) / alpha, -0.999999, 0.999999))
    gamma = max(alpha * np.sqrt(max(1.0 - rho**2, 1e-12)), 1e-12)
    delta_nig = max(float(delta_nig), 1e-12)
    lambda_ou = max(float(lambda_ou), 1e-12)
    return np.array(
        [
            np.log(gamma),
            np.arctanh(rho),
            np.log(delta_nig),
            np.log(lambda_ou),
        ],
        dtype=float,
    )


def _nig_raw_in_bounds(params_raw: np.ndarray) -> bool:
    """Keep the trial NIG optimizer inside the same guardrails as older code."""
    if not np.all(np.isfinite(params_raw)):
        return False
    return bool(
        -20 <= params_raw[0] <= 20
        and -8 <= params_raw[1] <= 8
        and -30 <= params_raw[2] <= 20
        and -20 <= params_raw[3] <= 20
    )


def _empirical_cumulants(values: np.ndarray) -> dict[str, float]:
    """Empirical cumulants 1-4 using population central moments."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"c1": np.nan, "c2": np.nan, "c3": np.nan, "c4": np.nan}
    c1 = float(np.mean(x))
    centered = x - c1
    c2 = float(np.mean(centered**2))
    c3 = float(np.mean(centered**3))
    c4 = float(np.mean(centered**4) - 3.0 * c2**2)
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


def _nig_stationary_moment_match_start(
    target_cumulants: dict[str, float],
    method: str,
) -> dict[str, float | bool | str]:
    """
    Fit stationary NIG(alpha,beta,delta) cumulants to target cumulants.

    This is a Wu-style stationary NIG-OU helper for multistart likelihood
    estimation. It is deliberately used only as an initial guess, not as the
    final estimator.
    """
    from scipy.optimize import least_squares

    target = np.array(
        [
            float(target_cumulants.get("c1", np.nan)),
            float(target_cumulants.get("c2", np.nan)),
            float(target_cumulants.get("c3", np.nan)),
            float(target_cumulants.get("c4", np.nan)),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(target)) or target[1] <= 0:
        return {"valid": False, "method": method, "reason": "Invalid target cumulants."}

    c2 = max(abs(float(target[1])), 1e-12)
    scales = np.array(
        [
            max(abs(float(target[0])), np.sqrt(c2) * 0.1, 1e-10),
            c2,
            max(abs(float(target[2])), c2 ** 1.5, 1e-12),
            max(abs(float(target[3])), c2**2, 1e-12),
        ],
        dtype=float,
    )

    def unpack(raw: np.ndarray) -> tuple[float, float, float]:
        alpha = float(np.exp(raw[0]))
        beta = float(alpha * np.tanh(raw[1]))
        delta_nig = float(np.exp(raw[2]))
        return alpha, beta, delta_nig

    def residuals(raw: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(raw)):
            return np.full(4, 1e6)
        if raw[0] < -20 or raw[0] > 20 or raw[1] < -8 or raw[1] > 8 or raw[2] < -30 or raw[2] > 20:
            return np.full(4, 1e6)
        alpha, beta, delta_nig = unpack(raw)
        pred = nig_stationary_cumulants(alpha, beta, delta_nig)
        pred_values = np.array([pred["c1"], pred["c2"], pred["c3"], pred["c4"]], dtype=float)
        if not np.all(np.isfinite(pred_values)):
            return np.full(4, 1e6)
        return (pred_values - target) / scales

    std = float(np.sqrt(c2))
    alpha_seeds = [max(v / max(std, 1e-8), 1e-4) for v in (0.5, 1.0, 2.0, 5.0)]
    skew_sign = np.sign(target[2]) if abs(target[2]) > 1e-14 else np.sign(target[0])
    rho_seeds = [0.0, 0.25 * skew_sign, -0.25 * skew_sign, 0.55 * skew_sign]

    best: Any | None = None
    for alpha0 in alpha_seeds:
        for rho0 in rho_seeds:
            delta0 = max(c2 * alpha0 * (1.0 - rho0**2) ** 1.5, 1e-8)
            raw0 = np.array(
                [
                    np.log(alpha0),
                    np.arctanh(float(np.clip(rho0, -0.95, 0.95))),
                    np.log(delta0),
                ],
                dtype=float,
            )
            res = least_squares(
                residuals,
                raw0,
                max_nfev=300,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
            cost = float(2.0 * res.cost)
            if best is None or cost < best[0]:
                best = (cost, res)

    if best is None:
        return {"valid": False, "method": method, "reason": "No moment-match start produced."}

    cost, res = best
    alpha, beta, delta_nig = unpack(res.x)
    valid = bool(
        res.success
        and np.isfinite(cost)
        and alpha > 0
        and abs(beta) < alpha
        and delta_nig > 0
    )
    return {
        "valid": valid,
        "method": method,
        "alpha0": float(alpha),
        "beta0": float(beta),
        "delta0": float(delta_nig),
        "moment_objective": float(cost),
        "reason": str(res.message),
    }


def _nig_multistart_candidates(
    x: np.ndarray,
    dt: float,
    lambda0: float,
    lambda_factors: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Build feasible NIG-OU likelihood starts from moments and lambda perturbations."""
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_candidate(
        alpha: float,
        beta: float,
        delta_nig: float,
        lambda_ou: float,
        method: str,
        moment_start_used: bool | None = None,
        reason: str | None = None,
    ) -> None:
        if not (
            np.isfinite(alpha)
            and np.isfinite(beta)
            and np.isfinite(delta_nig)
            and np.isfinite(lambda_ou)
            and alpha > 0
            and abs(beta) < alpha
            and delta_nig > 0
            and lambda_ou > 0
            and lambda_ou * dt < 20
        ):
            return
        raw = _nig_params_to_gamma_raw(alpha, beta, delta_nig, lambda_ou)
        if not _nig_raw_in_bounds(raw):
            return
        key = tuple(int(round(v * 1000)) for v in raw)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "delta": float(delta_nig),
                "lambda_ou": float(lambda_ou),
                "raw": raw,
                "method": method,
                "moment_start_used": moment_start_used,
                "reason": reason,
            }
        )

    x = np.asarray(x, dtype=float)
    sample_cumulants = _empirical_cumulants(x)
    sample_start = _nig_stationary_moment_match_start(
        sample_cumulants,
        method="sample_stationary_cumulant_match",
    )

    for factor in lambda_factors:
        lambda_candidate = max(float(lambda0) * float(factor), 1e-8)
        if not np.isfinite(lambda_candidate) or lambda_candidate * dt >= 20:
            continue

        y = _nig_ou_innovations(x, lambda_ou=lambda_candidate, dt=dt)
        starts = _nig_ou_moment_start(y, lambda_ou=lambda_candidate, dt=dt)
        add_candidate(
            float(starts["alpha0"]),
            float(starts["beta0"]),
            float(starts["delta0"]),
            lambda_candidate,
            method=f"valdivieso_moment_factor_{factor:g}",
            moment_start_used=bool(starts["moment_start_used"]),
            reason=str(starts["moment_start_reason"]),
        )

        y_cumulants = _empirical_cumulants(y)
        stationary_target: dict[str, float] = {}
        for order in range(1, 5):
            scale = float(np.expm1(order * lambda_candidate * dt))
            value = float(y_cumulants[f"c{order}"])
            stationary_target[f"c{order}"] = value / scale if scale > 0 else np.nan
        wu_start = _nig_stationary_moment_match_start(
            stationary_target,
            method=f"wu_stationary_innovation_cumulant_match_factor_{factor:g}",
        )
        if wu_start.get("valid"):
            add_candidate(
                float(wu_start["alpha0"]),
                float(wu_start["beta0"]),
                float(wu_start["delta0"]),
                lambda_candidate,
                method=str(wu_start["method"]),
                moment_start_used=True,
                reason=str(wu_start.get("reason")),
            )

        if sample_start.get("valid"):
            add_candidate(
                float(sample_start["alpha0"]),
                float(sample_start["beta0"]),
                float(sample_start["delta0"]),
                lambda_candidate,
                method=f"sample_stationary_cumulant_match_factor_{factor:g}",
                moment_start_used=True,
                reason=str(sample_start.get("reason")),
            )

        y_std = float(np.std(y[np.isfinite(y)], ddof=1)) if np.isfinite(y).sum() > 1 else 1e-4
        for alpha_multiplier in (0.5, 1.0, 2.0):
            alpha0 = max(alpha_multiplier / max(y_std, 1e-8), 1e-6)
            delta0 = max(y_std**2 * alpha0, 1e-10)
            add_candidate(
                alpha0,
                0.0,
                delta0,
                lambda_candidate,
                method=f"symmetric_fallback_factor_{factor:g}_scale_{alpha_multiplier:g}",
                moment_start_used=False,
                reason="Generic symmetric NIG fallback start.",
            )

    return candidates


def _limit_nig_candidates_for_scoring(
    candidates: list[dict[str, Any]],
    max_candidates: int | None,
    lambda0: float,
) -> list[dict[str, Any]]:
    """
    Keep a smaller, diverse set of NIG-OU starts before likelihood scoring.

    Scoring every candidate is useful but expensive. This helper keeps starts
    across the main construction methods and lambda perturbations, preferring
    candidates closer to the ACF-implied lambda while still preserving diversity.
    """
    if max_candidates is None or max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates

    def source_key(method: str) -> str:
        if method.startswith("wu_stationary"):
            return "wu_stationary"
        if method.startswith("valdivieso"):
            return "valdivieso"
        if method.startswith("sample_stationary"):
            return "sample_stationary"
        if method.startswith("symmetric"):
            return "symmetric"
        return "other"

    lambda0 = max(float(lambda0), 1e-12)
    priority = ["wu_stationary", "valdivieso", "sample_stationary", "symmetric", "other"]
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in priority}
    for candidate in candidates:
        grouped.setdefault(source_key(str(candidate.get("method", ""))), []).append(candidate)

    for group in grouped.values():
        group.sort(
            key=lambda item: (
                abs(np.log(max(float(item["lambda_ou"]), 1e-12) / lambda0)),
                str(item.get("method", "")),
            )
        )

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    while len(selected) < max_candidates:
        added = False
        for key in priority:
            group = grouped.get(key, [])
            while group and id(group[0]) in seen:
                group.pop(0)
            if not group:
                continue
            candidate = group.pop(0)
            selected.append(candidate)
            seen.add(id(candidate))
            added = True
            if len(selected) >= max_candidates:
                break
        if not added:
            break

    return selected







def ar1_formation_mean(spread: pd.Series | np.ndarray) -> dict[str, Any]:
    """AR(1) formation mean used as the fixed OU level for residual NIG fitting."""

    values = np.asarray(spread, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        return {"valid": False, "reason": "fewer than 100 observations"}
    lag = values[:-1]
    nxt = values[1:]
    lag_mean = float(np.mean(lag))
    nxt_mean = float(np.mean(nxt))
    denominator = float(np.sum((lag - lag_mean) ** 2))
    if denominator <= 0 or not np.isfinite(denominator):
        return {"valid": False, "reason": "zero or non-finite AR(1) denominator"}
    phi = float(np.sum((lag - lag_mean) * (nxt - nxt_mean)) / denominator)
    a = float(nxt_mean - phi * lag_mean)
    residuals = nxt - (a + phi * lag)
    eps_var = float(np.var(residuals, ddof=1))
    base = {
        "ar1_a": a,
        "ar1_phi": phi,
        "ar1_eps_var": eps_var,
        "ar1_observations": int(len(values)),
    }
    if not (0 < phi < 1) or not np.isfinite(eps_var) or eps_var < 0:
        return {
            **base,
            "valid": False,
            "reason": "AR(1) phi is outside (0,1) or innovation variance is invalid",
        }
    theta = float(-math.log(phi))
    mu = float(a / (1.0 - phi))
    sigma_eq = float(math.sqrt(eps_var / (1.0 - phi * phi)))
    return {
        **base,
        "valid": True,
        "reason": "",
        "mu_form": mu,
        "ar1_theta_per_minute": theta,
        "ar1_half_life_minutes": float(math.log(2.0) / theta),
        "ar1_sigma_eq": sigma_eq,
    }


def natural_nig_mean(alpha: float, beta: float, delta: float) -> float:
    """NIG stationary mean before applying the centred-residual location correction."""

    return float(delta * beta / np.sqrt(alpha * alpha - beta * beta))


def centered_stationary_cumulants(alpha: float, beta: float, delta: float) -> dict[str, float]:
    """Stationary cumulants after centring the residual process."""

    cumulants = nig_stationary_cumulants(alpha, beta, delta)
    cumulants["c1"] = 0.0
    return cumulants


def centered_zstar_cf(
    u: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
) -> np.ndarray:
    """CF of Z* = exp(lambda dt)Y_t - Y_{t-dt} with centred NIG stationary law."""

    u = np.asarray(u, dtype=float)
    e = np.exp(lambda_ou * dt)
    gamma = np.sqrt(alpha * alpha - beta * beta)
    ell = -delta_nig * beta / gamma
    old = nig_ou_zstar_cf(
        u,
        alpha=alpha,
        beta=beta,
        delta_nig=delta_nig,
        lambda_ou=lambda_ou,
        dt=dt,
    )
    return np.exp(1j * ell * (e - 1.0) * u) * old


def centered_zstar_cumulants(
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
) -> dict[str, float]:
    """Cumulants for the centred-residual NIG-OU transition transform."""

    stationary = centered_stationary_cumulants(alpha, beta, delta_nig)
    return {
        f"c{order}": float(np.expm1(order * float(lambda_ou) * float(dt)))
        * stationary[f"c{order}"]
        for order in range(1, 5)
    }


def centered_zstar_density_fft(
    y_values: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
    lambda_ou: float,
    dt: float,
    fft_grid_size: int,
    truncation_l: float,
    density_floor: float,
    fft_interval: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """FFT transition density for centred NIG residual innovations."""

    y = np.asarray(y_values, dtype=float)
    y = y[np.isfinite(y)]
    if fft_interval == "cumulant":
        left, right, interval_diagnostics = _nig_cumulant_interval_from_cumulants(
            centered_zstar_cumulants(alpha, beta, delta_nig, lambda_ou, dt),
            truncation_l=truncation_l,
        )
        interval_diagnostics["interval_source"] = "centered_residual_cumulant"
    elif fft_interval == "sample":
        left, right, interval_diagnostics = _nig_sample_fft_interval(
            y,
            truncation_l=truncation_l,
        )
    else:
        raise ValueError(f"Unknown NIG FFT interval: {fft_interval}")
    return _fft_density_from_cf_real_line(
        y,
        lambda u: centered_zstar_cf(u, alpha, beta, delta_nig, lambda_ou, dt),
        left=left,
        right=right,
        fft_grid_size=fft_grid_size,
        density_floor=density_floor,
        interval_diagnostics=interval_diagnostics,
    )


def centered_stationary_density(
    x_values: np.ndarray,
    alpha: float,
    beta: float,
    delta_nig: float,
    density_floor: float,
) -> np.ndarray:
    """Stationary centred-residual density."""

    gamma = np.sqrt(alpha * alpha - beta * beta)
    ell = -delta_nig * beta / gamma
    dens = stats.norminvgauss.pdf(
        np.asarray(x_values, dtype=float),
        alpha * delta_nig,
        beta * delta_nig,
        loc=ell,
        scale=delta_nig,
    )
    return np.maximum(np.asarray(dens, dtype=float), float(density_floor))


def estimate_centered_nig_residual_fft_multistart(
    spread: pd.Series | np.ndarray,
    dt: float = DT_MINUTE,
    fft_grid_size: int = 8192,
    truncation_l: float = 10.0,
    maxiter: int = 220,
    optimizer_method: str = "Powell",
    density_floor: float = 1e-300,
    lambda_factors: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
    lambda_max_lag: int = 20,
    max_optimizer_starts: int = 3,
    max_candidate_starts_to_score: int | None = None,
    fft_intervals: tuple[str, ...] = ("cumulant", "sample"),
    include_stationary_density: bool = True,
    wu_first: bool = True,
) -> dict[str, Any]:
    """FFT multistart likelihood for the centred residual process."""

    x = np.asarray(spread, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return {"valid": False, "reason": "Need at least 10 observations."}
    started = perf_counter()
    lambda0, acf1 = _nig_ou_lambda_start_multilag(x, dt=dt, max_lag=lambda_max_lag)
    candidates = _nig_multistart_candidates(
        x=x,
        dt=dt,
        lambda0=lambda0,
        lambda_factors=tuple(lambda_factors),
    )
    generated_candidate_starts = int(len(candidates))
    candidates = _limit_nig_candidates_for_scoring(
        candidates,
        max_candidates=max_candidate_starts_to_score,
        lambda0=lambda0,
    )
    if not candidates:
        return {"valid": False, "reason": "No feasible starts.", "initial_lambda_ou": float(lambda0)}

    def neg_loglik(params_raw: np.ndarray, fft_interval: str) -> float:
        if not _nig_raw_in_bounds(params_raw):
            return 1e100
        alpha, beta, delta_nig, lambda_ou = _nig_gamma_raw_to_params(params_raw)
        if not (
            alpha > 0
            and abs(beta) < alpha
            and delta_nig > 0
            and lambda_ou > 0
            and lambda_ou * dt < 20
        ):
            return 1e100
        y = _nig_ou_innovations(x, lambda_ou=lambda_ou, dt=dt)
        densities, _ = centered_zstar_density_fft(
            y,
            alpha=alpha,
            beta=beta,
            delta_nig=delta_nig,
            lambda_ou=lambda_ou,
            dt=dt,
            fft_grid_size=fft_grid_size,
            truncation_l=truncation_l,
            density_floor=density_floor,
            fft_interval=fft_interval,
        )
        if len(densities) != len(y) or not np.all(np.isfinite(densities)):
            return 1e100
        loglik = len(y) * lambda_ou * dt + float(np.sum(np.log(densities)))
        if include_stationary_density:
            stationary_density = centered_stationary_density(
                np.asarray([x[0]], dtype=float),
                alpha=alpha,
                beta=beta,
                delta_nig=delta_nig,
                density_floor=density_floor,
            )
            loglik += float(np.log(stationary_density[0]))
        return -loglik if np.isfinite(loglik) else 1e100

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        for fft_interval in fft_intervals:
            objective = float(neg_loglik(candidate["raw"], fft_interval=str(fft_interval)))
            if np.isfinite(objective):
                scored.append({**candidate, "fft_interval": str(fft_interval), "start_neg_loglik": objective})
    scored.sort(key=lambda item: item["start_neg_loglik"])
    if not scored:
        return {"valid": False, "reason": "Every candidate had non-finite likelihood."}

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()

    def add_selected(item: dict[str, Any]) -> None:
        source = str(item["method"]).split("_factor_")[0]
        interval = str(item.get("fft_interval", ""))
        lambda_key = int(round(np.log(float(item["lambda_ou"])) * 1000))
        key = (f"{source}:{interval}", lambda_key)
        if key not in selected_keys:
            selected.append(item)
            selected_keys.add(key)

    if wu_first:
        for item in [row for row in scored if str(row["method"]).startswith("wu_stationary")]:
            add_selected(item)
            if len(selected) >= max_optimizer_starts:
                break
    for item in scored[: max(1, max_optimizer_starts // 2)]:
        if len(selected) >= max_optimizer_starts:
            break
        add_selected(item)
    for item in scored:
        if len(selected) >= max_optimizer_starts:
            break
        add_selected(item)
    if not selected:
        selected = scored[:max_optimizer_starts]

    options: dict[str, Any] = {"maxiter": int(maxiter)}
    if optimizer_method == "Nelder-Mead":
        options.update({"xatol": 1e-4, "fatol": 1e-4})
    elif optimizer_method == "Powell":
        options.update({"xtol": 1e-4, "ftol": 1e-4})

    optimizer_rows: list[dict[str, Any]] = []
    best: tuple[float, Any, dict[str, Any]] | None = None
    for start_info in selected:
        interval = str(start_info.get("fft_interval", "cumulant"))
        result = minimize(
            lambda values, fft_interval=interval: neg_loglik(values, fft_interval=fft_interval),
            np.asarray(start_info["raw"], dtype=float),
            method=optimizer_method,
            options=options,
        )
        objective = float(result.fun) if np.isfinite(result.fun) else 1e100
        alpha_i, beta_i, delta_i, lambda_i = _nig_gamma_raw_to_params(result.x)
        optimizer_rows.append(
            {
                "start_method": start_info["method"],
                "fft_interval": interval,
                "start_neg_loglik": float(start_info["start_neg_loglik"]),
                "optimizer_success": bool(result.success),
                "optimizer_neg_loglik": objective,
                "alpha": float(alpha_i),
                "beta": float(beta_i),
                "delta": float(delta_i),
                "lambda_ou": float(lambda_i),
                "nfev": int(result.nfev) if hasattr(result, "nfev") else None,
                "nit": int(result.nit) if hasattr(result, "nit") else None,
            }
        )
        if best is None or objective < best[0]:
            best = (objective, result, start_info)
    if best is None:
        return {"valid": False, "reason": "No optimizer run completed."}

    _, result, best_start = best
    alpha_hat, beta_hat, delta_hat, lambda_hat = _nig_gamma_raw_to_params(result.x)
    z_hat = _nig_ou_innovations(x, lambda_ou=lambda_hat, dt=dt)
    best_interval = str(best_start.get("fft_interval", "cumulant"))
    densities_hat, density_diagnostics = centered_zstar_density_fft(
        z_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        delta_nig=delta_hat,
        lambda_ou=lambda_hat,
        dt=dt,
        fft_grid_size=fft_grid_size,
        truncation_l=truncation_l,
        density_floor=density_floor,
        fft_interval=best_interval,
    )
    loglik = len(z_hat) * lambda_hat * dt + float(np.sum(np.log(densities_hat)))
    if include_stationary_density:
        stationary_density = centered_stationary_density(
            np.asarray([x[0]], dtype=float),
            alpha=alpha_hat,
            beta=beta_hat,
            delta_nig=delta_hat,
            density_floor=density_floor,
        )
        loglik += float(np.log(stationary_density[0]))
    elapsed = perf_counter() - started
    valid = bool(np.isfinite(loglik) and alpha_hat > 0 and abs(beta_hat) < alpha_hat and delta_hat > 0 and lambda_hat > 0)
    start_table = pd.DataFrame(optimizer_rows).sort_values("optimizer_neg_loglik")
    best_row = start_table.iloc[0].to_dict() if not start_table.empty else {}
    natural_m = natural_nig_mean(alpha_hat, beta_hat, delta_hat)
    return {
        "valid": valid,
        "reason": None if valid else str(result.message),
        "estimation_method": "centered_nig_residual_transition_mle_fft_multistart",
        "centered_residual_location_corrected": True,
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "delta": float(delta_hat),
        "lambda_ou": float(lambda_hat),
        "lambda": float(lambda_hat),
        "internal_location_ell": float(-natural_m),
        "natural_nig_mean_before_location": float(natural_m),
        "residual_stationary_mean": 0.0,
        "loglik": float(loglik),
        "neg_loglik": float(-loglik),
        "dt": float(dt),
        "observations": int(len(x)),
        "increments": int(len(z_hat)),
        "acf1": float(acf1) if np.isfinite(acf1) else np.nan,
        "initial_lambda_ou": float(lambda0),
        "initial_alpha": float(best_start["alpha"]),
        "initial_beta": float(best_start["beta"]),
        "initial_delta": float(best_start["delta"]),
        "best_start_method": str(best_start["method"]),
        "best_fft_interval": best_interval,
        "best_start_neg_loglik": float(best_start["start_neg_loglik"]),
        "best_optimizer_start_method": str(best_row.get("start_method", best_start["method"])),
        "best_optimizer_fft_interval": str(best_row.get("fft_interval", best_interval)),
        "generated_candidate_starts": generated_candidate_starts,
        "candidate_starts": int(len(candidates)),
        "scored_starts": int(len(scored)),
        "optimizer_starts": int(len(selected)),
        "include_stationary_density": bool(include_stationary_density),
        "optimizer_method": optimizer_method,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit) if hasattr(result, "nit") else None,
        "optimizer_evaluations": int(result.nfev) if hasattr(result, "nfev") else None,
        "fit_seconds": float(elapsed),
        "fft_grid_size": int(fft_grid_size),
        "truncation_l": float(truncation_l),
        "density_floor": float(density_floor),
        **{f"fft_{key}": value for key, value in density_diagnostics.items()},
    }


def estimate_nig_ou_fixed_mean_fft_multistart(
    spread: pd.Series | np.ndarray,
    u_form: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate NIG-OU with the OU mean fixed to a formation-window level.

    Parameters
    ----------
    spread:
        Raw formation spread series.
    u_form:
        Fixed formation mean, normally `gaussian_fit["mu"]`.

    The likelihood is fitted to `spread - u_form`, but the returned process is
    explicitly the fixed-mean model `X_t = u_form + Y_t`.
    """

    if not np.isfinite(float(u_form)):
        return {"valid": False, "reason": "u_form must be finite.", "u_form": float(u_form)}
    x = np.asarray(spread, dtype=float)
    x = x[np.isfinite(x)]
    centered = x - float(u_form)
    fit = estimate_centered_nig_residual_fft_multistart(centered, **kwargs)
    out = dict(fit)
    out.update(
        {
            "estimation_method": "nig_ou_fixed_mean_transition_mle_fft_multistart",
            "fixed_mean": True,
            "u_form": float(u_form),
            "mu_form": float(u_form),
            "process_mean": float(u_form),
            "residual_mean_target": 0.0,
            "stationary_mean_total": float(u_form) + float(out.get("residual_stationary_mean", 0.0)),
            "centered_observations": int(len(centered)),
            "centered_sample_mean": float(np.mean(centered)) if len(centered) else np.nan,
        }
    )
    return out


__all__ = [
    "ar1_formation_mean",
    "estimate_centered_nig_residual_fft_multistart",
    "estimate_nig_ou_fixed_mean_fft_multistart",
]
