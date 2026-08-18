"""Wu-style estimator for the symmetric bilateral-Gamma OU model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AR1MeanReversionFit:
    """AR(1) long-run mean and OU mean-reversion fit."""

    intercept: float
    c: float
    mu: float
    kappa: float
    dt: float
    residuals: np.ndarray
    residual_mean: float
    residual_variance: float
    n_transitions: int
    r_squared: float


@dataclass(frozen=True)
class WuSymmetricBGOUFit:
    """Symmetric BG-OU fit from AR(1) innovations and Wu cumulant moments."""

    valid: bool
    mu: float
    u_form: float
    mu_form: float
    process_mean: float
    residual_stationary_mean: float
    internal_location_ell: float
    alpha: float
    beta: float
    kappa: float
    dt: float
    c: float
    loglik: float
    n_obs: int
    n_transitions: int
    success: bool
    message: str
    intercept: float
    atom_mass: float
    atom_location: float
    residual_atom_location: float
    stationary_variance: float
    stationary_excess_kurtosis: float
    stationary_sigma: float
    innovation_mean: float
    innovation_variance: float
    innovation_fourth_moment: float
    innovation_fourth_cumulant: float
    innovation_excess_kurtosis: float
    ar1_r_squared: float
    start_values: dict[str, float]

    @property
    def a(self) -> float:
        """Notebook-compatible alias for alpha."""
        return self.alpha

    @property
    def b(self) -> float:
        """Notebook-compatible alias for beta."""
        return self.beta

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_1d(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Need at least 10 finite observations.")
    if np.var(x) <= 0:
        raise ValueError("Observed series has zero variance.")
    return x


def estimate_ar1_mean_reversion(
    values: np.ndarray,
    *,
    dt: float,
) -> AR1MeanReversionFit:
    """Estimate the AR(1) long-run mean and OU speed by OLS."""
    x = _clean_1d(values)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    lagged = x[:-1]
    current = x[1:]
    design = np.column_stack((np.ones_like(lagged), lagged))
    coefficients, _, _, _ = np.linalg.lstsq(design, current, rcond=None)
    intercept = float(coefficients[0])
    c = float(coefficients[1])

    if not np.isfinite(c) or not (0.0 < c < 1.0):
        raise ValueError(f"AR(1) estimate must satisfy 0 < c < 1; obtained c={c:.8g}.")

    mu = float(intercept / (1.0 - c))
    kappa = float(-np.log(c) / dt)
    fitted = intercept + c * lagged
    residuals = current - fitted
    residual_mean = float(np.mean(residuals))
    residual_variance = float(np.mean((residuals - residual_mean) ** 2))

    total_ss = float(np.sum((current - np.mean(current)) ** 2))
    residual_ss = float(np.sum(residuals**2))
    r_squared = float(1.0 - residual_ss / total_ss) if total_ss > 0 else np.nan

    return AR1MeanReversionFit(
        intercept=intercept,
        c=c,
        mu=mu,
        kappa=kappa,
        dt=float(dt),
        residuals=residuals,
        residual_mean=residual_mean,
        residual_variance=residual_variance,
        n_transitions=int(residuals.size),
        r_squared=r_squared,
    )


def _innovation_moment_diagnostics(residuals: np.ndarray) -> dict[str, float]:
    """Return centered empirical innovation moments and cumulants."""
    eps = np.asarray(residuals, dtype=float)
    centered = eps - np.mean(eps)
    m2 = float(np.mean(centered**2))
    m4 = float(np.mean(centered**4))
    k4 = float(m4 - 3.0 * m2**2)
    excess = float(k4 / m2**2) if m2 > 0 else np.nan
    return {
        "mean": float(np.mean(eps)),
        "variance": m2,
        "fourth_moment": m4,
        "fourth_cumulant": k4,
        "excess_kurtosis": excess,
    }


def estimate_symmetric_bg_ou_wu_innovation_moments(
    values: np.ndarray,
    *,
    dt: float = 1.0 / 390.0,
) -> WuSymmetricBGOUFit:
    r"""Estimate symmetric BG-OU parameters from AR(1) innovation cumulants.

    The first step fits ``X_i = A + c X_{i-1} + epsilon_i`` by OLS and fixes
    the process mean to ``mu = A / (1 - c)``. The BG parameters are then
    estimated from the centered second and fourth cumulants of the innovations:

    ``k2 = 2 alpha (1-c^2) / beta^2``

    ``k4 = 12 alpha (1-c^4) / beta^4``

    which gives Wu-style closed-form estimates for ``alpha`` and ``beta``. A
    positive empirical fourth cumulant is required; otherwise the symmetric BG
    innovation-moment equations have no admissible solution.
    """
    x = _clean_1d(values)
    ar1 = estimate_ar1_mean_reversion(x, dt=dt)
    diagnostics = _innovation_moment_diagnostics(ar1.residuals)

    k2 = float(diagnostics["variance"])
    k4 = float(diagnostics["fourth_cumulant"])
    c = float(ar1.c)

    if not np.isfinite(k2) or k2 <= 0:
        raise ValueError("Innovation variance must be positive for Wu estimation.")
    if not np.isfinite(k4) or k4 <= 0:
        raise ValueError(
            "Innovation fourth cumulant must be positive for the symmetric BG Wu estimator."
        )

    one_minus_c2 = 1.0 - c**2
    alpha = float(3.0 * k2**2 * (1.0 + c**2) / (k4 * one_minus_c2))
    beta = float(np.sqrt(2.0 * alpha * one_minus_c2 / k2))
    stationary_variance = float(2.0 * alpha / beta**2)
    stationary_excess_kurtosis = float(3.0 / alpha)
    atom_mass = float(np.exp(2.0 * alpha * np.log(c)))
    atom_location = float((1.0 - c) * ar1.mu)
    success = bool(
        np.isfinite(alpha)
        and np.isfinite(beta)
        and np.isfinite(ar1.kappa)
        and alpha > 0
        and beta > 0
        and ar1.kappa > 0
        and 0.0 < c < 1.0
        and 0.0 <= atom_mass <= 1.0
    )

    return WuSymmetricBGOUFit(
        valid=success,
        mu=float(ar1.mu),
        u_form=float(ar1.mu),
        mu_form=float(ar1.mu),
        process_mean=float(ar1.mu),
        residual_stationary_mean=0.0,
        internal_location_ell=0.0,
        alpha=alpha,
        beta=beta,
        kappa=float(ar1.kappa),
        dt=float(ar1.dt),
        c=c,
        loglik=np.nan,
        n_obs=int(x.size),
        n_transitions=int(ar1.n_transitions),
        success=success,
        message=(
            "Wu innovation-moment estimates computed successfully."
            if success
            else "Wu innovation-moment estimates were non-finite or invalid."
        ),
        intercept=float(ar1.intercept),
        atom_mass=atom_mass,
        atom_location=atom_location,
        residual_atom_location=0.0,
        stationary_variance=stationary_variance,
        stationary_excess_kurtosis=stationary_excess_kurtosis,
        stationary_sigma=float(np.sqrt(stationary_variance)),
        innovation_mean=float(diagnostics["mean"]),
        innovation_variance=k2,
        innovation_fourth_moment=float(diagnostics["fourth_moment"]),
        innovation_fourth_cumulant=k4,
        innovation_excess_kurtosis=float(diagnostics["excess_kurtosis"]),
        ar1_r_squared=float(ar1.r_squared),
        start_values={
            "mu": float(ar1.mu),
            "intercept": float(ar1.intercept),
            "c": c,
            "kappa": float(ar1.kappa),
            "innovation_k2": k2,
            "innovation_k4": k4,
        },
    )


__all__ = [
    "AR1MeanReversionFit",
    "WuSymmetricBGOUFit",
    "estimate_ar1_mean_reversion",
    "estimate_symmetric_bg_ou_wu_innovation_moments",
]
