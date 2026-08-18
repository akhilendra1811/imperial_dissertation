"""Model registry used by clean experiment runners."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from levy_ou.config import DT_MINUTE
from levy_ou.estimators.cgmy_ou import estimate_cgmy_ou_fft_mle, stationary_cumulants_zero_mean
from levy_ou.estimators.cp_ou import estimate_cp_ou_fixed_mean
from levy_ou.estimators.gaussian_ou import fit_brownian_ou_from_spread
from levy_ou.estimators.nig_ou import estimate_nig_ou_fixed_mean_fft_multistart
from levy_ou.estimators.symmetric_bg_ou import estimate_symmetric_bg_ou_wu_innovation_moments
from levy_ou.experiments.outputs import scalarize_mapping
from levy_ou.simulation.cgmy_simulator import build_cgmy_ou_simulator_with_fallback
from levy_ou.simulation.nig_simulator import NIGOUFGMC, build_nig_ou_simulator_with_fallback
from levy_ou.simulation.symmetric_bg_ou import SymmetricBGOU


MODEL_CHOICES = ("gaussian", "cp", "nig", "cgmy", "symmetric_bg")
DEFAULT_SIM_FFT_GRID_SIZE = 2**15  # 32768: default NIG/CGMY simulation inversion grid.


def _finite_spread(spread: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(spread, dtype=float)
    return x[np.isfinite(x)]


def _simulate_gaussian_ou(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """Exact-discretisation Gaussian OU paths."""

    rng = np.random.default_rng(seed)
    theta = float(theta)
    mu = float(mu)
    sigma = float(sigma)
    rho = float(np.exp(-theta * dt))
    if theta > 0:
        eps_sigma = float(sigma * np.sqrt((1.0 - rho * rho) / (2.0 * theta)))
    else:
        eps_sigma = float(abs(sigma) * np.sqrt(dt))
    paths = np.zeros((int(n_paths), int(n_steps)), dtype=float)
    paths[:, 0] = float(x0)
    for step in range(1, int(n_steps)):
        paths[:, step] = mu + rho * (paths[:, step - 1] - mu) + rng.normal(scale=eps_sigma, size=int(n_paths))
    return paths


def _simulate_cp_ou(
    theta: float,
    mu: float,
    sigma: float,
    lambda_jump: float,
    eta: float,
    delta_jump: float,
    x0: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """Euler CP-OU paths using the fitted double-exponential jump parameters."""

    rng = np.random.default_rng(seed)
    theta = float(theta)
    mu = float(mu)
    sigma = float(sigma)
    lambda_jump = float(lambda_jump)
    eta = float(eta)
    delta_jump = float(delta_jump)
    dt = float(dt)
    paths = np.zeros((int(n_paths), int(n_steps)), dtype=float)
    paths[:, 0] = float(x0)
    use_jumps = bool(np.isfinite(lambda_jump) and lambda_jump > 0.0 and np.isfinite(eta) and eta > 0.0)
    for step in range(1, int(n_steps)):
        previous = paths[:, step - 1]
        drift = theta * (mu - previous) * dt
        diffusion = sigma * np.sqrt(dt) * rng.normal(size=int(n_paths))
        jumps = np.zeros(int(n_paths), dtype=float)
        if use_jumps:
            counts = rng.poisson(lambda_jump * dt, size=int(n_paths))
            for path_idx, count in enumerate(counts):
                if count <= 0:
                    continue
                signs = rng.choice(np.array([-1.0, 1.0]), size=int(count))
                excess = rng.exponential(scale=1.0 / eta, size=int(count))
                jumps[path_idx] = float(np.sum(signs * (delta_jump + excess)))
        paths[:, step] = previous + drift + diffusion + jumps
    return paths


def fit_model(
    model: str,
    spread: pd.Series | np.ndarray,
    gaussian_fit: dict[str, Any] | None = None,
    seed: int = 123,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit one named model to a spread and return scalar-friendly metadata."""

    del seed
    model_key = str(model).lower()
    x = _finite_spread(spread)
    if model_key not in MODEL_CHOICES:
        return {"valid": False, "model": model_key, "reason": f"unknown model {model!r}"}
    if len(x) < 10:
        return {"valid": False, "model": model_key, "reason": "need at least 10 finite observations"}

    started = perf_counter()
    try:
        if model_key == "gaussian":
            fit = fit_brownian_ou_from_spread(x)
        else:
            gfit = gaussian_fit if gaussian_fit is not None else fit_brownian_ou_from_spread(x)
            u_form = float(gfit.get("mu", np.mean(x)))
            if model_key == "cp":
                fit = estimate_cp_ou_fixed_mean(x, u_form=u_form, minutes_per_day=int(kwargs.get("minutes_per_day", 390)))
            elif model_key == "nig":
                fit = estimate_nig_ou_fixed_mean_fft_multistart(
                    x,
                    u_form=u_form,
                    fft_grid_size=int(kwargs.get("fft_grid_size", 512)),
                    truncation_l=float(kwargs.get("truncation_l", 6.0)),
                    maxiter=int(kwargs.get("maxiter", 5)),
                    max_optimizer_starts=int(kwargs.get("max_optimizer_starts", 1)),
                    max_candidate_starts_to_score=int(kwargs.get("max_candidate_starts_to_score", 2)),
                    fft_intervals=tuple(kwargs.get("fft_intervals", ("sample",))),
                    include_stationary_density=bool(kwargs.get("include_stationary_density", False)),
                )
            elif model_key == "symmetric_bg":
                bg_fit = estimate_symmetric_bg_ou_wu_innovation_moments(
                    x,
                    dt=float(kwargs.get("dt", DT_MINUTE)),
                )
                fit = {
                    "model": "symmetric_bg",
                    "estimation_method": "symmetric_bg_ou_wu_innovation_moments",
                    "mean_method": "ar1_ols_long_run_mean",
                    "u_form_source": "symmetric_bg_ar1_ols",
                    "gaussian_u_form": float(u_form),
                    **bg_fit.to_dict(),
                    "theta": bg_fit.kappa,
                    "stationary_sigma": float(np.sqrt(bg_fit.stationary_variance)),
                }
            else:
                fit = estimate_cgmy_ou_fft_mle(
                    x,
                    dt=float(kwargs.get("dt", DT_MINUTE)),
                    fft_grid_size=int(kwargs.get("fft_grid_size", 512)),
                    truncation_l=float(kwargs.get("truncation_l", 6.0)),
                    maxiter=int(kwargs.get("maxiter", 5)),
                    max_optimizer_starts=int(kwargs.get("max_optimizer_starts", 1)),
                    max_candidate_starts_to_score=int(kwargs.get("max_candidate_starts_to_score", 2)),
                    fft_intervals=tuple(kwargs.get("fft_intervals", ("sample",))),
                    include_stationary_density=bool(kwargs.get("include_stationary_density", False)),
                    include_ecf_start=bool(kwargs.get("include_ecf_start", True)),
                    lambda_max_lag=int(kwargs.get("lambda_max_lag", 20)),
                )
    except Exception as exc:
        fit = {"valid": False, "reason": repr(exc)}

    return {
        "model": model_key,
        "fit_elapsed_seconds": float(perf_counter() - started),
        **fit,
    }


def fit_models_for_spread(
    spread: pd.Series | np.ndarray,
    models: list[str] | tuple[str, ...] = MODEL_CHOICES,
    seed: int = 123,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Fit several models to one spread, sharing the Gaussian formation mean."""

    x = _finite_spread(spread)
    gaussian_fit = fit_brownian_ou_from_spread(x)
    rows: list[dict[str, Any]] = []
    for model in models:
        fit = fit_model(model, x, gaussian_fit=gaussian_fit, seed=seed, **kwargs)
        rows.append(fit)
    return rows


def model_scale(fit: dict[str, Any], fallback_spread: np.ndarray | None = None) -> float:
    """Return the stationary spread scale to use for threshold grids."""

    for key in ("sigma_eq", "stationary_sigma", "nig_stationary_std_residual", "residual_stationary_std"):
        value = fit.get(key)
        if value is not None and np.isfinite(float(value)) and float(value) > 0:
            return float(value)
    if str(fit.get("model", "")).lower() == "cgmy":
        try:
            cumulants = stationary_cumulants_zero_mean(
                C=float(fit["C"]),
                G=float(fit["G"]),
                M=float(fit["M"]),
                Y=float(fit["Y"]),
            )
            std = float(np.sqrt(cumulants["c2"]))
            if np.isfinite(std) and std > 0.0:
                return std
        except Exception:
            pass
    if fallback_spread is not None:
        std = float(np.nanstd(np.asarray(fallback_spread, dtype=float), ddof=1))
        if np.isfinite(std) and std > 0:
            return std
    return 1e-4


def simulate_paths_from_fit(
    fit: dict[str, Any],
    x0: float,
    n_paths: int,
    n_steps: int,
    seed: int = 123,
    dt: float = DT_MINUTE,
    sim_fft_grid_size: int = DEFAULT_SIM_FFT_GRID_SIZE,
    nig_du: float = 20.0,
    cgmy_du: float = 0.5,
) -> np.ndarray:
    """Simulate paths from a fitted model for threshold optimisation."""

    model = str(fit.get("model", "")).lower()
    n_observations = int(n_steps)
    if n_observations <= 0:
        raise ValueError("Need n_steps > 0.")
    transition_steps = max(n_observations - 1, 1)
    if model == "gaussian":
        return _simulate_gaussian_ou(
            theta=float(fit["theta"]),
            mu=float(fit["mu"]),
            sigma=float(fit["sigma"]),
            x0=float(x0),
            n_paths=n_paths,
            n_steps=n_observations,
            dt=1.0,
            seed=seed,
        )
    if model == "cp":
        return _simulate_cp_ou(
            theta=float(fit.get("theta", fit.get("kappa", 1.0))),
            mu=float(fit.get("u_form", fit.get("mu", 0.0))),
            sigma=float(fit.get("sigma", model_scale(fit))),
            lambda_jump=float(fit.get("lambda_jump", 0.0)),
            eta=float(fit.get("eta", np.nan)),
            delta_jump=float(fit.get("delta", 0.0)),
            x0=float(x0),
            n_paths=n_paths,
            n_steps=n_observations,
            dt=float(fit.get("dt", dt)),
            seed=seed,
        )
    if model == "nig":
        simulator, simulator_diagnostics = build_nig_ou_simulator_with_fallback(
            alpha=float(fit["alpha"]),
            beta=float(fit["beta"]),
            mu=float(fit["internal_location_ell"]),
            delta=float(fit["delta"]),
            lam=float(fit["lambda_ou"]),
            dt=dt,
            seed=seed,
            shifted_attempts=(
                {"n_fft": int(sim_fft_grid_size), "shift_fraction": 0.95},
                {"n_fft": max(int(sim_fft_grid_size) * 2, 2**16), "shift_fraction": 0.90},
            ),
            density_n_fft=max(int(sim_fft_grid_size), 2**16),
            density_du=float(nig_du),
            right_tail_rate=str(fit.get("right_tail_rate", "survival_match")),
        )
        residual_x0 = float(x0) - float(fit.get("u_form", 0.0))
        paths = simulator.simulate_paths(n_paths=int(n_paths), n_steps=transition_steps, x0=residual_x0)
        simulate_paths_from_fit.last_simulator_diagnostics = simulator_diagnostics
        return paths[:, :n_observations] + float(fit.get("u_form", 0.0))
    if model == "cgmy":
        simulator, simulator_diagnostics = build_cgmy_ou_simulator_with_fallback(
            C=float(fit["C"]),
            G=float(fit["G"]),
            M=float(fit["M"]),
            Y=float(fit["Y"]),
            long_run_mean=float(fit.get("gaussian_mean", fit.get("long_run_mean", fit.get("mu")))),
            lam=float(fit.get("lambda_ou", fit.get("lambda"))),
            dt=dt,
            seed=seed,
            shifted_attempts=(
                {"n_fft": int(sim_fft_grid_size), "shift_fraction": 0.95},
                {"n_fft": max(int(sim_fft_grid_size) * 2, 2**16), "shift_fraction": 0.90},
            ),
            density_n_fft=max(int(sim_fft_grid_size), 2**16),
            density_du=float(cgmy_du),
        )
        paths = simulator.simulate_paths(n_paths=int(n_paths), n_steps=transition_steps, x0=float(x0))
        simulate_paths_from_fit.last_simulator_diagnostics = simulator_diagnostics
        return paths[:, :n_observations]
    if model == "symmetric_bg":
        simulator = SymmetricBGOU(
            mu=float(fit["mu"]),
            alpha=float(fit["alpha"]),
            beta=float(fit["beta"]),
            kappa=float(fit["kappa"]),
            dt=float(fit.get("dt", dt)),
            seed=seed,
        )
        paths = np.zeros((int(n_paths), n_observations), dtype=float)
        for path_idx in range(int(n_paths)):
            paths[path_idx] = simulator.simulate(
                n_observations,
                x0=float(x0),
                stationary_start=False,
                method=str(fit.get("simulation_method", "compound_poisson")),
            )
        simulate_paths_from_fit.last_simulator_diagnostics = {
            "simulation_method": str(fit.get("simulation_method", "compound_poisson")),
            "model": "symmetric_bg",
        }
        return paths
    raise ValueError(f"Cannot simulate unknown model {model!r}.")


def model_result_row(fit: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    """Return one compact estimator row."""

    return {**metadata, **scalarize_mapping(fit)}


__all__ = [
    "DT_MINUTE",
    "MODEL_CHOICES",
    "fit_model",
    "fit_models_for_spread",
    "model_result_row",
    "model_scale",
    "simulate_paths_from_fit",
]
