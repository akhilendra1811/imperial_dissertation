"""Monte Carlo study for the fixed-mean NIG-OU estimator.

The study deliberately reuses the maintained NIG-OU simulator and estimator in
``levy_ou.estimators.nig_ou`` / ``levy_ou.simulation.nig_simulator``.  The only
new estimator logic here is the diagnostic Wu-only estimator and the paired
Wu-single-start likelihood refinement used for comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from levy_ou.config import DT_MINUTE
from levy_ou.estimators.gaussian_ou import fit_brownian_ou_from_spread
from levy_ou.estimators.nig_ou import (
    _empirical_cumulants,
    _nig_gamma_raw_to_params,
    _nig_ou_innovations,
    _nig_ou_lambda_start_multilag,
    _nig_params_to_gamma_raw,
    _nig_raw_in_bounds,
    _nig_stationary_moment_match_start,
    centered_stationary_density,
    centered_zstar_cf,
    centered_zstar_density_fft,
    estimate_nig_ou_fixed_mean_fft_multistart,
)
from levy_ou.simulation.nig_simulator import build_nig_ou_simulator_with_fallback

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path("outputs/nig_simulation_study")
FORMATION_OBSERVATIONS = 30 * 390
DEFAULT_LAMBDA_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
DEFAULT_FFT_INTERVALS = ("cumulant", "sample")
FINAL_GAUSSIAN_NIG_BACKTESTS = (
    ("energy", 2008, "energy_2008/backtests/energy_2008_gaussian_top10_nig_fixed_mean_backtest_npaths1000_gamma0_025_05_1_2_4_grid025_30_process8"),
    ("energy", 2024, "energy_2024/backtests/energy_2024_gaussian_top10_nig_fixed_mean_backtest_npaths1000_gamma0_025_05_1_2_4_grid025_30_process8"),
    ("communication", 2024, "communication_2024/backtests/communication_2024_gaussian_top10_nig_fixed_mean_backtest_npaths1000_gamma0_025_05_1_2_4_grid025_30_process8"),
)
FINAL_ADF_NIG_BACKTESTS = (
    ("energy", 2008, "energy_2008/adf_capped10/backtests/energy_2008_adf_capped10_nig_backtest_npaths1000_gamma0_025_05_1_2_4_grid025_30_process8"),
    ("energy", 2024, "energy_2024/adf_capped10/backtests/energy_2024_adf_capped10_nig_backtest_npaths1000_gamma0_025_05_1_2_4_grid025_30_process8"),
    ("communication", 2024, "communication_2024/adf_capped10/backtests/communication_2024_adf_capped10_nig_backtest_npaths1000_gamma0_025_05_1_2_4_grid025_30_process8"),
)


@dataclass(frozen=True)
class EstimationSettings:
    dt: float = DT_MINUTE
    fft_grid_size: int = 8192
    truncation_l: float = 10.0
    maxiter: int = 220
    optimizer_method: str = "Powell"
    density_floor: float = 1e-300
    lambda_factors: tuple[float, ...] = DEFAULT_LAMBDA_FACTORS
    lambda_max_lag: int = 20
    max_optimizer_starts: int = 3
    max_candidate_starts_to_score: int | None = None
    fft_intervals: tuple[str, ...] = DEFAULT_FFT_INTERVALS
    include_stationary_density: bool = True
    wu_first: bool = True

    def kwargs(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "fft_grid_size": self.fft_grid_size,
            "truncation_l": self.truncation_l,
            "maxiter": self.maxiter,
            "optimizer_method": self.optimizer_method,
            "density_floor": self.density_floor,
            "lambda_factors": self.lambda_factors,
            "lambda_max_lag": self.lambda_max_lag,
            "max_optimizer_starts": self.max_optimizer_starts,
            "max_candidate_starts_to_score": self.max_candidate_starts_to_score,
            "fft_intervals": self.fft_intervals,
            "include_stationary_density": self.include_stationary_density,
            "wu_first": self.wu_first,
        }


@dataclass(frozen=True)
class SimulationSettings:
    sim_fft_grid_size: int = 32768
    first_shift_fraction: float = 0.95
    second_shift_fraction: float = 0.90
    density_du: float = 20.0
    right_tail_rate: str = "survival_match"
    density_negative_mass_tol: float = 1e-4
    density_raw_area_tol: float = 1e-3

    @property
    def shifted_attempts(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"n_fft": int(self.sim_fft_grid_size), "shift_fraction": float(self.first_shift_fraction)},
            {"n_fft": max(int(self.sim_fft_grid_size) * 2, 2**16), "shift_fraction": float(self.second_shift_fraction)},
        )

    @property
    def density_n_fft(self) -> int:
        return max(int(self.sim_fft_grid_size), 2**16)


@dataclass(frozen=True)
class Calibration:
    case_id: int
    role: str
    sector: str
    year: int
    source_backtest: str
    source_file: str
    pair: str
    window: str
    u: float
    lambda_ou: float
    alpha: float
    beta: float
    delta: float
    settings: EstimationSettings


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value).strip().lower() in {"true", "1", "yes"}


def derived_nig(alpha: float, beta: float, delta: float, lambda_ou: float, dt: float) -> dict[str, float]:
    gamma = math.sqrt(max(alpha * alpha - beta * beta, np.nan))
    var = delta * alpha * alpha / (gamma**3)
    std = math.sqrt(var) if var >= 0 else np.nan
    skew = 3.0 * beta / (alpha * math.sqrt(delta * gamma)) if delta > 0 and gamma > 0 else np.nan
    ex_kurt = 3.0 * (alpha * alpha + 4.0 * beta * beta) / (delta * alpha * alpha * gamma) if delta > 0 and gamma > 0 else np.nan
    hl_model = math.log(2.0) / lambda_ou if lambda_ou > 0 else np.nan
    return {
        "gamma": gamma,
        "internal_location_ell": -delta * beta / gamma if gamma > 0 else np.nan,
        "rho_beta": beta / alpha if alpha > 0 else np.nan,
        "stationary_variance": var,
        "stationary_std": std,
        "stationary_skewness": skew,
        "stationary_excess_kurtosis": ex_kurt,
        "half_life_model_time": hl_model,
        "half_life_minutes": hl_model / dt if dt > 0 else np.nan,
    }


def innovation_cumulants(alpha: float, beta: float, delta: float, lambda_ou: float, dt: float) -> dict[str, float]:
    d = derived_nig(alpha, beta, delta, lambda_ou, dt)
    k2 = d["stationary_variance"]
    std = d["stationary_std"]
    k3 = d["stationary_skewness"] * (std**3) if std > 0 else np.nan
    k4 = d["stationary_excess_kurtosis"] * (k2**2) if k2 > 0 else np.nan
    out = {}
    for r, kx in [(1, 0.0), (2, k2), (3, k3), (4, k4)]:
        out[f"innov_c{r}"] = math.expm1(r * lambda_ou * dt) * kx
    return out


def settings_from_row(row: pd.Series) -> EstimationSettings:
    return EstimationSettings(
        dt=safe_float(row.get("dt"), DT_MINUTE),
        fft_grid_size=int(safe_float(row.get("fft_grid_size"), 8192)),
        truncation_l=safe_float(row.get("truncation_l"), 10.0),
        maxiter=220,
        optimizer_method=str(row.get("optimizer_method", "Powell") or "Powell"),
        density_floor=safe_float(row.get("density_floor"), 1e-300),
        lambda_factors=DEFAULT_LAMBDA_FACTORS,
        lambda_max_lag=20,
        max_optimizer_starts=int(safe_float(row.get("optimizer_starts"), 3)),
        max_candidate_starts_to_score=int(safe_float(row.get("candidate_starts"), 0)) or None,
        fft_intervals=DEFAULT_FFT_INTERVALS,
        include_stationary_density=parse_bool(row.get("include_stationary_density"), True),
        wu_first=True,
    )


def final_estimate_files(repo_root: Path) -> list[tuple[str, int, str, Path]]:
    files: list[tuple[str, int, str, Path]] = []
    for sector, year, rel in FINAL_GAUSSIAN_NIG_BACKTESTS + FINAL_ADF_NIG_BACKTESTS:
        path = repo_root / "output_NEW" / rel / "estimates.csv"
        if path.exists():
            files.append((sector, year, rel, path))
    return files


def load_candidate_pool(repo_root: Path) -> pd.DataFrame:
    frames = []
    for sector, year, source_backtest, path in final_estimate_files(repo_root):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["sector"] = sector
        df["year"] = year
        df["source_backtest"] = source_backtest
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No final NIG fixed-mean estimates.csv files found under output_NEW.")
    pool = pd.concat(frames, ignore_index=True)
    for col in ["alpha", "beta", "delta", "lambda_ou"]:
        pool[col] = pd.to_numeric(pool[col], errors="coerce")
    if "u_form" not in pool.columns:
        pool["u_form"] = pd.to_numeric(pool.get("mu_form", np.nan), errors="coerce")
    pool["valid_for_study"] = (
        np.isfinite(pool["alpha"]) & np.isfinite(pool["beta"]) & np.isfinite(pool["delta"]) & np.isfinite(pool["lambda_ou"])
        & (pool["alpha"] > 0) & (pool["delta"] > 0) & (pool["lambda_ou"] > 0) & (pool["beta"].abs() < pool["alpha"])
        & np.isfinite(pd.to_numeric(pool["u_form"], errors="coerce"))
    )
    rows = []
    for _, row in pool[pool["valid_for_study"]].iterrows():
        d = derived_nig(float(row.alpha), float(row.beta), float(row.delta), float(row.lambda_ou), safe_float(row.get("dt"), DT_MINUTE))
        rows.append(d)
    derived = pd.DataFrame(rows, index=pool[pool["valid_for_study"]].index)
    for col in derived.columns:
        pool[col] = derived[col]
    return pool


def _pair_label(row: pd.Series) -> str:
    if "pair" in row and pd.notna(row.get("pair")):
        return str(row.get("pair"))
    ticker_a = str(row.get("ticker_a", ""))
    ticker_b = str(row.get("ticker_b", ""))
    if ticker_a or ticker_b:
        return f"{ticker_a}-{ticker_b}"
    return str(row.get("pair_id", row.get("pair_index", "")))


def _window_label(row: pd.Series) -> str:
    if "window" in row and pd.notna(row.get("window")):
        return str(row.get("window"))
    if "window_id" in row and pd.notna(row.get("window_id")):
        return str(row.get("window_id"))
    start = str(row.get("formation_start", ""))
    end = str(row.get("formation_end", ""))
    return f"{start}:{end}" if start or end else ""


def _identity(row: pd.Series) -> tuple[str, str, str]:
    return str(row.get("source_backtest", "")), _pair_label(row), _window_label(row)


def select_calibrations(pool: pd.DataFrame) -> pd.DataFrame:
    df_all = pool[pool["valid_for_study"]].copy()
    if df_all.empty:
        raise ValueError("No valid NIG calibration rows found.")
    # Avoid numerical boundary fits dominating the representative cases. These
    # are still valid empirical rows, but the study cases should be interpretable
    # calibrations rather than near-degenerate optimizer-boundary solutions.
    df = df_all[(df_all["alpha"] > 1e-2) & (df_all["stationary_excess_kurtosis"] < 5_000)].copy()
    if len(df) < 4:
        df = df_all.copy()
    selected_indices: list[int] = []
    used: set[tuple[str, str, str]] = set()
    used_dataset: set[tuple[str, int]] = set()
    med_lambda = float(df["lambda_ou"].median())
    med_kurt = float(df["stationary_excess_kurtosis"].median())
    scale_lambda = float(df["lambda_ou"].quantile(0.75) - df["lambda_ou"].quantile(0.25)) or 1.0
    scale_kurt = float(df["stationary_excess_kurtosis"].quantile(0.75) - df["stationary_excess_kurtosis"].quantile(0.25)) or 1.0

    def take(role: str, candidates: pd.DataFrame, score_col: str | None = None, ascending: bool = True) -> None:
        nonlocal selected_indices
        if score_col is not None:
            candidates = candidates.sort_values(score_col, ascending=ascending)
        passes = [candidates[candidates.apply(lambda r: (str(r.get("sector", "")), int(r.get("year", 0))) not in used_dataset, axis=1)], candidates]
        for block in passes:
            for idx, row in block.iterrows():
                ident = _identity(row)
                dataset = (str(row.get("sector", "")), int(row.get("year", 0)))
                if ident not in used and idx not in selected_indices:
                    selected_indices.append(idx)
                    used.add(ident)
                    used_dataset.add(dataset)
                    df.loc[idx, "selection_reason"] = role
                    return
        idx = candidates.index[0]
        row = candidates.loc[idx]
        selected_indices.append(idx)
        used_dataset.add((str(row.get("sector", "")), int(row.get("year", 0))))
        df.loc[idx, "selection_reason"] = role

    df["moderate_score"] = (
        df["rho_beta"].abs()
        + (df["lambda_ou"] - med_lambda).abs() / scale_lambda
        + (df["stationary_excess_kurtosis"] - med_kurt).abs() / scale_kurt
    )
    take("near-symmetric / moderate: small |beta/alpha| with central lambda and kurtosis", df, "moderate_score", True)
    pos_target = float(df.loc[df["stationary_skewness"] > 0, "stationary_skewness"].quantile(0.95))
    neg_target = float(df.loc[df["stationary_skewness"] < 0, "stationary_skewness"].quantile(0.05))
    df["positive_skew_score"] = (df["stationary_skewness"] - pos_target).abs()
    df["negative_skew_score"] = (df["stationary_skewness"] - neg_target).abs()
    take("positively skewed: closest to the positive 95th-percentile stationary skewness after excluding boundary fits", df[df["stationary_skewness"] > 0], "positive_skew_score", True)
    take("negatively skewed: closest to the negative 5th-percentile stationary skewness after excluding boundary fits", df[df["stationary_skewness"] < 0], "negative_skew_score", True)
    df["tail_score"] = df["stationary_excess_kurtosis"].rank(pct=True) + (df["lambda_ou"] - med_lambda).abs().rank(pct=True)
    take("heavy-tail / extreme: high excess kurtosis with different mean reversion", df, "tail_score", False)
    out = df.loc[selected_indices].copy().reset_index(drop=False).rename(columns={"index": "source_row_index"})
    out.insert(0, "case_id", np.arange(1, len(out) + 1))
    out.insert(1, "calibration_role", ["near_symmetric_moderate", "positive_skew", "negative_skew", "heavy_tail_extreme"][: len(out)])
    return out


def calibration_from_row(row: pd.Series) -> Calibration:
    pair = _pair_label(row)
    window = _window_label(row)
    return Calibration(
        case_id=int(row["case_id"]),
        role=str(row["calibration_role"]),
        sector=str(row["sector"]),
        year=int(row["year"]),
        source_backtest=str(row["source_backtest"]),
        source_file=str(row["source_file"]),
        pair=pair,
        window=window,
        u=safe_float(row.get("u_form", row.get("mu_form"))),
        lambda_ou=float(row["lambda_ou"]),
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        delta=float(row["delta"]),
        settings=settings_from_row(row),
    )


def write_settings_csv(output_dir: Path, selected: pd.DataFrame, sim_settings: SimulationSettings) -> None:
    rows: list[dict[str, Any]] = []
    settings_source = "output_NEW final NIG fixed-mean estimates.csv plus output_NEW/run_all_025_30.sh"
    base = {
        "formation_observations": FORMATION_OBSERVATIONS,
        "lambda_factors": DEFAULT_LAMBDA_FACTORS,
        "lambda_max_lag": 20,
        "maxiter": 220,
        "optimizer_tolerances": "Powell xtol=1e-4, ftol=1e-4",
        "optimizer_method": "Powell",
        "fft_intervals": DEFAULT_FFT_INTERVALS,
        "wu_first": True,
        "parameter_transform": "raw=(log_gamma, atanh(beta/alpha), log_delta, log_lambda); alpha=gamma/sqrt(1-rho^2), beta=rho*alpha",
        "parameter_constraints": "alpha>0, |beta|<alpha, delta>0, lambda>0, lambda*dt<20; raw bounds [-20,20], rho raw [-8,8], log_delta [-30,20], log_lambda [-20,20]",
        "simulation_shifted_attempt_1": f"n_fft={sim_settings.sim_fft_grid_size}, shift_fraction={sim_settings.first_shift_fraction}",
        "simulation_shifted_attempt_2": f"n_fft={max(sim_settings.sim_fft_grid_size * 2, 2**16)}, shift_fraction={sim_settings.second_shift_fraction}",
        "density_fft_fallback": f"n_fft={sim_settings.density_n_fft}, du={sim_settings.density_du}, right_tail_rate={sim_settings.right_tail_rate}",
        "random_seed": "seed_base + case_id * 1_000_000 + replication_id",
    }
    for setting, value in base.items():
        rows.append({"setting": setting, "value": value, "source_file": settings_source, "source_field": setting, "notes": "production behaviour mirrored unless --smoke-test is used"})
    for _, row in selected.iterrows():
        for setting in ["dt", "fft_grid_size", "truncation_l", "density_floor", "optimizer_method", "candidate_starts", "generated_candidate_starts", "scored_starts", "optimizer_starts", "include_stationary_density", "best_fft_interval"]:
            rows.append({"setting": f"case_{int(row.case_id)}.{setting}", "value": row.get(setting, ""), "source_file": row["source_file"], "source_field": setting, "notes": row["source_backtest"]})
    pd.DataFrame(rows).to_csv(output_dir / "study_settings.csv", index=False)


def seed_for(seed_base: int, case_id: int, replication_id: int) -> int:
    return int(seed_base) + int(case_id) * 1_000_000 + int(replication_id)


def build_simulator(cal: Calibration, seed: int, sim_settings: SimulationSettings):
    d = derived_nig(cal.alpha, cal.beta, cal.delta, cal.lambda_ou, cal.settings.dt)
    return build_nig_ou_simulator_with_fallback(
        alpha=cal.alpha,
        beta=cal.beta,
        mu=d["internal_location_ell"],
        delta=cal.delta,
        lam=cal.lambda_ou,
        dt=cal.settings.dt,
        seed=seed,
        shifted_attempts=sim_settings.shifted_attempts,
        density_n_fft=sim_settings.density_n_fft,
        density_du=sim_settings.density_du,
        right_tail_rate=sim_settings.right_tail_rate,
        density_negative_mass_tol=sim_settings.density_negative_mass_tol,
        density_raw_area_tol=sim_settings.density_raw_area_tol,
    )


def simulate_spread(cal: Calibration, seed: int, n_observations: int, sim_settings: SimulationSettings) -> tuple[np.ndarray, dict[str, Any]]:
    sim, diag = build_simulator(cal, seed, sim_settings)
    residual = np.asarray(sim.simulate(n=int(n_observations), stationary_start=True), dtype=float)
    return cal.u + residual, diag


def gaussian_mean_once(spread: np.ndarray) -> tuple[float, dict[str, Any]]:
    fit = fit_brownian_ou_from_spread(pd.Series(spread))
    u_hat = safe_float(fit.get("mu"), float(np.mean(spread)))
    return u_hat, fit


def wu_only_estimator(x: np.ndarray, settings: EstimationSettings) -> dict[str, Any]:
    started = perf_counter()
    try:
        lambda0, acf1 = _nig_ou_lambda_start_multilag(x, dt=settings.dt, max_lag=settings.lambda_max_lag)
        y = _nig_ou_innovations(x, lambda_ou=lambda0, dt=settings.dt)
        cy = _empirical_cumulants(y)
        stationary = {f"c{r}": float(cy[f"c{r}"]) / math.expm1(r * lambda0 * settings.dt) for r in (1, 2, 3, 4)}
        match = _nig_stationary_moment_match_start(stationary, method="wu_only_stationary_innovation_cumulant_match")
        if match is None or not bool(match.get("valid", False)):
            raise ValueError(str((match or {}).get("reason", "Wu cumulant match failed")))
        out = {
            "success": True,
            "estimator": "wu_only",
            "lambda_ou": float(lambda0),
            "alpha": float(match["alpha0"]),
            "beta": float(match["beta0"]),
            "delta": float(match["delta0"]),
            "acf1": float(acf1),
            "runtime_seconds": perf_counter() - started,
            "failure_reason": "",
            "wu_match_objective": safe_float(match.get("moment_objective")),
        }
        for r in (1, 2, 3, 4):
            out[f"wu_empirical_innov_c{r}"] = float(cy[f"c{r}"])
            out[f"wu_implied_stationary_c{r}"] = float(stationary[f"c{r}"])
        return out
    except Exception as exc:
        return {"success": False, "estimator": "wu_only", "runtime_seconds": perf_counter() - started, "failure_reason": f"wu_only: {type(exc).__name__}: {exc}"}


def transition_loglik(x: np.ndarray, params: dict[str, float], settings: EstimationSettings) -> tuple[float, str]:
    best = (-np.inf, "")
    for interval in settings.fft_intervals:
        try:
            lam = float(params["lambda_ou"])
            y = _nig_ou_innovations(x, lambda_ou=lam, dt=settings.dt)
            dens, _ = centered_zstar_density_fft(
                y,
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
                delta_nig=float(params["delta"]),
                lambda_ou=lam,
                dt=settings.dt,
                fft_grid_size=settings.fft_grid_size,
                truncation_l=settings.truncation_l,
                density_floor=settings.density_floor,
                fft_interval=str(interval),
            )
            ll = len(y) * lam * settings.dt + float(np.sum(np.log(dens)))
            if settings.include_stationary_density:
                sd = centered_stationary_density(np.asarray([x[0]], dtype=float), float(params["alpha"]), float(params["beta"]), float(params["delta"]), settings.density_floor)
                ll += float(np.log(sd[0]))
            if np.isfinite(ll) and ll > best[0]:
                best = (ll, str(interval))
        except Exception:
            continue
    return best


def wu_single_start_mle(x: np.ndarray, wu: dict[str, Any], settings: EstimationSettings) -> dict[str, Any]:
    started = perf_counter()
    if not wu.get("success"):
        return {"success": False, "estimator": "wu_single_start_mle", "runtime_seconds": 0.0, "failure_reason": "Wu start unavailable"}
    try:
        raw0 = _nig_params_to_gamma_raw(float(wu["alpha"]), float(wu["beta"]), float(wu["delta"]), float(wu["lambda_ou"]))
        start_params = {"alpha": float(wu["alpha"]), "beta": float(wu["beta"]), "delta": float(wu["delta"]), "lambda_ou": float(wu["lambda_ou"])}
        start_ll, start_interval = transition_loglik(x, start_params, settings)

        def objective(raw: np.ndarray, interval: str) -> float:
            if not _nig_raw_in_bounds(raw):
                return 1e100
            alpha, beta, delta, lam = _nig_gamma_raw_to_params(raw)
            if not (alpha > 0 and abs(beta) < alpha and delta > 0 and lam > 0 and lam * settings.dt < 20):
                return 1e100
            ll, _ = transition_loglik(x, {"alpha": alpha, "beta": beta, "delta": delta, "lambda_ou": lam}, EstimationSettings(**{**settings.__dict__, "fft_intervals": (interval,)}))
            return -ll if np.isfinite(ll) else 1e100

        options: dict[str, Any] = {"maxiter": int(settings.maxiter)}
        if settings.optimizer_method == "Powell":
            options.update({"xtol": 1e-4, "ftol": 1e-4})
        elif settings.optimizer_method == "Nelder-Mead":
            options.update({"xatol": 1e-4, "fatol": 1e-4})
        best: tuple[float, Any, str] | None = None
        for interval in settings.fft_intervals:
            result = minimize(lambda values, interval=interval: objective(values, str(interval)), raw0, method=settings.optimizer_method, options=options)
            value = float(result.fun) if np.isfinite(result.fun) else 1e100
            if best is None or value < best[0]:
                best = (value, result, str(interval))
        if best is None:
            raise RuntimeError("no optimizer result")
        _, result, interval = best
        alpha, beta, delta, lam = _nig_gamma_raw_to_params(result.x)
        final_ll, final_interval = transition_loglik(x, {"alpha": alpha, "beta": beta, "delta": delta, "lambda_ou": lam}, settings)
        return {
            "success": bool(result.success and np.isfinite(final_ll)),
            "estimator": "wu_single_start_mle",
            "alpha": float(alpha), "beta": float(beta), "delta": float(delta), "lambda_ou": float(lam),
            "runtime_seconds": perf_counter() - started,
            "failure_reason": "" if result.success else str(result.message),
            "wu_start_alpha": start_params["alpha"], "wu_start_beta": start_params["beta"], "wu_start_delta": start_params["delta"], "wu_start_lambda_ou": start_params["lambda_ou"],
            "start_loglik": float(start_ll), "start_fft_interval": start_interval,
            "estimated_loglik": float(final_ll), "best_fft_interval": final_interval,
            "loglik_improvement_from_start": float(final_ll - start_ll),
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
            "optimizer_iterations": int(getattr(result, "nit", -1)), "optimizer_evaluations": int(getattr(result, "nfev", -1)),
            "optimizer_starts": int(len(settings.fft_intervals)),
        }
    except Exception as exc:
        return {"success": False, "estimator": "wu_single_start_mle", "runtime_seconds": perf_counter() - started, "failure_reason": f"wu_single_start_mle: {type(exc).__name__}: {exc}"}


def full_mle_estimator(spread: np.ndarray, u_hat: float, settings: EstimationSettings) -> dict[str, Any]:
    started = perf_counter()
    try:
        fit = estimate_nig_ou_fixed_mean_fft_multistart(pd.Series(spread), u_form=float(u_hat), **settings.kwargs())
        out = {"estimator": "full_mle", "success": bool(fit.get("valid", False)), "runtime_seconds": perf_counter() - started, "failure_reason": str(fit.get("reason") or "")}
        out.update({k: v for k, v in fit.items() if isinstance(v, (str, int, float, bool, np.integer, np.floating)) or v is None})
        return out
    except Exception as exc:
        return {"success": False, "estimator": "full_mle", "runtime_seconds": perf_counter() - started, "failure_reason": f"full_mle: {type(exc).__name__}: {exc}"}


def cf_metrics(true_params: dict[str, float], est_params: dict[str, Any], settings: EstimationSettings, v_grid: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    true_ic = innovation_cumulants(true_params["alpha"], true_params["beta"], true_params["delta"], true_params["lambda_ou"], settings.dt)
    sigma_y = math.sqrt(max(true_ic["innov_c2"], np.finfo(float).tiny))
    u_grid = v_grid / sigma_y
    true_phi = centered_zstar_cf(u_grid, true_params["alpha"], true_params["beta"], true_params["delta"], true_params["lambda_ou"], settings.dt)
    est_phi = centered_zstar_cf(u_grid, float(est_params["alpha"]), float(est_params["beta"]), float(est_params["delta"]), float(est_params["lambda_ou"]), settings.dt)
    diff = est_phi - true_phi
    absdiff = np.abs(diff)
    metrics = {
        "cf_mse": float(np.mean(absdiff**2)),
        "cf_mae": float(np.mean(absdiff)),
        "cf_max_abs_error": float(np.max(absdiff)),
        "cf_real_rmse": float(np.sqrt(np.mean(np.real(diff) ** 2))),
        "cf_imag_rmse": float(np.sqrt(np.mean(np.imag(diff) ** 2))),
    }
    rows = pd.DataFrame({
        "standardized_frequency_v": v_grid,
        "actual_frequency_u": u_grid,
        "true_cf_real": np.real(true_phi),
        "true_cf_imag": np.imag(true_phi),
        "fitted_cf_real": np.real(est_phi),
        "fitted_cf_imag": np.imag(est_phi),
        "absolute_complex_error": absdiff,
    })
    return metrics, rows


def decorate_result(base: dict[str, Any], cal: Calibration, u_hat: float, x_est: np.ndarray, true_ll: float, true_interval: str, v_grid: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    row = {
        "case_id": cal.case_id, "calibration_role": cal.role, "sector": cal.sector, "year": cal.year,
        "source_backtest": cal.source_backtest, "source_file": cal.source_file, "pair": cal.pair, "window": cal.window,
        "true_u": cal.u, "estimated_u": u_hat, "mean_error": u_hat - cal.u,
        "true_lambda_ou": cal.lambda_ou, "true_alpha": cal.alpha, "true_beta": cal.beta, "true_delta": cal.delta,
        "true_loglik": true_ll, "true_loglik_interval": true_interval,
    }
    row.update(base)
    cf_long = pd.DataFrame()
    if base.get("success") and all(k in base for k in ("alpha", "beta", "delta", "lambda_ou")):
        true_params = {"alpha": cal.alpha, "beta": cal.beta, "delta": cal.delta, "lambda_ou": cal.lambda_ou}
        est_params = {"alpha": float(base["alpha"]), "beta": float(base["beta"]), "delta": float(base["delta"]), "lambda_ou": float(base["lambda_ou"])}
        est_ll, interval = transition_loglik(x_est, est_params, cal.settings)
        row["estimated_loglik"] = est_ll
        row["estimated_loglik_interval"] = interval
        row["loglik_minus_true"] = est_ll - true_ll
        for name in ("lambda_ou", "alpha", "beta", "delta"):
            err = est_params[name] - true_params[name]
            row[f"{name}_error"] = err
            row[f"{name}_abs_error"] = abs(err)
            row[f"{name}_squared_error"] = err * err
            denom = abs(true_params[name])
            row[f"{name}_relative_error"] = err / denom if denom > 1e-8 else np.nan
        td = derived_nig(cal.alpha, cal.beta, cal.delta, cal.lambda_ou, cal.settings.dt)
        ed = derived_nig(est_params["alpha"], est_params["beta"], est_params["delta"], est_params["lambda_ou"], cal.settings.dt)
        for k, tv in td.items():
            ev = ed[k]
            row[f"true_{k}"] = tv
            row[f"estimated_{k}"] = ev
            row[f"{k}_error"] = ev - tv
            row[f"{k}_abs_error"] = abs(ev - tv) if np.isfinite(ev - tv) else np.nan
        ti = innovation_cumulants(cal.alpha, cal.beta, cal.delta, cal.lambda_ou, cal.settings.dt)
        ei = innovation_cumulants(est_params["alpha"], est_params["beta"], est_params["delta"], est_params["lambda_ou"], cal.settings.dt)
        for k, tv in ti.items():
            ev = ei[k]
            row[f"true_{k}"] = tv
            row[f"estimated_{k}"] = ev
            row[f"{k}_error"] = ev - tv
            row[f"{k}_abs_error"] = abs(ev - tv) if np.isfinite(ev - tv) else np.nan
            row[f"{k}_relative_error"] = (ev - tv) / abs(tv) if abs(tv) > 1e-10 else np.nan
        metrics, cf_long = cf_metrics(true_params, est_params, cal.settings, v_grid)
        row.update(metrics)
    return row, cf_long


def run_one(args: tuple[Calibration, int, int, int, SimulationSettings, np.ndarray, bool, int]) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    cal, replication_id, seed_base, n_observations, sim_settings, v_grid, save_paths, smoke_maxiter = args
    seed = seed_for(seed_base, cal.case_id, replication_id)
    settings = cal.settings
    if smoke_maxiter > 0:
        settings = EstimationSettings(**{**settings.__dict__, "fft_grid_size": min(settings.fft_grid_size, 512), "truncation_l": min(settings.truncation_l, 6.0), "maxiter": smoke_maxiter, "max_optimizer_starts": 1, "max_candidate_starts_to_score": 2, "fft_intervals": ("sample",)})
        cal = Calibration(**{**cal.__dict__, "settings": settings})
    spread, sim_diag = simulate_spread(cal, seed, n_observations, sim_settings)
    u_hat, gfit = gaussian_mean_once(spread)
    x_est = spread - u_hat
    true_params = {"alpha": cal.alpha, "beta": cal.beta, "delta": cal.delta, "lambda_ou": cal.lambda_ou}
    true_ll, true_interval = transition_loglik(x_est, true_params, settings)
    wu = wu_only_estimator(x_est, settings)
    single = wu_single_start_mle(x_est, wu, settings)
    full = full_mle_estimator(spread, u_hat, settings)
    rows: list[dict[str, Any]] = []
    cf_frames = []
    for est in (wu, single, full):
        row, cf = decorate_result(est, cal, u_hat, x_est, true_ll, true_interval, v_grid)
        row["replication_id"] = replication_id
        row["seed"] = seed
        row["n_observations"] = len(spread)
        row["gaussian_lambda_ou"] = safe_float(gfit.get("lambda_ou", gfit.get("lambda")))
        row["gaussian_sigma"] = safe_float(gfit.get("sigma"))
        if not cf.empty:
            cf.insert(0, "estimator", est["estimator"])
            cf.insert(0, "replication_id", replication_id)
            cf.insert(0, "case_id", cal.case_id)
            cf_frames.append(cf)
        rows.append(row)
    diag = {"case_id": cal.case_id, "replication_id": replication_id, "seed": seed, **sim_diag}
    path_record = {"case_id": cal.case_id, "replication_id": replication_id, "seed": seed, "spread_json": json.dumps(spread.tolist()) if save_paths else ""}
    return rows, pd.concat(cf_frames, ignore_index=True) if cf_frames else pd.DataFrame(), diag, path_record


def write_checkpoint(output_dir: Path, rep_rows: list[dict[str, Any]], cf_rows: list[pd.DataFrame], sim_diags: list[dict[str, Any]], path_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rep_rows).sort_values(["case_id", "replication_id", "estimator"]).to_csv(output_dir / "replication_results.csv", index=False)
    if cf_rows:
        pd.concat(cf_rows, ignore_index=True).to_csv(output_dir / "cf_values_long.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "cf_values_long.csv", index=False)
    pd.DataFrame(sim_diags).sort_values(["case_id", "replication_id"]).to_csv(output_dir / "simulator_numerical_diagnostics.csv", index=False)
    if path_rows and any(r.get("spread_json") for r in path_rows):
        pd.DataFrame(path_rows).to_csv(output_dir / "synthetic_paths.csv", index=False)


def completed_keys(output_dir: Path) -> set[tuple[int, int]]:
    path = output_dir / "replication_results.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["case_id", "replication_id"])
    counts = df.groupby(["case_id", "replication_id"]).size()
    return {tuple(map(int, key)) for key, n in counts.items() if int(n) >= 3}


def aggregate_outputs(output_dir: Path) -> None:
    rep = pd.read_csv(output_dir / "replication_results.csv")
    numeric_cols = [c for c in rep.columns if c.endswith(("_error", "_abs_error", "_squared_error")) or c in ["cf_mse", "cf_mae", "estimated_loglik", "loglik_minus_true", "runtime_seconds", "alpha", "beta", "delta", "lambda_ou"]]
    rows = []
    for (case_id, estimator), g in rep.groupby(["case_id", "estimator"], dropna=False):
        row = {"case_id": case_id, "estimator": estimator, "n_rows": len(g), "success_rate": float(g["success"].mean()), "failure_rate": float(1.0 - g["success"].mean()), "n_valid_estimates": int(g["success"].sum())}
        for c in numeric_cols:
            s = pd.to_numeric(g[c], errors="coerce").dropna()
            if s.empty:
                continue
            row[f"{c}_mean"] = float(s.mean()); row[f"{c}_median"] = float(s.median()); row[f"{c}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            row[f"{c}_p05"] = float(s.quantile(0.05)); row[f"{c}_p25"] = float(s.quantile(0.25)); row[f"{c}_p50"] = float(s.quantile(0.50)); row[f"{c}_p75"] = float(s.quantile(0.75)); row[f"{c}_p95"] = float(s.quantile(0.95))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "summary_by_case_estimator.csv", index=False)
    case_summary = pd.DataFrame(rows)
    overall = case_summary.groupby("estimator", as_index=False).mean(numeric_only=True)
    overall.to_csv(output_dir / "summary_overall_estimator.csv", index=False)

    comparisons = []
    for case_id, g in rep.groupby("case_id"):
        piv = g.pivot(index="replication_id", columns="estimator")
        for a, b in [("wu_only", "wu_single_start_mle"), ("wu_single_start_mle", "full_mle"), ("wu_only", "full_mle")]:
            if a not in piv.columns.get_level_values(1) or b not in piv.columns.get_level_values(1):
                continue
            row = {"case_id": case_id, "comparison": f"{a}_vs_{b}"}
            for metric in ["alpha_abs_error", "beta_abs_error", "delta_abs_error", "lambda_ou_abs_error", "stationary_std_abs_error", "stationary_skewness_abs_error", "stationary_excess_kurtosis_abs_error", "cf_mse"]:
                if metric not in piv.columns.get_level_values(0):
                    continue
                av = pd.to_numeric(piv[(metric, a)], errors="coerce")
                bv = pd.to_numeric(piv[(metric, b)], errors="coerce")
                diff = bv - av
                row[f"{metric}_fraction_b_better"] = float((bv < av).mean())
                row[f"{metric}_mean_paired_change"] = float(diff.mean())
                row[f"{metric}_median_paired_change"] = float(diff.median())
            if ("estimated_loglik", a) in piv.columns and ("estimated_loglik", b) in piv.columns:
                d = pd.to_numeric(piv[("estimated_loglik", b)], errors="coerce") - pd.to_numeric(piv[("estimated_loglik", a)], errors="coerce")
                row["mean_loglik_improvement"] = float(d.mean()); row["median_loglik_improvement"] = float(d.median())
                row["same_optimum_rate_loglik_1e_4"] = float((d.abs() <= 1e-4).mean())
            if ("runtime_seconds", a) in piv.columns and ("runtime_seconds", b) in piv.columns:
                ratio = pd.to_numeric(piv[("runtime_seconds", b)], errors="coerce") / pd.to_numeric(piv[("runtime_seconds", a)], errors="coerce")
                row["mean_runtime_ratio_b_over_a"] = float(ratio.replace([np.inf, -np.inf], np.nan).mean())
                row["median_runtime_ratio_b_over_a"] = float(ratio.replace([np.inf, -np.inf], np.nan).median())
            comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(output_dir / "paired_estimator_comparisons.csv", index=False)

    full = rep[rep["estimator"] == "full_mle"].copy()
    freq_cols = ["best_start_method", "best_optimizer_start_method", "best_fft_interval", "best_optimizer_fft_interval"]
    if not full.empty:
        freq = full.groupby([c for c in freq_cols if c in full.columns], dropna=False).size().reset_index(name="count")
        freq["fraction"] = freq["count"] / float(len(full))
        freq.to_csv(output_dir / "full_mle_start_method_frequencies.csv", index=False)
    else:
        pd.DataFrame(columns=freq_cols + ["count", "fraction"]).to_csv(output_dir / "full_mle_start_method_frequencies.csv", index=False)

    failures = rep[~rep["success"].fillna(False)].copy()
    if not failures.empty:
        failures["failure_stage"] = failures["estimator"]
        failures.groupby(["case_id", "estimator", "failure_stage", "failure_reason"], dropna=False).size().reset_index(name="count").to_csv(output_dir / "failure_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["case_id", "estimator", "failure_stage", "failure_reason", "count"]).to_csv(output_dir / "failure_summary.csv", index=False)


def simulator_validation(output_dir: Path, calibrations: list[Calibration], sim_diags: pd.DataFrame, seed_base: int, sim_settings: SimulationSettings, v_grid: np.ndarray, smoke: bool) -> None:
    rows = []
    n_diag = 1000 if smoke else 50000
    for cal in calibrations:
        seed = seed_for(seed_base + 777_000_000, cal.case_id, 0)
        sim, diag = build_simulator(cal, seed, sim_settings)
        innov = sim.sample_innovations(n_diag)
        centered = innov - float(np.mean(innov))
        k2 = float(np.mean(centered**2)); k3 = float(np.mean(centered**3)); k4 = float(np.mean(centered**4) - 3.0 * k2 * k2)
        theo = innovation_cumulants(cal.alpha, cal.beta, cal.delta, cal.lambda_ou, cal.settings.dt)
        residual = sim.simulate(n=min(FORMATION_OBSERVATIONS, 5000 if smoke else FORMATION_OBSERVATIONS), stationary_start=True)
        row = {"case_id": cal.case_id, "diagnostic_innovation_sample_size": n_diag, "empirical_innov_c1": float(np.mean(innov)), "empirical_innov_c2": k2, "empirical_innov_c3": k3, "empirical_innov_c4": k4, **{f"theoretical_{k}": v for k, v in theo.items()}, **diag}
        for lag in (1, 5, 10, 20, 60, 390):
            if len(residual) > lag:
                acf = float(np.corrcoef(residual[:-lag], residual[lag:])[0, 1])
                target = math.exp(-cal.lambda_ou * lag * cal.settings.dt)
                row[f"acf_lag_{lag}"] = acf; row[f"theoretical_acf_lag_{lag}"] = target; row[f"acf_lag_{lag}_abs_error"] = abs(acf - target)
        true_ic = innovation_cumulants(cal.alpha, cal.beta, cal.delta, cal.lambda_ou, cal.settings.dt)
        u_grid = v_grid / math.sqrt(max(true_ic["innov_c2"], np.finfo(float).tiny))
        emp_phi = np.exp(1j * np.outer(u_grid, innov)).mean(axis=1)
        true_phi = centered_zstar_cf(u_grid, cal.alpha, cal.beta, cal.delta, cal.lambda_ou, cal.settings.dt)
        err = np.abs(emp_phi - true_phi)
        row["empirical_innovation_cf_mse"] = float(np.mean(err**2)); row["empirical_innovation_cf_mae"] = float(np.mean(err))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "simulator_validation.csv", index=False)


def write_readme(output_dir: Path) -> None:
    text = """# Fixed-Mean NIG-OU Simulation Study

This folder is produced by `scripts/run_nig_simulation_study.py`. The normal run uses the final recovered production settings from `output_NEW`, generates stationary fixed-mean NIG-OU formation samples, estimates the Gaussian formation mean once per sample, and compares Wu-only, Wu-single-start FFT MLE, and the full production multistart FFT MLE on paired synthetic datasets.

Key files: `study_settings.csv`, `selected_calibrations.csv`, `replication_results.csv`, `summary_by_case_estimator.csv`, `summary_overall_estimator.csv`, `paired_estimator_comparisons.csv`, `cf_values_long.csv`, `simulator_validation.csv`, `simulator_numerical_diagnostics.csv`, `full_mle_start_method_frequencies.csv`, and `failure_summary.csv`.
"""
    (output_dir / "README.md").write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixed-mean NIG-OU Monte Carlo simulation study.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--replications-per-case", type=int, default=250)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=20260807)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cf-grid-points", type=int, default=201)
    parser.add_argument("--cf-standardized-max", type=float, default=20.0)
    parser.add_argument("--save-synthetic-paths", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use tiny optimizer settings and one replication per case for code validation only.")
    args = parser.parse_args(argv)

    repo_root = REPO_ROOT
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_settings = SimulationSettings()
    pool = load_candidate_pool(repo_root)
    selected = select_calibrations(pool)
    for col in ["alpha", "beta", "delta", "lambda_ou"]:
        selected[f"true_{col}"] = selected[col]
    selected.to_csv(output_dir / "selected_calibrations.csv", index=False)
    write_settings_csv(output_dir, selected, sim_settings)
    write_readme(output_dir)
    calibrations = [calibration_from_row(row) for _, row in selected.iterrows()]

    reps_per_case = 1 if args.smoke_test else int(args.replications_per_case)
    smoke_maxiter = 2 if args.smoke_test else 0
    v_grid = np.linspace(0.0, float(args.cf_standardized_max), int(args.cf_grid_points))
    done = completed_keys(output_dir) if args.resume else set()
    tasks = [(cal, rep, args.seed_base, FORMATION_OBSERVATIONS, sim_settings, v_grid, args.save_synthetic_paths, smoke_maxiter) for cal in calibrations for rep in range(reps_per_case) if (cal.case_id, rep) not in done]

    rep_rows: list[dict[str, Any]] = []
    cf_rows: list[pd.DataFrame] = []
    sim_diags: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    if args.resume and (output_dir / "replication_results.csv").exists():
        rep_rows.extend(pd.read_csv(output_dir / "replication_results.csv").to_dict("records"))
        if (output_dir / "cf_values_long.csv").exists():
            cf_rows.append(pd.read_csv(output_dir / "cf_values_long.csv"))
        if (output_dir / "simulator_numerical_diagnostics.csv").exists():
            sim_diags.extend(pd.read_csv(output_dir / "simulator_numerical_diagnostics.csv").to_dict("records"))

    completed = 0
    if tasks:
        if int(args.n_jobs) == 1:
            iterator = map(run_one, tasks)
            for rows, cf, diag, path_record in iterator:
                rep_rows.extend(rows); sim_diags.append(diag); path_rows.append(path_record)
                if not cf.empty: cf_rows.append(cf)
                completed += 1
                if completed % max(1, int(args.checkpoint_every)) == 0:
                    write_checkpoint(output_dir, rep_rows, cf_rows, sim_diags, path_rows)
        else:
            with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as executor:
                futures = [executor.submit(run_one, task) for task in tasks]
                for fut in as_completed(futures):
                    rows, cf, diag, path_record = fut.result()
                    rep_rows.extend(rows); sim_diags.append(diag); path_rows.append(path_record)
                    if not cf.empty: cf_rows.append(cf)
                    completed += 1
                    if completed % max(1, int(args.checkpoint_every)) == 0:
                        write_checkpoint(output_dir, rep_rows, cf_rows, sim_diags, path_rows)

    write_checkpoint(output_dir, rep_rows, cf_rows, sim_diags, path_rows)
    aggregate_outputs(output_dir)
    simulator_validation(output_dir, calibrations, pd.DataFrame(sim_diags), args.seed_base, sim_settings, v_grid, bool(args.smoke_test))
    print(f"wrote NIG simulation study outputs to {output_dir}")
    print(f"completed synthetic datasets this run: {completed}; total estimator rows: {len(rep_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
