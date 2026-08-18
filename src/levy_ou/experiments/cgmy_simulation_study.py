"""Monte Carlo study for the CGMY-OU estimator."""

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
from scipy import optimize

from levy_ou.estimators.cgmy_ou import (
    DT,
    _asymmetric_stationary_moment_start,
    _build_candidates,
    _empirical_cumulants,
    _lambda_start_multilag,
    _raw_in_bounds,
    _stationary_target_from_zstar,
    asymmetric_params_to_raw,
    asymmetric_raw_to_params,
    estimate_cgmy_ou_fft_mle,
    gaussian_ar1_start,
    stationary_cumulants_zero_mean,
    stationary_density_fft,
    stationary_cf_zero_mean,
    valdivieso_innovations,
    zstar_cf,
    zstar_cumulants,
    zstar_density_fft,
)
from levy_ou.simulation.cgmy_simulator import (
    build_cgmy_ou_simulator_with_fallback,
    innovation_cumulants as simulator_innovation_cumulants,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path("outputs/cgmy_simulation_study")
FORMATION_OBSERVATIONS = 30 * 390
DEFAULT_LAMBDA_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
DEFAULT_FFT_INTERVALS = ("cumulant", "sample")
FINAL_CGMY_ESTIMATE_FILES = (
    ("energy", 2008, "gaussian_top10", "outputs/energy_2008/estimates/gaussian_top10_cgmy_asymmetric/model_estimates.csv"),
    ("energy", 2024, "gaussian_top10", "outputs/energy_2024/estimates/gaussian_top10_cgmy_asymmetric/model_estimates.csv"),
    ("communication", 2024, "gaussian_top10", "outputs/communication_2024/estimates/gaussian_top10_cgmy_asymmetric/model_estimates.csv"),
    ("energy", 2008, "adf_capped10", "outputs/energy_2008/adf_capped10/estimates/adf_capped10_all_models/cgmy_model_estimates.csv"),
    ("energy", 2024, "adf_capped10", "outputs/energy_2024/adf_capped10/estimates/adf_capped10_all_models/cgmy_model_estimates.csv"),
    ("communication", 2024, "adf_capped10", "outputs/communication_2024/adf_capped10/estimates/adf_capped10_levy_models/cgmy_model_estimates.csv"),
)


@dataclass(frozen=True)
class EstimationSettings:
    dt: float = DT
    fft_grid_size: int = 8192
    truncation_l: float = 10.0
    maxiter: int = 220
    optimizer_method: str = "Powell"
    density_floor: float = 1e-300
    lambda_factors: tuple[float, ...] = DEFAULT_LAMBDA_FACTORS
    lambda_max_lag: int = 20
    max_optimizer_starts: int = 3
    max_candidate_starts_to_score: int | None = 16
    fft_intervals: tuple[str, ...] = DEFAULT_FFT_INTERVALS
    include_stationary_density: bool = False
    include_ecf_start: bool = True
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
            "include_ecf_start": self.include_ecf_start,
            "wu_first": self.wu_first,
        }


@dataclass(frozen=True)
class SimulationSettings:
    sim_fft_grid_size: int = 32768
    first_shift_fraction: float = 0.95
    second_shift_fraction: float = 0.90
    density_du: float = 20.0
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
    selection_scope: str
    source_file: str
    pair: str
    window: str
    u: float
    C: float
    G: float
    M: float
    Y: float
    lambda_ou: float
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


def cgmy_derived(C: float, G: float, M: float, Y: float, lambda_ou: float, dt: float) -> dict[str, float]:
    stat = stationary_cumulants_zero_mean(C, G, M, Y)
    k2 = stat["c2"]
    std = math.sqrt(k2) if k2 > 0 else np.nan
    skew = stat["c3"] / (std**3) if std > 0 else np.nan
    ex_kurt = stat["c4"] / (k2**2) if k2 > 0 else np.nan
    hl_model = math.log(2.0) / lambda_ou if lambda_ou > 0 else np.nan
    return {
        "stationary_variance": k2,
        "stationary_std": std,
        "stationary_skewness": skew,
        "stationary_excess_kurtosis": ex_kurt,
        "tempering_asymmetry_log_m_over_g": math.log(M / G) if G > 0 and M > 0 else np.nan,
        "activity_Y": Y,
        "half_life_model_time": hl_model,
        "half_life_minutes": hl_model / dt if dt > 0 else np.nan,
    }


def cgmy_innovation_cumulants(C: float, G: float, M: float, Y: float, lambda_ou: float, dt: float) -> dict[str, float]:
    return zstar_cumulants(C, G, M, Y, lambda_ou, dt)


def settings_from_row(row: pd.Series) -> EstimationSettings:
    intervals = str(row.get("fft_intervals", "cumulant,sample") or "cumulant,sample").replace(";", ",")
    fft_intervals = tuple(x.strip() for x in intervals.split(",") if x.strip()) or DEFAULT_FFT_INTERVALS
    return EstimationSettings(
        dt=safe_float(row.get("dt"), DT),
        fft_grid_size=int(safe_float(row.get("fft_grid_size"), 8192)),
        truncation_l=safe_float(row.get("truncation_l"), 10.0),
        maxiter=220,
        optimizer_method=str(row.get("optimizer_method", "Powell") or "Powell"),
        density_floor=safe_float(row.get("density_floor"), 1e-300),
        max_optimizer_starts=int(safe_float(row.get("optimizer_starts"), 3)),
        max_candidate_starts_to_score=int(safe_float(row.get("candidate_starts"), 16)) or None,
        fft_intervals=fft_intervals,
        include_stationary_density=parse_bool(row.get("include_stationary_density"), False),
        include_ecf_start=parse_bool(row.get("include_ecf_start"), True),
    )


def _pair_label(row: pd.Series) -> str:
    ticker_a = str(row.get("ticker_a", ""))
    ticker_b = str(row.get("ticker_b", ""))
    return f"{ticker_a}-{ticker_b}" if ticker_a or ticker_b else str(row.get("pair_index", ""))


def _window_label(row: pd.Series) -> str:
    if "window_id" in row and pd.notna(row.get("window_id")):
        return str(row.get("window_id"))
    start = str(row.get("formation_start", ""))
    end = str(row.get("formation_end", ""))
    return f"{start}:{end}" if start or end else ""


def load_candidate_pool(repo_root: Path) -> pd.DataFrame:
    frames = []
    for sector, year, scope, rel in FINAL_CGMY_ESTIMATE_FILES:
        path = repo_root / rel
        if not path.exists():
            continue
        df = pd.read_csv(path).copy()
        if df.empty:
            continue
        df["sector"] = sector
        df["year"] = year
        df["selection_scope"] = scope
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No final CGMY estimate files found.")
    pool = pd.concat(frames, ignore_index=True)
    for col in ["C", "G", "M", "Y", "lambda_ou"]:
        pool[col] = pd.to_numeric(pool[col], errors="coerce")
    if "mu" not in pool.columns:
        pool["mu"] = pd.to_numeric(pool.get("gaussian_mean", np.nan), errors="coerce")
    pool["valid_for_study"] = (
        np.isfinite(pool["C"]) & np.isfinite(pool["G"]) & np.isfinite(pool["M"]) & np.isfinite(pool["Y"]) & np.isfinite(pool["lambda_ou"])
        & (pool["C"] > 0) & (pool["G"] > 0) & (pool["M"] > 0) & (pool["Y"] > 0) & (pool["Y"] < 1) & (pool["lambda_ou"] > 0)
        & np.isfinite(pd.to_numeric(pool["mu"], errors="coerce"))
    )
    d_rows = []
    idxs = []
    for idx, row in pool[pool["valid_for_study"]].iterrows():
        d_rows.append(cgmy_derived(float(row.C), float(row.G), float(row.M), float(row.Y), float(row.lambda_ou), safe_float(row.get("dt"), DT)))
        idxs.append(idx)
    derived = pd.DataFrame(d_rows, index=idxs)
    for col in derived.columns:
        if col in pool.columns:
            pool = pool.drop(columns=[col])
        pool[col] = derived[col]
    return pool


def select_calibrations(pool: pd.DataFrame) -> pd.DataFrame:
    df_all = pool[pool["valid_for_study"]].copy()
    if df_all.empty:
        raise ValueError("No valid CGMY rows found.")
    df = df_all[(df_all["stationary_excess_kurtosis"] < 5_000) & (df_all["stationary_variance"] > 0)].copy()
    if len(df) < 4:
        df = df_all.copy()
    selected: list[int] = []
    used_dataset: set[tuple[str, int]] = set()

    def take(role: str, candidates: pd.DataFrame, score: str, ascending: bool = True) -> None:
        candidates = candidates.sort_values(score, ascending=ascending)
        for prefer_new in (True, False):
            for idx, row in candidates.iterrows():
                dataset = (str(row.get("sector", "")), int(row.get("year", 0)))
                if idx in selected:
                    continue
                if prefer_new and dataset in used_dataset:
                    continue
                selected.append(idx); used_dataset.add(dataset); df.loc[idx, "selection_reason"] = role
                return
        idx = candidates.index[0]
        selected.append(idx); df.loc[idx, "selection_reason"] = role

    med_lambda = float(df["lambda_ou"].median())
    med_y = float(df["Y"].median())
    med_kurt = float(df["stationary_excess_kurtosis"].median())
    df["moderate_score"] = (df["tempering_asymmetry_log_m_over_g"].abs() + (df["lambda_ou"] - med_lambda).abs() / (df["lambda_ou"].std() or 1.0) + (df["Y"] - med_y).abs())
    df["positive_skew_score"] = (df["stationary_skewness"] - float(df.loc[df["stationary_skewness"] > 0, "stationary_skewness"].quantile(0.95))).abs()
    df["negative_skew_score"] = (df["stationary_skewness"] - float(df.loc[df["stationary_skewness"] < 0, "stationary_skewness"].quantile(0.05))).abs()
    df["heavy_tail_score"] = df["stationary_excess_kurtosis"].rank(pct=True) + df["Y"].rank(pct=True) + (df["lambda_ou"] - med_lambda).abs().rank(pct=True)
    take("near-symmetric/moderate: small tempering asymmetry with central lambda and activity", df, "moderate_score", True)
    take("positively skewed: positive 95th-percentile stationary skewness", df[df["stationary_skewness"] > 0], "positive_skew_score", True)
    take("negatively skewed: negative 5th-percentile stationary skewness", df[df["stationary_skewness"] < 0], "negative_skew_score", True)
    take("heavy-tail/extreme: high kurtosis/activity with different mean reversion", df, "heavy_tail_score", False)
    out = df.loc[selected].copy().reset_index(drop=False).rename(columns={"index": "source_row_index"})
    out.insert(0, "case_id", np.arange(1, len(out) + 1))
    out.insert(1, "calibration_role", ["near_symmetric_moderate", "positive_skew", "negative_skew", "heavy_tail_extreme"][: len(out)])
    return out


def calibration_from_row(row: pd.Series) -> Calibration:
    return Calibration(
        case_id=int(row["case_id"]), role=str(row["calibration_role"]), sector=str(row["sector"]), year=int(row["year"]),
        selection_scope=str(row["selection_scope"]), source_file=str(row["source_file"]), pair=_pair_label(row), window=_window_label(row),
        u=safe_float(row.get("mu", row.get("gaussian_mean"))), C=float(row["C"]), G=float(row["G"]), M=float(row["M"]), Y=float(row["Y"]), lambda_ou=float(row["lambda_ou"]), settings=settings_from_row(row),
    )


def seed_for(seed_base: int, case_id: int, replication_id: int) -> int:
    return int(seed_base) + int(case_id) * 1_000_000 + int(replication_id)


def build_simulator(cal: Calibration, seed: int, sim_settings: SimulationSettings):
    return build_cgmy_ou_simulator_with_fallback(
        C=cal.C, G=cal.G, M=cal.M, Y=cal.Y, long_run_mean=cal.u, lam=cal.lambda_ou, dt=cal.settings.dt, seed=seed,
        shifted_attempts=sim_settings.shifted_attempts, density_n_fft=sim_settings.density_n_fft, density_du=sim_settings.density_du,
        density_negative_mass_tol=sim_settings.density_negative_mass_tol, density_raw_area_tol=sim_settings.density_raw_area_tol, build_stationary=True,
    )


def simulate_spread(cal: Calibration, seed: int, n_observations: int, sim_settings: SimulationSettings) -> tuple[np.ndarray, dict[str, Any]]:
    sim, diag = build_simulator(cal, seed, sim_settings)
    return np.asarray(sim.simulate(n=int(n_observations), stationary_start=True), dtype=float), diag


def moment_only_estimator(centered_x: np.ndarray, settings: EstimationSettings) -> dict[str, Any]:
    started = perf_counter()
    try:
        lambda0, acf1 = _lambda_start_multilag(centered_x, settings.dt, settings.lambda_max_lag)
        z = valdivieso_innovations(centered_x, lambda0, settings.dt)
        zc = _empirical_cumulants(z)
        target = _stationary_target_from_zstar(zc, lambda0, settings.dt)
        match = _asymmetric_stationary_moment_start(target, "cgmy_stationary_innovation_cumulant_match")
        if not match.get("valid"):
            raise ValueError(str(match.get("reason", "moment match failed")))
        out = {"success": True, "estimator": "moment_only", "C": float(match["C"]), "G": float(match["G"]), "M": float(match["M"]), "Y": float(match["Y"]), "lambda_ou": float(lambda0), "acf1": float(acf1), "runtime_seconds": perf_counter() - started, "failure_reason": "", "moment_objective": safe_float(match.get("moment_objective"))}
        for k, v in zc.items(): out[f"empirical_innov_{k}"] = v
        for k, v in target.items(): out[f"implied_stationary_{k}"] = v
        return out
    except Exception as exc:
        return {"success": False, "estimator": "moment_only", "runtime_seconds": perf_counter() - started, "failure_reason": f"moment_only: {type(exc).__name__}: {exc}"}


def transition_loglik(centered_x: np.ndarray, params: dict[str, float], settings: EstimationSettings) -> tuple[float, str]:
    best = (-np.inf, "")
    for interval in settings.fft_intervals:
        try:
            lam = float(params["lambda_ou"])
            z = valdivieso_innovations(centered_x, lam, settings.dt)
            dens, _ = zstar_density_fft(z, float(params["C"]), float(params["G"]), float(params["M"]), float(params["Y"]), lam, settings.dt, settings.fft_grid_size, settings.truncation_l, settings.density_floor, str(interval))
            ll = len(z) * lam * settings.dt + float(np.sum(np.log(dens)))
            if settings.include_stationary_density:
                sd, _ = stationary_density_fft(np.asarray([centered_x[0]]), float(params["C"]), float(params["G"]), float(params["M"]), float(params["Y"]), settings.fft_grid_size, settings.truncation_l, settings.density_floor, str(interval), centered_x)
                ll += float(np.log(sd[0]))
            if np.isfinite(ll) and ll > best[0]: best = (ll, str(interval))
        except Exception:
            continue
    return best


def single_start_mle(centered_x: np.ndarray, moment: dict[str, Any], settings: EstimationSettings) -> dict[str, Any]:
    started = perf_counter()
    if not moment.get("success"):
        return {"success": False, "estimator": "moment_single_start_mle", "runtime_seconds": 0.0, "failure_reason": "moment start unavailable"}
    try:
        raw0 = asymmetric_params_to_raw(float(moment["C"]), float(moment["G"]), float(moment["M"]), float(moment["Y"]), float(moment["lambda_ou"]))
        start_params = {k: float(moment[k]) for k in ("C", "G", "M", "Y", "lambda_ou")}
        start_ll, start_interval = transition_loglik(centered_x, start_params, settings)
        def objective(raw: np.ndarray, interval: str) -> float:
            if not _raw_in_bounds(raw): return 1e100
            C, G, M, Y, lam = asymmetric_raw_to_params(raw)
            if not (C > 0 and G > 0 and M > 0 and 0 < Y < 1 and lam > 0 and lam * settings.dt < 20): return 1e100
            ll, _ = transition_loglik(centered_x, {"C": C, "G": G, "M": M, "Y": Y, "lambda_ou": lam}, EstimationSettings(**{**settings.__dict__, "fft_intervals": (interval,)}))
            return -ll if np.isfinite(ll) else 1e100
        options: dict[str, Any] = {"maxiter": int(settings.maxiter)}
        if settings.optimizer_method == "Powell": options.update({"xtol": 1e-4, "ftol": 1e-4})
        elif settings.optimizer_method == "Nelder-Mead": options.update({"xatol": 1e-4, "fatol": 1e-4})
        best = None
        for interval in settings.fft_intervals:
            res = optimize.minimize(lambda values, interval=interval: objective(values, str(interval)), raw0, method=settings.optimizer_method, options=options)
            val = float(res.fun) if np.isfinite(res.fun) else 1e100
            if best is None or val < best[0]: best = (val, res, str(interval))
        assert best is not None
        _, res, interval = best
        C, G, M, Y, lam = asymmetric_raw_to_params(res.x)
        ll, final_interval = transition_loglik(centered_x, {"C": C, "G": G, "M": M, "Y": Y, "lambda_ou": lam}, settings)
        return {"success": bool(res.success and np.isfinite(ll)), "estimator": "moment_single_start_mle", "C": C, "G": G, "M": M, "Y": Y, "lambda_ou": lam, "runtime_seconds": perf_counter() - started, "failure_reason": "" if res.success else str(res.message), "start_loglik": start_ll, "start_fft_interval": start_interval, "estimated_loglik": ll, "best_fft_interval": final_interval, "loglik_improvement_from_start": ll - start_ll, "optimizer_success": bool(res.success), "optimizer_message": str(res.message), "optimizer_iterations": int(getattr(res, "nit", -1)), "optimizer_evaluations": int(getattr(res, "nfev", -1)), "optimizer_starts": len(settings.fft_intervals), **{f"moment_start_{k}": v for k, v in start_params.items()}}
    except Exception as exc:
        return {"success": False, "estimator": "moment_single_start_mle", "runtime_seconds": perf_counter() - started, "failure_reason": f"single_start: {type(exc).__name__}: {exc}"}


def full_mle(spread: np.ndarray, settings: EstimationSettings) -> dict[str, Any]:
    started = perf_counter()
    try:
        fit = estimate_cgmy_ou_fft_mle(pd.Series(spread), **settings.kwargs())
        out = {"estimator": "full_mle", "success": bool(fit.get("valid", False)), "runtime_seconds": perf_counter() - started, "failure_reason": str(fit.get("reason") or "")}
        out.update({k: v for k, v in fit.items() if isinstance(v, (str, int, float, bool, np.integer, np.floating)) or v is None})
        return out
    except Exception as exc:
        return {"success": False, "estimator": "full_mle", "runtime_seconds": perf_counter() - started, "failure_reason": f"full_mle: {type(exc).__name__}: {exc}"}


def cf_metrics(true_params: dict[str, float], est_params: dict[str, float], settings: EstimationSettings, v_grid: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    ic = cgmy_innovation_cumulants(**true_params, dt=settings.dt)
    sigma_y = math.sqrt(max(ic["c2"], np.finfo(float).tiny))
    u_grid = v_grid / sigma_y
    true_phi = zstar_cf(u_grid, true_params["C"], true_params["G"], true_params["M"], true_params["Y"], true_params["lambda_ou"], settings.dt)
    est_phi = zstar_cf(u_grid, est_params["C"], est_params["G"], est_params["M"], est_params["Y"], est_params["lambda_ou"], settings.dt)
    diff = est_phi - true_phi
    absdiff = np.abs(diff)
    return {"cf_mse": float(np.mean(absdiff**2)), "cf_mae": float(np.mean(absdiff)), "cf_max_abs_error": float(np.max(absdiff)), "cf_real_rmse": float(np.sqrt(np.mean(diff.real**2))), "cf_imag_rmse": float(np.sqrt(np.mean(diff.imag**2)))}, pd.DataFrame({"standardized_frequency_v": v_grid, "actual_frequency_u": u_grid, "true_cf_real": true_phi.real, "true_cf_imag": true_phi.imag, "fitted_cf_real": est_phi.real, "fitted_cf_imag": est_phi.imag, "absolute_complex_error": absdiff})


def decorate(base: dict[str, Any], cal: Calibration, u_hat: float, centered_x: np.ndarray, true_ll: float, true_interval: str, v_grid: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    row = {"case_id": cal.case_id, "calibration_role": cal.role, "sector": cal.sector, "year": cal.year, "selection_scope": cal.selection_scope, "source_file": cal.source_file, "pair": cal.pair, "window": cal.window, "true_u": cal.u, "estimated_u": u_hat, "mean_error": u_hat - cal.u, "true_C": cal.C, "true_G": cal.G, "true_M": cal.M, "true_Y": cal.Y, "true_lambda_ou": cal.lambda_ou, "true_loglik": true_ll, "true_loglik_interval": true_interval}
    row.update(base)
    cf = pd.DataFrame()
    if base.get("success") and all(k in base for k in ("C", "G", "M", "Y", "lambda_ou")):
        true_params = {"C": cal.C, "G": cal.G, "M": cal.M, "Y": cal.Y, "lambda_ou": cal.lambda_ou}
        est_params = {k: float(base[k]) for k in ("C", "G", "M", "Y", "lambda_ou")}
        ll, interval = transition_loglik(centered_x, est_params, cal.settings)
        row["estimated_loglik"] = ll; row["estimated_loglik_interval"] = interval; row["loglik_minus_true"] = ll - true_ll
        for p in ("C", "G", "M", "Y", "lambda_ou"):
            err = est_params[p] - true_params[p]
            row[f"{p}_error"] = err; row[f"{p}_abs_error"] = abs(err); row[f"{p}_squared_error"] = err * err; row[f"{p}_relative_error"] = err / abs(true_params[p]) if abs(true_params[p]) > 1e-8 else np.nan
        td = cgmy_derived(cal.C, cal.G, cal.M, cal.Y, cal.lambda_ou, cal.settings.dt); ed = cgmy_derived(est_params["C"], est_params["G"], est_params["M"], est_params["Y"], est_params["lambda_ou"], cal.settings.dt)
        for k, tv in td.items(): row[f"true_{k}"] = tv; row[f"estimated_{k}"] = ed[k]; row[f"{k}_error"] = ed[k] - tv; row[f"{k}_abs_error"] = abs(ed[k] - tv) if np.isfinite(ed[k] - tv) else np.nan
        ti = cgmy_innovation_cumulants(cal.C, cal.G, cal.M, cal.Y, cal.lambda_ou, cal.settings.dt); ei = cgmy_innovation_cumulants(est_params["C"], est_params["G"], est_params["M"], est_params["Y"], est_params["lambda_ou"], cal.settings.dt)
        for k, tv in ti.items(): row[f"true_innov_{k}"] = tv; row[f"estimated_innov_{k}"] = ei[k]; row[f"innov_{k}_error"] = ei[k] - tv; row[f"innov_{k}_abs_error"] = abs(ei[k] - tv) if np.isfinite(ei[k] - tv) else np.nan
        metrics, cf = cf_metrics(true_params, est_params, cal.settings, v_grid); row.update(metrics)
    return row, cf


def run_one(args: tuple[Calibration, int, int, int, SimulationSettings, np.ndarray, bool, int]):
    cal, rep, seed_base, n_obs, sim_settings, v_grid, save_paths, smoke_maxiter = args
    seed = seed_for(seed_base, cal.case_id, rep)
    settings = cal.settings
    if smoke_maxiter > 0:
        settings = EstimationSettings(**{**settings.__dict__, "fft_grid_size": min(settings.fft_grid_size, 512), "truncation_l": min(settings.truncation_l, 6.0), "maxiter": smoke_maxiter, "max_optimizer_starts": 1, "max_candidate_starts_to_score": 2, "fft_intervals": ("sample",), "include_ecf_start": False})
        cal = Calibration(**{**cal.__dict__, "settings": settings})
    spread, sim_diag = simulate_spread(cal, seed, n_obs, sim_settings)
    gfit = gaussian_ar1_start(spread, settings.dt)
    u_hat = float(gfit["mu"])
    centered_x = spread - u_hat
    true_params = {"C": cal.C, "G": cal.G, "M": cal.M, "Y": cal.Y, "lambda_ou": cal.lambda_ou}
    true_ll, true_interval = transition_loglik(centered_x, true_params, settings)
    m = moment_only_estimator(centered_x, settings)
    single = single_start_mle(centered_x, m, settings)
    full = full_mle(spread, settings)
    rows = []; cf_frames = []
    for est in (m, single, full):
        row, cf = decorate(est, cal, u_hat, centered_x, true_ll, true_interval, v_grid)
        row.update({"replication_id": rep, "seed": seed, "n_observations": len(spread), "gaussian_lambda_ou": gfit.get("lambda"), "gaussian_resid_std": gfit.get("resid_std")})
        rows.append(row)
        if not cf.empty:
            cf.insert(0, "estimator", est["estimator"]); cf.insert(0, "replication_id", rep); cf.insert(0, "case_id", cal.case_id); cf_frames.append(cf)
    return rows, pd.concat(cf_frames, ignore_index=True) if cf_frames else pd.DataFrame(), {"case_id": cal.case_id, "replication_id": rep, "seed": seed, **sim_diag}, {"case_id": cal.case_id, "replication_id": rep, "seed": seed, "spread_json": json.dumps(spread.tolist()) if save_paths else ""}


def write_settings_csv(output_dir: Path, selected: pd.DataFrame, sim_settings: SimulationSettings) -> None:
    rows = []
    for setting, value in {
        "formation_observations": FORMATION_OBSERVATIONS, "dt": DT, "lambda_factors": DEFAULT_LAMBDA_FACTORS, "lambda_max_lag": 20, "maxiter": 220,
        "optimizer": "Powell with xtol=1e-4, ftol=1e-4", "fft_intervals": DEFAULT_FFT_INTERVALS, "include_ecf_start": True,
        "parameter_transform": "raw=(log_C, log_G, log_M, logit((Y-0.02)/0.96), log_lambda)", "constraints": "C>0, G>0, M>0, 0<Y<1, lambda>0, lambda*dt<20; raw bounds (-40,40)",
        "simulation_shifted_attempt_1": f"n_fft={sim_settings.sim_fft_grid_size}, shift_fraction={sim_settings.first_shift_fraction}", "simulation_shifted_attempt_2": f"n_fft={max(sim_settings.sim_fft_grid_size*2, 2**16)}, shift_fraction={sim_settings.second_shift_fraction}", "density_fft_fallback": f"n_fft={sim_settings.density_n_fft}, du={sim_settings.density_du}",
    }.items(): rows.append({"setting": setting, "value": value, "source_file": "CGMY production estimator/simulator and final estimate CSVs", "source_field": setting, "notes": "production behaviour mirrored unless --smoke-test is used"})
    for _, row in selected.iterrows():
        for setting in ["dt", "fft_grid_size", "truncation_l", "density_floor", "optimizer_method", "candidate_starts", "generated_candidate_starts", "scored_starts", "optimizer_starts", "include_stationary_density", "include_ecf_start", "best_fft_interval"]:
            rows.append({"setting": f"case_{int(row.case_id)}.{setting}", "value": row.get(setting, ""), "source_file": row["source_file"], "source_field": setting, "notes": row.get("selection_scope", "")})
    pd.DataFrame(rows).to_csv(output_dir / "study_settings.csv", index=False)


def write_checkpoint(output_dir: Path, rep_rows: list[dict[str, Any]], cf_rows: list[pd.DataFrame], sim_diags: list[dict[str, Any]], path_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rep_rows).sort_values(["case_id", "replication_id", "estimator"]).to_csv(output_dir / "replication_results.csv", index=False)
    (pd.concat(cf_rows, ignore_index=True) if cf_rows else pd.DataFrame()).to_csv(output_dir / "cf_values_long.csv", index=False)
    pd.DataFrame(sim_diags).sort_values(["case_id", "replication_id"]).to_csv(output_dir / "simulator_numerical_diagnostics.csv", index=False)
    if path_rows and any(r.get("spread_json") for r in path_rows): pd.DataFrame(path_rows).to_csv(output_dir / "synthetic_paths.csv", index=False)


def completed_keys(output_dir: Path) -> set[tuple[int, int]]:
    path = output_dir / "replication_results.csv"
    if not path.exists(): return set()
    df = pd.read_csv(path, usecols=["case_id", "replication_id"])
    counts = df.groupby(["case_id", "replication_id"]).size()
    return {tuple(map(int, k)) for k, n in counts.items() if int(n) >= 3}


def aggregate_outputs(output_dir: Path) -> None:
    rep = pd.read_csv(output_dir / "replication_results.csv")
    numeric = [c for c in rep.columns if c.endswith(("_error", "_abs_error", "_squared_error")) or c in ["cf_mse", "cf_mae", "estimated_loglik", "loglik_minus_true", "runtime_seconds", "C", "G", "M", "Y", "lambda_ou"]]
    rows = []
    for (case_id, estimator), g in rep.groupby(["case_id", "estimator"], dropna=False):
        row = {"case_id": case_id, "estimator": estimator, "n_rows": len(g), "success_rate": float(g["success"].mean()), "failure_rate": float(1 - g["success"].mean()), "n_valid_estimates": int(g["success"].sum())}
        for c in numeric:
            s = pd.to_numeric(g[c], errors="coerce").dropna()
            if s.empty: continue
            row[f"{c}_mean"] = float(s.mean()); row[f"{c}_median"] = float(s.median()); row[f"{c}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            for q in [0.05, 0.25, 0.5, 0.75, 0.95]: row[f"{c}_p{int(q*100):02d}"] = float(s.quantile(q))
        rows.append(row)
    case_summary = pd.DataFrame(rows); case_summary.to_csv(output_dir / "summary_by_case_estimator.csv", index=False); case_summary.groupby("estimator", as_index=False).mean(numeric_only=True).to_csv(output_dir / "summary_overall_estimator.csv", index=False)
    comps = []
    for case_id, g in rep.groupby("case_id"):
        piv = g.pivot(index="replication_id", columns="estimator")
        for a, b in [("moment_only", "moment_single_start_mle"), ("moment_single_start_mle", "full_mle"), ("moment_only", "full_mle")]:
            if a not in piv.columns.get_level_values(1) or b not in piv.columns.get_level_values(1): continue
            row = {"case_id": case_id, "comparison": f"{a}_vs_{b}"}
            for metric in ["C_abs_error", "G_abs_error", "M_abs_error", "Y_abs_error", "lambda_ou_abs_error", "stationary_std_abs_error", "stationary_skewness_abs_error", "stationary_excess_kurtosis_abs_error", "cf_mse"]:
                if metric not in piv.columns.get_level_values(0): continue
                av = pd.to_numeric(piv[(metric, a)], errors="coerce"); bv = pd.to_numeric(piv[(metric, b)], errors="coerce"); diff = bv - av
                row[f"{metric}_fraction_b_better"] = float((bv < av).mean()); row[f"{metric}_mean_paired_change"] = float(diff.mean()); row[f"{metric}_median_paired_change"] = float(diff.median())
            if ("estimated_loglik", a) in piv.columns and ("estimated_loglik", b) in piv.columns:
                d = pd.to_numeric(piv[("estimated_loglik", b)], errors="coerce") - pd.to_numeric(piv[("estimated_loglik", a)], errors="coerce")
                row["mean_loglik_improvement"] = float(d.mean()); row["median_loglik_improvement"] = float(d.median()); row["same_optimum_rate_loglik_1e_4"] = float((d.abs() <= 1e-4).mean())
            comps.append(row)
    pd.DataFrame(comps).to_csv(output_dir / "paired_estimator_comparisons.csv", index=False)
    full = rep[rep["estimator"] == "full_mle"].copy()
    cols = [c for c in ["best_start_method", "best_optimizer_start_method", "best_fft_interval", "best_optimizer_fft_interval"] if c in full.columns]
    (full.groupby(cols, dropna=False).size().reset_index(name="count") if cols and not full.empty else pd.DataFrame(columns=cols + ["count"])).to_csv(output_dir / "full_mle_start_method_frequencies.csv", index=False)
    failures = rep[~rep["success"].fillna(False)].copy()
    if not failures.empty:
        failures["failure_stage"] = failures["estimator"]; failures.groupby(["case_id", "estimator", "failure_stage", "failure_reason"], dropna=False).size().reset_index(name="count").to_csv(output_dir / "failure_summary.csv", index=False)
    else: pd.DataFrame(columns=["case_id", "estimator", "failure_stage", "failure_reason", "count"]).to_csv(output_dir / "failure_summary.csv", index=False)


def simulator_validation(output_dir: Path, calibrations: list[Calibration], seed_base: int, sim_settings: SimulationSettings, v_grid: np.ndarray, smoke: bool) -> None:
    rows = []
    n_diag = 1000 if smoke else 50000
    for cal in calibrations:
        sim, diag = build_simulator(cal, seed_for(seed_base + 777000000, cal.case_id, 0), sim_settings)
        innov = sim.sample_innovations(n_diag); centered = innov - float(np.mean(innov)); k2 = float(np.mean(centered**2)); k3 = float(np.mean(centered**3)); k4 = float(np.mean(centered**4) - 3*k2*k2)
        theo = simulator_innovation_cumulants(cal.C, cal.G, cal.M, cal.Y, math.exp(-cal.lambda_ou * cal.settings.dt))
        path = sim.simulate(n=min(FORMATION_OBSERVATIONS, 5000 if smoke else FORMATION_OBSERVATIONS), stationary_start=True)
        row = {"case_id": cal.case_id, "diagnostic_innovation_sample_size": n_diag, "empirical_innov_c1": float(np.mean(innov)), "empirical_innov_c2": k2, "empirical_innov_c3": k3, "empirical_innov_c4": k4, **{f"theoretical_{k}": v for k, v in theo.items()}, **diag}
        for lag in (1, 5, 10, 20, 60, 390):
            if len(path) > lag:
                acf = float(np.corrcoef(path[:-lag], path[lag:])[0, 1]); target = math.exp(-cal.lambda_ou * lag * cal.settings.dt); row[f"acf_lag_{lag}"] = acf; row[f"theoretical_acf_lag_{lag}"] = target; row[f"acf_lag_{lag}_abs_error"] = abs(acf - target)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "simulator_validation.csv", index=False)


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text("# Asymmetric CGMY-OU Simulation Study\n\nProduced by `scripts/run_cgmy_simulation_study.py`. It compares moment-only stationary CGMY starts, moment-single-start FFT MLE, and the full production asymmetric CGMY-OU multistart FFT MLE on paired stationary synthetic formation windows.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Asymmetric CGMY-OU Monte Carlo simulation study.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR)); parser.add_argument("--replications-per-case", type=int, default=250); parser.add_argument("--n-jobs", type=int, default=8); parser.add_argument("--seed-base", type=int, default=20260808); parser.add_argument("--checkpoint-every", type=int, default=10); parser.add_argument("--resume", action="store_true"); parser.add_argument("--cf-grid-points", type=int, default=201); parser.add_argument("--cf-standardized-max", type=float, default=20.0); parser.add_argument("--save-synthetic-paths", action="store_true"); parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir); output_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir; output_dir.mkdir(parents=True, exist_ok=True)
    sim_settings = SimulationSettings(); pool = load_candidate_pool(REPO_ROOT); selected = select_calibrations(pool); selected.to_csv(output_dir / "selected_calibrations.csv", index=False); write_settings_csv(output_dir, selected, sim_settings); write_readme(output_dir)
    calibrations = [calibration_from_row(row) for _, row in selected.iterrows()]
    reps = 1 if args.smoke_test else int(args.replications_per_case); smoke_maxiter = 2 if args.smoke_test else 0; v_grid = np.linspace(0, float(args.cf_standardized_max), int(args.cf_grid_points)); done = completed_keys(output_dir) if args.resume else set()
    tasks = [(cal, rep, args.seed_base, FORMATION_OBSERVATIONS, sim_settings, v_grid, args.save_synthetic_paths, smoke_maxiter) for cal in calibrations for rep in range(reps) if (cal.case_id, rep) not in done]
    rep_rows: list[dict[str, Any]] = []; cf_rows: list[pd.DataFrame] = []; sim_diags: list[dict[str, Any]] = []; path_rows: list[dict[str, Any]] = []
    if args.resume and (output_dir / "replication_results.csv").exists():
        rep_rows.extend(pd.read_csv(output_dir / "replication_results.csv").to_dict("records"))
        if (output_dir / "cf_values_long.csv").exists(): cf_rows.append(pd.read_csv(output_dir / "cf_values_long.csv"))
        if (output_dir / "simulator_numerical_diagnostics.csv").exists(): sim_diags.extend(pd.read_csv(output_dir / "simulator_numerical_diagnostics.csv").to_dict("records"))
    completed = 0
    if int(args.n_jobs) == 1:
        for rows, cf, diag, path_record in map(run_one, tasks):
            rep_rows.extend(rows); sim_diags.append(diag); path_rows.append(path_record); completed += 1
            if not cf.empty: cf_rows.append(cf)
            if completed % max(1, int(args.checkpoint_every)) == 0: write_checkpoint(output_dir, rep_rows, cf_rows, sim_diags, path_rows)
    else:
        with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as ex:
            futures = [ex.submit(run_one, task) for task in tasks]
            for fut in as_completed(futures):
                rows, cf, diag, path_record = fut.result(); rep_rows.extend(rows); sim_diags.append(diag); path_rows.append(path_record); completed += 1
                if not cf.empty: cf_rows.append(cf)
                if completed % max(1, int(args.checkpoint_every)) == 0: write_checkpoint(output_dir, rep_rows, cf_rows, sim_diags, path_rows)
    write_checkpoint(output_dir, rep_rows, cf_rows, sim_diags, path_rows); aggregate_outputs(output_dir); simulator_validation(output_dir, calibrations, args.seed_base, sim_settings, v_grid, bool(args.smoke_test))
    print(f"wrote CGMY simulation study outputs to {output_dir}"); print(f"completed synthetic datasets this run: {completed}; total estimator rows: {len(rep_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
