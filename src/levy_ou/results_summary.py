"""Portfolio result summaries for final OU pair backtests.

The backtest ``profits.csv`` files are pair-window summaries, not portfolio
series.  This module rebuilds daily marked-to-market pair-slot returns from
``trades.csv`` and the original quote panel, then aggregates those returns into
overlapping rolling-window portfolios.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from levy_ou.spreads import build_spread_with_anchor


PANEL_COLUMNS = [
    "timestamp_utc",
    "timestamp_ny",
    "trade_date",
    "market_time",
    "ticker",
    "model_price_close",
    "bid_close",
    "ask_close",
    "mid_close",
    "model_ready_price",
]


MODEL_ALIASES = {
    "all": "all",
    "gaussian": "gaussian",
    "gaussian_ou": "gaussian",
    "ou": "gaussian",
    "optimal_gaussian": "bertram_gaussian",
    "optimal_gaussian_ou": "bertram_gaussian",
    "bertram": "bertram_gaussian",
    "bertram_gaussian": "bertram_gaussian",
    "bertram_gaussian_ou": "bertram_gaussian",
    "zeng_lee": "zeng_lee_gaussian",
    "zeng_lee_gaussian": "zeng_lee_gaussian",
    "zeng_lee_gaussian_ou": "zeng_lee_gaussian",
    "zeng_lee_new": "zeng_lee_gaussian_new",
    "zeng_lee_gaussian_new": "zeng_lee_gaussian_new",
    "zeng_lee_gaussian_new_ou": "zeng_lee_gaussian_new",
    "zeng_lee_conventional": "zeng_lee_gaussian_conventional",
    "zeng_lee_gaussian_conventional": "zeng_lee_gaussian_conventional",
    "zeng_lee_gaussian_conventional_ou": "zeng_lee_gaussian_conventional",
    "gaussian_fixed_sigma_eq": "gaussian_fixed_sigma_eq",
    "fixed_sigma_eq": "gaussian_fixed_sigma_eq",
    "formation_mean_std": "formation_mean_std",
    "basic_baseline": "formation_mean_std",
    "bg": "symmetric_bg",
    "symmetric-bg": "symmetric_bg",
    "symmetric_bg": "symmetric_bg",
    "symmetric_bg_wu": "symmetric_bg",
    "nig": "nig",
    "nig_ou": "nig",
    "nig_fixed_mean": "nig",
    "cgmy": "cgmy",
    "cgmy_ou": "cgmy",
    "cgmy_asymmetric": "cgmy",
}

RETURN_BASIS_TO_PROFIT_COL = {
    "bid_ask": "bid_ask_total_return_on_gross",
    "midquote_fixed_bps": "midquote_fixed_bps_total_return_on_gross",
    "midquote": "midquote_total_return_on_gross",
}

RETURN_BASIS_TO_TRADE_COL = {
    "bid_ask": "bid_ask_return_on_gross",
    "midquote_fixed_bps": "midquote_fixed_bps_return_on_gross",
    "midquote": "midquote_return_on_gross",
}

RETURN_ALIASES = {
    "all": "all",
    "bid_ask": "bid_ask",
    "bidask": "bid_ask",
    "bid-ask": "bid_ask",
    "bid_ask_total_return_on_gross": "bid_ask",
    "midquote_fixed_bps": "midquote_fixed_bps",
    "fixed_bps": "midquote_fixed_bps",
    "midquote_fixed_cost": "midquote_fixed_bps",
    "midquote_fixed_bps_total_return_on_gross": "midquote_fixed_bps",
    "midquote": "midquote",
    "mid": "midquote",
    "midquote_total_return_on_gross": "midquote",
}

DEFAULT_FAMA_FRENCH_DAILY_FACTORS_PATH = Path(
    os.environ.get("FAMA_FRENCH_DAILY_FACTORS_PATH", "data/fama_french/F-F_Research_Data_Factors_daily.csv")
)
GROUP_SENTINEL = "__MISSING__"
GROUP_KEY_COLUMNS = [
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "window_id",
    "pair_id",
    "threshold_row",
]

BASE_RESULT_GROUP_COLUMNS = [
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
]


def spread_leg_directions(direction: str) -> tuple[float, float]:
    if direction == "long_spread":
        return 1.0, -1.0
    if direction == "short_spread":
        return -1.0, 1.0
    raise ValueError("direction must be 'long_spread' or 'short_spread'.")


def load_lobster_panel(
    data_path: str | Path,
    tickers: list[str] | tuple[str, ...] | set[str],
    start_date: str,
    end_date: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    wanted = {str(ticker).upper() for ticker in tickers}
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(data_path, usecols=PANEL_COLUMNS, compression="infer", chunksize=int(chunksize)):
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
        ready = chunk["model_ready_price"].astype(str).str.lower().eq("true")
        mask = ready & chunk["ticker"].isin(wanted) & chunk["trade_date"].astype(str).between(str(start_date), str(end_date))
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    panel = pd.concat(chunks, ignore_index=True)
    return panel.sort_values(["timestamp_utc", "ticker"]).reset_index(drop=True)


def _pivot_pair_from_ticker_frames(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    start: str,
    end: str,
    anchor_a: float | None = None,
    anchor_b: float | None = None,
) -> pd.DataFrame:
    date_a = frame_a["trade_date_str"] if "trade_date_str" in frame_a.columns else frame_a["trade_date"].astype(str)
    date_b = frame_b["trade_date_str"] if "trade_date_str" in frame_b.columns else frame_b["trade_date"].astype(str)
    a = frame_a[date_a.between(str(start), str(end))].copy()
    b = frame_b[date_b.between(str(start), str(end))].copy()
    if a.empty or b.empty:
        return pd.DataFrame()
    a = a[
        [
            "timestamp_utc",
            "timestamp_ny",
            "trade_date",
            "market_time",
            "model_price_close",
            "mid_close",
            "bid_close",
            "ask_close",
        ]
    ].rename(
        columns={
            "model_price_close": "price_a",
            "mid_close": "mid_a",
            "bid_close": "bid_a",
            "ask_close": "ask_a",
        }
    )
    b = b[
        [
            "timestamp_utc",
            "model_price_close",
            "mid_close",
            "bid_close",
            "ask_close",
        ]
    ].rename(
        columns={
            "model_price_close": "price_b",
            "mid_close": "mid_b",
            "bid_close": "bid_b",
            "ask_close": "ask_b",
        }
    )
    out = a.merge(b, on="timestamp_utc", how="inner").sort_values("timestamp_utc")
    needed = ["price_a", "price_b", "mid_a", "mid_b", "bid_a", "ask_a", "bid_b", "ask_b"]
    out = out.dropna(subset=needed)
    out = out[(out[needed] > 0.0).all(axis=1)]
    if out.empty:
        return pd.DataFrame()
    if anchor_a is None or anchor_b is None:
        anchor_a = float(out["price_a"].iloc[0])
        anchor_b = float(out["price_b"].iloc[0])
    out["spread"] = build_spread_with_anchor(
        out["price_a"].to_numpy(dtype=float),
        out["price_b"].to_numpy(dtype=float),
        anchor_a=float(anchor_a),
        anchor_b=float(anchor_b),
    )
    return out.reset_index(drop=True)

SUMMARY_LEADING_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "boundary_policy",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
    "sample_start_date",
    "sample_end_date",
    "num_daily_observations",
    "num_windows",
    "min_live_vintages",
    "max_live_vintages",
    "required_live_vintages",
    "first_available_date",
    "last_available_date",
    "first_complete_date",
    "last_complete_date",
    "dates_excluded_at_start",
    "dates_excluded_at_end",
    "retained_complete_dates",
    "minimum_live_vintages_before_filter",
    "maximum_live_vintages_before_filter",
    "minimum_live_vintages_after_filter",
    "maximum_live_vintages_after_filter",
    "complete_committee_validation_passed",
    "committed_pair_slots_per_window",
    "test_period_net_return",
    "annualized_net_return",
    "annualized_net_return_additive",
    "annualized_net_return_minus_additive",
    "annualized_volatility",
    "annualized_sharpe_rf_zero",
    "annualized_sharpe_fama_french_rf",
    "maximum_drawdown",
    "mean_daily_return",
    "median_daily_return",
    "daily_return_std",
    "positive_day_rate",
    "minimum_daily_return",
    "maximum_daily_return",
    "newey_west_lags",
    "newey_west_tstat",
    "newey_west_pvalue",
]

DAILY_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "boundary_policy",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
    "date",
    "num_live_vintages",
    "num_selected_pairs",
    "num_pairs_with_open_positions",
    "num_trades_opened",
    "num_trades_closed",
    "risk_free_daily_fama_french",
    "daily_pnl_return_on_initial_committed_capital",
    "daily_return",
    "wealth_index",
    "drawdown",
]

WINDOW_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
    "window_id",
    "trading_start_date",
    "trading_end_date",
    "number_of_trading_days",
    "committed_pair_slots",
    "actual_selected_pairs",
    "pairs_that_traded",
    "pair_participation_rate",
    "completed_trades",
    "forced_exits",
    "forced_exit_rate",
    "compounded_window_return",
    "annualized_window_return_additive",
    "mean_daily_window_return",
    "median_daily_window_return",
    "positive_day_rate",
    "mean_trade_return",
    "median_trade_return",
    "mean_holding_minutes",
    "median_holding_minutes",
]

TRADE_METRIC_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
    "committed_pair_slots",
    "actual_selected_pairs",
    "pairs_that_traded",
    "pair_participation_rate",
    "num_completed_trades",
    "trades_per_window",
    "trades_per_committed_pair_slot",
    "win_rate",
    "mean_trade_return",
    "median_trade_return",
    "average_gain",
    "median_gain",
    "average_loss",
    "median_loss",
    "mean_holding_minutes",
    "median_holding_minutes",
    "num_forced_exits",
    "forced_exit_rate",
    "mean_normal_exit_return",
    "median_normal_exit_return",
    "mean_forced_exit_return",
    "median_forced_exit_return",
]

THRESHOLD_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "num_threshold_rows",
    "num_valid_threshold_rows",
    "valid_threshold_rate",
    "median_d_plus",
    "q25_d_plus",
    "q75_d_plus",
    "median_d_minus",
    "q25_d_minus",
    "q75_d_minus",
    "median_d_plus_over_scale",
    "q25_d_plus_over_scale",
    "q75_d_plus_over_scale",
    "median_d_minus_over_scale",
    "q25_d_minus_over_scale",
    "q75_d_minus_over_scale",
    "mean_threshold_scale",
    "median_threshold_scale",
    "q25_threshold_scale",
    "q75_threshold_scale",
    "median_normalized_asymmetry",
    "q25_normalized_asymmetry",
    "q75_normalized_asymmetry",
    "symmetric_rate",
    "upper_farther_rate",
    "lower_farther_rate",
    "median_asymmetry",
    "q25_asymmetry",
    "q75_asymmetry",
    "median_absolute_asymmetry",
    "q75_absolute_asymmetry",
    "d_plus_lower_boundary_rate",
    "d_plus_upper_boundary_rate",
    "d_minus_lower_boundary_rate",
    "d_minus_upper_boundary_rate",
]

THRESHOLD_ROW_DERIVED_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "window_id",
    "pair_id",
    "threshold_row",
    "threshold_scale",
    "d_plus",
    "d_minus",
    "d_plus_over_scale",
    "d_minus_over_scale",
    "threshold_difference_over_scale",
    "asymmetry",
    "absolute_asymmetry",
    "entry_width",
    "entry_width_over_scale",
    "symmetric",
    "upper_farther",
    "lower_farther",
]

MODEL_DAILY_DIFFERENCE_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "boundary_policy",
    "selection_scope",
    "comparison_family",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
    "baseline_model",
    "comparison_model",
    "date",
    "baseline_daily_return",
    "comparison_daily_return",
    "daily_return_difference",
]


MODEL_COMPARISON_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "boundary_policy",
    "selection_scope",
    "comparison_family",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
    "baseline_model",
    "comparison_model",
    "number_of_matched_days",
    "mean_daily_return_difference",
    "median_daily_return_difference",
    "test_period_return_difference",
    "annualized_return_difference",
    "newey_west_tstat_difference",
    "newey_west_pvalue_difference",
]

COST_IMPACT_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "boundary_policy",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "basis_1",
    "basis_2",
    "number_of_matched_days",
    "test_period_return_basis_1",
    "test_period_return_basis_2",
    "test_period_return_difference",
    "annualized_return_basis_1",
    "annualized_return_basis_2",
    "annualized_return_difference",
    "mean_daily_return_difference",
    "median_daily_return_difference",
]

SELECTION_COMPARISON_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "row_type",
    "window_id",
    "unrestricted_pair_count",
    "adf_pair_count",
    "overlap_count",
    "unrestricted_retained_rate",
    "jaccard_similarity",
    "unrestricted_pairs_excluded_count",
    "unrestricted_pairs_excluded_rate",
    "adf_windows_with_fewer_than_10_pairs",
    "mean_adf_selected_pairs",
    "proportion_of_adf_windows_with_fewer_than_10_pairs",
    "mean_overlap_count",
    "mean_unrestricted_retained_rate",
    "mean_jaccard_similarity",
    "mean_unrestricted_excluded_rate",
]

BOUNDARY_DIAGNOSTIC_COLUMNS = [
    "case",
    "sector",
    "year",
    "aggregation_mode",
    "boundary_policy",
    "selection_scope",
    "source_backtest",
    "comparison_family",
    "model",
    "optimization_cost_case",
    "gamma_multiplier",
    "return_basis",
    "required_live_vintages",
    "first_available_date",
    "last_available_date",
    "first_complete_date",
    "last_complete_date",
    "dates_excluded_at_start",
    "dates_excluded_at_end",
    "retained_complete_dates",
    "minimum_live_vintages_before_filter",
    "maximum_live_vintages_before_filter",
    "minimum_live_vintages_after_filter",
    "maximum_live_vintages_after_filter",
    "complete_committee_validation_passed",
    "retained_sample_contiguous_on_actual_calendar",
    "every_retained_date_has_required_unique_windows",
    "first_ten_unfiltered_live_counts",
    "final_ten_unfiltered_live_counts",
]


@dataclass(frozen=True)
class SummaryConfig:
    sector: str
    year: str
    model: str
    selection_scope: str
    outputs_root: Path
    return_basis: str
    pair_slots: int
    trading_window_days: int
    annualisation_days: int
    hac_lags: int
    boundary_policy: str = "not_applicable"
    data_path: Path | None = None
    fama_french_factors_path: Path = DEFAULT_FAMA_FRENCH_DAILY_FACTORS_PATH
    reconciliation_tolerance: float = 1e-6
    cli_args: dict[str, Any] | None = None


def canonical_case(sector: str, year: str) -> str:
    raw = sector.strip().lower().replace("-", " ")
    sector_aliases = {
        "communication": "communication",
        "communications": "communication",
        "communication_services": "communication",
        "communication services": "communication",
        "comm": "communication",
        "energy": "energy",
    }
    normalized = raw.replace(" ", "_") if raw not in sector_aliases else raw
    sector_key = sector_aliases.get(normalized, normalized.replace(" ", "_"))
    return f"{sector_key}_{str(year).strip()}"


def normalize_key_value(value: Any) -> Any:
    if value is None:
        return GROUP_SENTINEL
    try:
        if pd.isna(value):
            return GROUP_SENTINEL
    except (TypeError, ValueError):
        pass
    return value


def normalize_group_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = GROUP_SENTINEL
        out[col] = out[col].map(normalize_key_value)
    return out


def group_key_tuple(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(normalize_key_value(value) for value in values)


def comparison_family_from_source(source_backtest: str, selection_scope: str) -> str:
    family = str(source_backtest).lower()
    replacements = [
        "gaussian_top10",
        "adf_capped10",
        "symmetric_bg_wu",
        "symmetric_bg",
        "nig_fixed_mean_comparable",
        "nig_fixed_mean",
        "cgmy_asymmetric_corrected_simulation_repaired",
        "cgmy_asymmetric_maxiter20_checkpointed",
        "cgmy_asymmetric",
        "gaussian_fixed_sigma_eq",
        "formation_mean_std",
        "fixed_sigma_eq",
        "zeng_lee_gaussian_conventional",
        "zeng_lee_gaussian_new",
        "zeng_lee_gaussian",
        "zeng_lee",
        "bertram_gaussian",
        "bertram",
        "gaussian_ou",
        "gaussian",
        "levy",
    ]
    for token in replacements:
        family = family.replace(token, "{family}")
    while "{family}_{family}" in family:
        family = family.replace("{family}_{family}", "{family}")
    family = family.strip("_")
    if not family:
        family = f"{selection_scope}_{GROUP_SENTINEL}"
    return family


def canonical_model(model: str) -> str:
    key = str(model).strip().lower()
    if key not in MODEL_ALIASES:
        allowed = ", ".join(sorted(MODEL_ALIASES))
        raise SystemExit(f"Unknown model {model!r}. Known aliases: {allowed}")
    return MODEL_ALIASES[key]


def canonical_return_basis(value: str | None) -> str:
    key = str(value or "bid_ask").strip().lower()
    if key not in RETURN_ALIASES:
        allowed = ", ".join(["all", *RETURN_BASIS_TO_PROFIT_COL])
        raise SystemExit(f"Unknown return basis {value!r}. Use one of: {allowed}")
    return RETURN_ALIASES[key]


def requested_return_bases(value: str) -> list[str]:
    basis = canonical_return_basis(value)
    return list(RETURN_BASIS_TO_PROFIT_COL) if basis == "all" else [basis]


def pair_id(a: Any, b: Any) -> str:
    left, right = sorted([str(a).upper(), str(b).upper()])
    return f"{left}__{right}"


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def safe_mean(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def safe_median(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.median()) if len(arr) else float("nan")


def quantile(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    return float(arr.quantile(q)) if len(arr) else float("nan")


def product_return(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    return float(np.prod(1.0 + arr) - 1.0)


def additive_test_period_return(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return float(arr.sum()) if len(arr) else float("nan")


def additive_wealth(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return 1.0 + np.cumsum(arr)


def additive_annualized_return(final_wealth: float, n_days: int, annualisation_days: int) -> float:
    if n_days <= 0 or not np.isfinite(final_wealth) or final_wealth <= 0:
        return float("nan")
    return float(final_wealth ** (annualisation_days / n_days) - 1.0)


def linear_additive_annualized_return(test_period_return: float, n_days: int, annualisation_days: int) -> float:
    if n_days <= 0 or not np.isfinite(test_period_return):
        return float("nan")
    return float(test_period_return * annualisation_days / n_days)


def hac_mean_test(values: Iterable[float], lags: int) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < 2 or np.nanstd(arr, ddof=1) <= 0:
        return float("nan"), float("nan")
    try:
        import statsmodels.api as sm

        x = np.ones((len(arr), 1), dtype=float)
        result = sm.OLS(arr, x).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
        return float(result.tvalues[0]), float(result.pvalues[0])
    except Exception:
        # Normal approximation fallback for environments without statsmodels.
        mean = float(np.mean(arr))
        centered = arr - mean
        lag = min(int(lags), len(arr) - 1)
        gamma0 = float(np.dot(centered, centered) / len(arr))
        long_var = gamma0
        for k in range(1, lag + 1):
            cov = float(np.dot(centered[k:], centered[:-k]) / len(arr))
            long_var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
        se = math.sqrt(max(long_var, 0.0) / len(arr))
        if se <= 0:
            return float("nan"), float("nan")
        tstat = mean / se
        pvalue = math.erfc(abs(tstat) / math.sqrt(2.0))
        return float(tstat), float(pvalue)


def load_fama_french_daily_rf(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Fama-French daily factors file not found: {path}")
    lines = path.read_text().splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if "Mkt-RF" in line and "SMB" in line and "HML" in line and "RF" in line:
            header_idx = idx
            break
    if header_idx is None:
        raise SystemExit(f"Could not find Fama-French factor header in {path}")
    frame = pd.read_csv(path, skiprows=header_idx)
    first_col = frame.columns[0]
    frame = frame.rename(columns={first_col: "ff_date"})
    frame["ff_date"] = frame["ff_date"].astype(str).str.strip()
    frame = frame[frame["ff_date"].str.fullmatch(r"\d{8}", na=False)].copy()
    frame["date"] = pd.to_datetime(frame["ff_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    frame["risk_free_daily_fama_french"] = pd.to_numeric(frame["RF"], errors="coerce") / 100.0
    frame = frame.dropna(subset=["date", "risk_free_daily_fama_french"])
    if frame.empty:
        raise SystemExit(f"No usable daily RF rows found in {path}")
    return frame[["date", "risk_free_daily_fama_french"]].drop_duplicates("date").reset_index(drop=True)


def infer_data_path(case: str, case_dir: Path, backtest_dirs: list[Path], explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise SystemExit(f"Explicit --data-path does not exist: {explicit}")

    candidates: list[Path] = []
    for directory in backtest_dirs:
        for name in ("run_summary.json", "repair_summary.json"):
            path = directory / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if "data_path" in payload:
                candidates.append(Path(str(payload["data_path"])))
            settings = payload.get("settings", {})
            if isinstance(settings, dict) and "data_path" in settings:
                candidates.append(Path(str(settings["data_path"])))

    root = case_dir.parents[1] if case_dir.parent.name == "outputs" else Path.cwd()
    fallback = {
        "energy_2008": root / "data/processed_lobster_energy_2008/lobster_minute_prices_model_ready.csv.gz",
        "energy_2024": root / "data/processed_lobster_energy/lobster_minute_prices_model_ready.csv.gz",
        "communication_2024": root / "data/processed_lobster_communication_2024/lobster_minute_prices_model_ready.csv.gz",
        "communication_2023": root
        / "data/processed_lobster_communication_services/lobster_minute_prices_model_ready.csv.gz",
    }
    if case in fallback:
        candidates.append(fallback[case])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not find the quote panel needed for daily mark-to-market reconstruction. "
        f"Checked: {[str(path) for path in candidates]}. Pass --data-path explicitly."
    )


def selection_scope_from_path(path: Path, case_dir: Path) -> str:
    rel = path.relative_to(case_dir)
    return "adf_capped10" if rel.parts and rel.parts[0] == "adf_capped10" else "gaussian_top10"


def find_backtest_dirs(case_dir: Path, selection_scope: str) -> list[Path]:
    roots: list[Path] = []
    if selection_scope in {"all", "gaussian_top10"}:
        roots.append(case_dir / "backtests")
    if selection_scope in {"all", "adf_capped10"}:
        roots.append(case_dir / "adf_capped10" / "backtests")

    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for profits in sorted(root.rglob("profits.csv")):
            if profits.with_name("trades.csv").exists():
                dirs.append(profits.parent)
    return dirs


def add_keys(frame: pd.DataFrame, source_backtest: str, selection_scope: str) -> pd.DataFrame:
    out = frame.copy()
    out["selection_scope"] = selection_scope
    out["source_backtest"] = source_backtest
    out["comparison_family"] = comparison_family_from_source(source_backtest, selection_scope)
    if "model" in out.columns:
        out["model"] = out["model"].astype(str).str.lower().map(lambda value: MODEL_ALIASES.get(value, value))
    out["pair_id"] = [pair_id(a, b) for a, b in zip(out["ticker_a"], out["ticker_b"])]
    if "gamma_multiplier" in out.columns:
        out["gamma_multiplier"] = pd.to_numeric(out["gamma_multiplier"], errors="coerce")
    if "threshold_row" in out.columns:
        out["threshold_row"] = pd.to_numeric(out["threshold_row"], errors="coerce")
    return normalize_group_columns(out, GROUP_KEY_COLUMNS)


def model_filter(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    if model == "all" or frame.empty or "model" not in frame.columns:
        return frame
    canonical = frame["model"].astype(str).str.lower().map(lambda value: MODEL_ALIASES.get(value, value))
    if model == "zeng_lee_gaussian":
        return frame[canonical.astype(str).str.startswith("zeng_lee_gaussian")]
    return frame[canonical == model]


def load_backtests(config: SummaryConfig, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    case = canonical_case(config.sector, config.year)
    case_dir = config.outputs_root / case
    if not case_dir.exists():
        raise SystemExit(f"Could not find cleaned output folder: {case_dir}")

    dirs = find_backtest_dirs(case_dir, config.selection_scope)
    if not dirs:
        raise SystemExit(f"No backtest folders found under {case_dir} for selection_scope={config.selection_scope}.")

    profit_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for directory in dirs:
        scope = selection_scope_from_path(directory, case_dir)
        source = directory.name
        profits = pd.read_csv(directory / "profits.csv", low_memory=False)
        trades = pd.read_csv(directory / "trades.csv", low_memory=False)
        profits = add_keys(profits, source, scope)
        trades = add_keys(trades, source, scope) if not trades.empty else trades
        profits = model_filter(profits, config.model)
        trades = model_filter(trades, config.model) if not trades.empty else trades
        if not profits.empty:
            profit_frames.append(profits)
            trade_frames.append(trades)

    if not profit_frames:
        raise SystemExit("No profit rows remain after model/window filtering.")
    profits_all = pd.concat(profit_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return profits_all, trades_all, dirs


def find_adf_audit_path(case: str, case_dir: Path) -> Path | None:
    candidates = [
        case_dir / "adf_capped10/adf_tests/adf_all_test_outcomes.csv",
        case_dir / "adf_capped10/adf_tests/stationarity_adf_all_pairs/adf_all_pair_windows.csv",
        Path("outputs") / case / "adf_capped10/adf_tests/adf_all_test_outcomes.csv",
        Path("outputs") / case / "adf_capped10/adf_tests/stationarity_adf_all_pairs/adf_all_pair_windows.csv",
        Path("outputs") / "stationarity" / f"{case}_all_pairs/adf_all_pair_windows.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_zero_adf_windows(config: SummaryConfig, mode: str, case_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    if config.selection_scope not in {"all", "adf_capped10"}:
        return pd.DataFrame(), []
    case = canonical_case(config.sector, config.year)
    path = find_adf_audit_path(case, case_dir)
    if path is None:
        return pd.DataFrame(), ["ADF cash-vintage audit skipped: no all-pair ADF audit file found."]

    audit = pd.read_csv(path, low_memory=False)
    pass_col = "adf_pass_5pct" if "adf_pass_5pct" in audit.columns else "adf_pass"
    required = {"window_id", "trading_start", "trading_end", pass_col}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise SystemExit(f"ADF audit file {path} is missing columns required for cash-vintage audit: {missing}")

    audit["_adf_pass"] = truthy(audit[pass_col])
    window_meta = (
        audit.sort_values("window_id")
        .groupby("window_id", dropna=False)
        .agg(
            adf_pass_count=("_adf_pass", "sum"),
            trading_start=("trading_start", "first"),
            trading_end=("trading_end", "first"),
        )
        .reset_index()
    )
    window_meta["window_id"] = pd.to_numeric(window_meta["window_id"], errors="coerce")
    window_meta = window_meta.dropna(subset=["window_id"]).copy()
    window_meta["window_id"] = window_meta["window_id"].astype(int)
    zero = window_meta[window_meta["adf_pass_count"].astype(int).eq(0)].copy()
    return zero[["window_id", "trading_start", "trading_end", "adf_pass_count"]], [
        f"ADF cash-vintage audit loaded from {path}; zero-selected windows={len(zero)}."
    ]


def load_threshold_rows(
    config: SummaryConfig,
    mode: str,
    backtest_dirs: list[Path],
) -> tuple[pd.DataFrame, list[str]]:
    """Load individual threshold selections and add row-level diagnostics.

    The output retains every original thresholds.csv column and adds the
    pair-window quantities needed for threshold-distribution plots.
    """

    case = canonical_case(config.sector, config.year)
    case_dir = config.outputs_root / case
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []

    for directory in backtest_dirs:
        path = directory / "thresholds.csv"
        if not path.exists():
            warnings.append(f"thresholds.csv missing for {directory}; raw threshold rows not written for this source")
            continue

        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        if not {"ticker_a", "ticker_b"}.issubset(frame.columns):
            warnings.append(f"thresholds.csv missing ticker_a/ticker_b for {directory}; source skipped")
            continue

        scope = selection_scope_from_path(directory, case_dir)
        frame = add_keys(frame, directory.name, scope)
        frame = model_filter(frame, config.model)
        if frame.empty:
            continue

        valid = truthy(frame["threshold_valid"]) if "threshold_valid" in frame.columns else pd.Series(True, index=frame.index)
        frame = frame[valid].copy()
        if frame.empty:
            continue

        d_plus = pd.to_numeric(frame.get("d_plus", pd.Series(np.nan, index=frame.index)), errors="coerce")
        d_minus = pd.to_numeric(frame.get("d_minus", pd.Series(np.nan, index=frame.index)), errors="coerce")
        scale = pd.to_numeric(frame.get("threshold_scale", pd.Series(np.nan, index=frame.index)), errors="coerce")
        if scale.isna().all() and "d_plus_sigma" in frame.columns:
            d_plus_sigma = pd.to_numeric(frame["d_plus_sigma"], errors="coerce")
            scale = d_plus / d_plus_sigma.replace(0, np.nan)

        frame["threshold_scale"] = scale
        frame["d_plus"] = d_plus
        frame["d_minus"] = d_minus
        frame["d_plus_over_scale"] = d_plus / scale.replace(0, np.nan)
        frame["d_minus_over_scale"] = d_minus / scale.replace(0, np.nan)
        frame["threshold_difference_over_scale"] = (d_plus - d_minus) / scale.replace(0, np.nan)

        denominator = d_plus + d_minus
        frame["asymmetry"] = np.where(
            denominator > 0.0,
            (d_plus - d_minus) / denominator,
            np.nan,
        )
        frame["absolute_asymmetry"] = pd.to_numeric(frame["asymmetry"], errors="coerce").abs()
        frame["entry_width"] = (d_plus + d_minus) / 2.0
        frame["entry_width_over_scale"] = (
            frame["d_plus_over_scale"] + frame["d_minus_over_scale"]
        ) / 2.0

        tol = 1e-8
        plus_over = pd.to_numeric(frame["d_plus_over_scale"], errors="coerce")
        minus_over = pd.to_numeric(frame["d_minus_over_scale"], errors="coerce")
        frame["symmetric"] = np.isclose(plus_over, minus_over, atol=tol, rtol=0.0)
        frame["upper_farther"] = plus_over > minus_over + tol
        frame["lower_farther"] = minus_over > plus_over + tol

        frame["case"] = case
        frame["sector"] = config.sector
        frame["year"] = str(config.year)
        frame["aggregation_mode"] = mode
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=THRESHOLD_ROW_DERIVED_COLUMNS), warnings

    out = pd.concat(frames, ignore_index=True, sort=False)
    leading = [col for col in THRESHOLD_ROW_DERIVED_COLUMNS if col in out.columns]
    remaining = [col for col in out.columns if col not in leading]
    sort_cols = [
        col
        for col in [
            "selection_scope",
            "source_backtest",
            "model",
            "optimization_cost_case",
            "gamma_multiplier",
            "window_id",
            "pair_id",
            "threshold_row",
        ]
        if col in out.columns
    ]
    return out[leading + remaining].sort_values(sort_cols).reset_index(drop=True), warnings


def validate_inputs(profits: pd.DataFrame, trades: pd.DataFrame) -> list[str]:
    diagnostics: list[str] = []
    valid = profits if "threshold_valid" not in profits.columns else profits[truthy(profits["threshold_valid"])]
    key_cols = [
        "selection_scope",
        "source_backtest",
        "comparison_family",
        "model",
        "optimization_cost_case",
        "gamma_multiplier",
        "window_id",
        "pair_id",
        "threshold_row",
    ]
    dup = valid.duplicated([col for col in key_cols if col in valid.columns], keep=False)
    if dup.any():
        sample = valid.loc[dup, [col for col in key_cols if col in valid.columns]].head(5).to_dict("records")
        raise SystemExit(f"Duplicated pair-window keys in profits.csv; sample={sample}")

    if not trades.empty:
        trade_key_cols = [col for col in [*key_cols, "trade_id"] if col in trades.columns]
        dup_trades = trades.duplicated(trade_key_cols, keep=False)
        if dup_trades.any():
            sample = trades.loc[dup_trades, trade_key_cols].head(5).to_dict("records")
            raise SystemExit(f"Duplicated trade identifiers in trades.csv; sample={sample}")

        entry = pd.to_datetime(trades["entry_trade_date"], errors="coerce")
        exit_ = pd.to_datetime(trades["exit_trade_date"], errors="coerce")
        start = pd.to_datetime(trades["trading_start"], errors="coerce")
        end = pd.to_datetime(trades["trading_end"], errors="coerce")
        outside = (entry < start) | (entry > end) | (exit_ < start) | (exit_ > end)
        if outside.fillna(False).any():
            raise SystemExit("Some trade dates lie outside their stated trading window.")

    diagnostics.append("input key validation passed")
    return diagnostics


def load_daily_quote_cache(
    profits: pd.DataFrame,
    data_path: Path,
) -> tuple[dict[tuple[Any, ...], pd.DataFrame], list[str], list[dict[str, Any]], list[str]]:
    needed = profits[["ticker_a", "ticker_b", "formation_start", "formation_end", "trading_start", "trading_end"]].drop_duplicates()
    tickers = sorted(set(needed["ticker_a"].astype(str).str.upper()) | set(needed["ticker_b"].astype(str).str.upper()))
    start = str(needed["formation_start"].min())
    end = str(needed["trading_end"].max())
    panel = load_lobster_panel(data_path, tickers=tickers, start_date=start, end_date=end)
    if panel.empty:
        raise SystemExit(f"Quote panel load returned no rows from {data_path}")
    panel["trade_date_str"] = panel["trade_date"].astype(str)
    actual_calendar = sorted(pd.to_datetime(panel["trade_date_str"]).dt.strftime("%Y-%m-%d").unique())
    ticker_frames = {
        ticker: group.sort_values("timestamp_utc").reset_index(drop=True)
        for ticker, group in panel.groupby("ticker", sort=False)
    }

    cache: dict[tuple[Any, ...], pd.DataFrame] = {}
    skipped: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in needed.itertuples(index=False):
        ticker_a = str(row.ticker_a).upper()
        ticker_b = str(row.ticker_b).upper()
        key = (
            ticker_a,
            ticker_b,
            str(row.formation_start),
            str(row.formation_end),
            str(row.trading_start),
            str(row.trading_end),
        )
        if ticker_a not in ticker_frames or ticker_b not in ticker_frames:
            msg = f"Quote panel is missing ticker(s) for {ticker_a}-{ticker_b}."
            warnings.append(msg)
            skipped.append(
                {
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "formation_start": str(row.formation_start),
                    "formation_end": str(row.formation_end),
                    "trading_start": str(row.trading_start),
                    "trading_end": str(row.trading_end),
                    "reason": msg,
                    "quote_key": key,
                }
            )
            continue
        formation = _pivot_pair_from_ticker_frames(
            ticker_frames[ticker_a],
            ticker_frames[ticker_b],
            row.formation_start,
            row.formation_end,
        )
        if formation.empty:
            trading = pd.DataFrame()
        else:
            trading = _pivot_pair_from_ticker_frames(
                ticker_frames[ticker_a],
                ticker_frames[ticker_b],
                row.trading_start,
                row.trading_end,
                anchor_a=float(formation["price_a"].iloc[0]),
                anchor_b=float(formation["price_b"].iloc[0]),
            )
        if trading.empty:
            msg = (
                "Could not reconstruct trading quotes for "
                f"{row.ticker_a}-{row.ticker_b}, {row.trading_start} to {row.trading_end}."
            )
            warnings.append(msg)
            skipped.append(
                {
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "formation_start": str(row.formation_start),
                    "formation_end": str(row.formation_end),
                    "trading_start": str(row.trading_start),
                    "trading_end": str(row.trading_end),
                    "reason": msg,
                    "quote_key": key,
                }
            )
            continue
        daily = (
            trading.sort_values("timestamp_utc")
            .groupby("trade_date", as_index=False)
            .tail(1)
            .copy()
            .sort_values("trade_date")
        )
        daily["date"] = pd.to_datetime(daily["trade_date"]).dt.strftime("%Y-%m-%d")
        keep = ["date", "mid_a", "mid_b", "bid_a", "ask_a", "bid_b", "ask_b"]
        cache[key] = daily[keep].reset_index(drop=True)
    return cache, actual_calendar, skipped, warnings


def quote_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        str(row["ticker_a"]).upper(),
        str(row["ticker_b"]).upper(),
        str(row["formation_start"]),
        str(row["formation_end"]),
        str(row["trading_start"]),
        str(row["trading_end"]),
    )


def trade_lookup(trades: pd.DataFrame) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    if trades.empty:
        return {}
    key_cols = [
        "selection_scope",
        "source_backtest",
        "comparison_family",
        "model",
        "optimization_cost_case",
        "gamma_multiplier",
        "window_id",
        "pair_id",
        "threshold_row",
    ]
    lookup: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    record_cols = [
        "entry_i",
        "exit_i",
        "trade_id",
        "entry_trade_date",
        "exit_trade_date",
        "direction",
        "entry_mid_a",
        "entry_mid_b",
        "entry_exec_a",
        "entry_exec_b",
        "fixed_bps_entry_cost_pnl_per_dollar_leg",
        "bid_ask_return_on_gross",
        "midquote_fixed_bps_return_on_gross",
        "midquote_return_on_gross",
        "forced_exit",
    ]
    available_record_cols = [col for col in record_cols if col in trades.columns]
    compact = trades[[*key_cols, *available_record_cols]].copy()
    sort_cols = [col for col in [*key_cols, "entry_i", "exit_i", "trade_id"] if col in compact.columns]
    if sort_cols:
        compact = compact.sort_values(sort_cols, na_position="last")
    columns = list(compact.columns)
    key_indices = [columns.index(col) for col in key_cols]
    record_indices = [columns.index(col) for col in available_record_cols]
    for values in compact.itertuples(index=False, name=None):
        key = group_key_tuple(values[idx] for idx in key_indices)
        record = {col: values[idx] for col, idx in zip(available_record_cols, record_indices)}
        lookup.setdefault(key, []).append(record)
    return lookup


def profit_key(row: pd.Series) -> tuple[Any, ...]:
    return group_key_tuple((
        row["selection_scope"],
        row["source_backtest"],
        row["comparison_family"],
        row["model"],
        row["optimization_cost_case"],
        row["gamma_multiplier"],
        row["window_id"],
        row["pair_id"],
        row["threshold_row"],
    ))


def validate_fixed_bps_execution_costs(trades: pd.DataFrame, tolerance: float = 1e-10) -> list[str]:
    if trades.empty or "midquote_fixed_bps_return_on_gross" not in trades.columns:
        return ["fixed-bps validation skipped: no fixed-bps trade rows available"]
    required = [
        "fixed_bps_half_turn_rate",
        "fixed_bps_entry_cost_pnl_per_dollar_leg",
        "fixed_bps_exit_cost_pnl_per_dollar_leg",
        "fixed_bps_cost_pnl_per_dollar_leg",
        "fixed_bps_cost_return_on_gross",
        "entry_mid_a",
        "entry_mid_b",
        "exit_mid_a",
        "exit_mid_b",
    ]
    missing = [col for col in required if col not in trades.columns]
    if missing:
        raise SystemExit(f"Cannot validate fixed-bps execution timing; trades.csv missing columns: {missing}")
    check = trades[required].apply(pd.to_numeric, errors="coerce")
    rate = check["fixed_bps_half_turn_rate"]
    expected_entry = rate * 2.0
    expected_exit = rate * (check["exit_mid_a"] / check["entry_mid_a"] + check["exit_mid_b"] / check["entry_mid_b"])
    expected_total = expected_entry + expected_exit
    failures = (
        (check["fixed_bps_entry_cost_pnl_per_dollar_leg"] - expected_entry).abs().gt(tolerance)
        | (check["fixed_bps_exit_cost_pnl_per_dollar_leg"] - expected_exit).abs().gt(tolerance)
        | (check["fixed_bps_cost_pnl_per_dollar_leg"] - expected_total).abs().gt(tolerance)
        | (check["fixed_bps_cost_return_on_gross"] - expected_total / 2.0).abs().gt(tolerance)
    )
    if failures.fillna(True).any():
        sample = trades.loc[failures.fillna(True), ["source_backtest", "window_id", "pair_id", "trade_id"]].head(5).to_dict("records")
        raise SystemExit(f"Fixed-bps execution cost validation failed; sample={sample}")
    return [
        "fixed-bps cost timing validated: entry cost allocated at entry, exit cost allocated at exit; "
        "entry_cost=half_turn_rate*2, exit_cost=half_turn_rate*(exit_mid_a/entry_mid_a + exit_mid_b/entry_mid_b), "
        "all converted to return_on_gross by dividing costs by 2"
    ]


def cumulative_trade_return(trade: dict[str, Any], quote: dict[str, Any], basis: str, final: bool) -> float:
    if final:
        return finite_float(trade[RETURN_BASIS_TO_TRADE_COL[basis]], 0.0)

    direction_a, direction_b = spread_leg_directions(str(trade["direction"]))
    if basis == "midquote" or basis == "midquote_fixed_bps":
        ret = direction_a * (finite_float(quote["mid_a"]) / finite_float(trade["entry_mid_a"]) - 1.0)
        ret += direction_b * (finite_float(quote["mid_b"]) / finite_float(trade["entry_mid_b"]) - 1.0)
        ret /= 2.0
        if basis == "midquote_fixed_bps":
            ret -= finite_float(trade.get("fixed_bps_entry_cost_pnl_per_dollar_leg", 0.0)) / 2.0
        return float(ret)

    if basis == "bid_ask":
        if str(trade["direction"]) == "long_spread":
            exit_a = finite_float(quote["bid_a"])
            exit_b = finite_float(quote["ask_b"])
        else:
            exit_a = finite_float(quote["ask_a"])
            exit_b = finite_float(quote["bid_b"])
        ret = direction_a * (exit_a / finite_float(trade["entry_exec_a"]) - 1.0)
        ret += direction_b * (exit_b / finite_float(trade["entry_exec_b"]) - 1.0)
        return float(ret / 2.0)
    raise ValueError(f"Unknown return basis: {basis}")


def pair_daily_returns(
    profit_row: pd.Series,
    trades: pd.DataFrame | list[dict[str, Any]],
    daily_quotes: pd.DataFrame,
    return_bases: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    dates = list(daily_quotes["date"].astype(str))
    date_index = {date: idx for idx, date in enumerate(dates)}
    returns_by_basis = {basis: np.zeros(len(dates), dtype=float) for basis in return_bases}
    num_open_positions = np.zeros(len(dates), dtype=int)
    num_trades_opened = np.zeros(len(dates), dtype=int)
    num_trades_closed = np.zeros(len(dates), dtype=int)

    quote_columns = [col for col in ["date", "mid_a", "mid_b", "bid_a", "ask_a", "bid_b", "ask_b"] if col in daily_quotes.columns]
    quote_records = [
        dict(zip(quote_columns, values))
        for values in daily_quotes[quote_columns].itertuples(index=False, name=None)
    ]
    trade_records = trades.to_dict("records") if isinstance(trades, pd.DataFrame) else trades
    if trade_records:
        for trade_s in trade_records:
            entry_date = str(trade_s["entry_trade_date"])
            exit_date = str(trade_s["exit_trade_date"])
            entry_idx = date_index.get(entry_date)
            exit_idx = date_index.get(exit_date)
            if entry_idx is None or exit_idx is None or exit_idx < entry_idx:
                continue
            previous = {basis: 0.0 for basis in return_bases}
            for idx in range(entry_idx, exit_idx + 1):
                date = dates[idx]
                quote = quote_records[idx]
                is_final = date == exit_date
                num_open_positions[idx] += 1
                if date == entry_date:
                    num_trades_opened[idx] += 1
                if is_final:
                    num_trades_closed[idx] += 1
                for basis in return_bases:
                    current = cumulative_trade_return(trade_s, quote, basis, final=is_final)
                    returns_by_basis[basis][idx] += current - previous[basis]
                    previous[basis] = current

    out_data: dict[str, Any] = {"date": dates}
    for basis in return_bases:
        out_data[f"{basis}_pair_return"] = returns_by_basis[basis]
    out_data["num_open_positions"] = num_open_positions
    out_data["num_trades_opened"] = num_trades_opened
    out_data["num_trades_closed"] = num_trades_closed
    out = pd.DataFrame(out_data)

    expected = {
        basis: finite_float(profit_row.get(RETURN_BASIS_TO_PROFIT_COL[basis], 0.0), 0.0)
        for basis in return_bases
    }
    return out, expected


def build_vintage_daily(
    profits: pd.DataFrame,
    trades: pd.DataFrame,
    quote_cache: dict[tuple[Any, ...], pd.DataFrame],
    return_bases: list[str],
    pair_slots: int,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    valid = profits if "threshold_valid" not in profits.columns else profits[truthy(profits["threshold_valid"])].copy()
    lookup = trade_lookup(trades)
    rows: list[pd.DataFrame] = []
    id_records: list[dict[str, Any]] = []
    reconciliation_messages: list[str] = []
    trade_count_errors: list[str] = []
    forced_count_errors: list[str] = []
    skipped_quote_rows = 0

    valid_columns = list(valid.columns)
    for profit_tuple in valid.itertuples(index=False, name=None):
        profit = dict(zip(valid_columns, profit_tuple))
        key = group_key_tuple(
            (
                profit["selection_scope"],
                profit["source_backtest"],
                profit["comparison_family"],
                profit["model"],
                profit["optimization_cost_case"],
                profit["gamma_multiplier"],
                profit["window_id"],
                profit["pair_id"],
                profit["threshold_row"],
            )
        )
        selected_trades = lookup.get(key, [])
        q_key = (
            str(profit["ticker_a"]).upper(),
            str(profit["ticker_b"]).upper(),
            str(profit["formation_start"]),
            str(profit["formation_end"]),
            str(profit["trading_start"]),
            str(profit["trading_end"]),
        )
        daily_quotes = quote_cache.get(q_key)
        if daily_quotes is None:
            skipped_quote_rows += 1
            continue
        pair_daily, expected = pair_daily_returns(profit, selected_trades, daily_quotes, return_bases)
        for basis in return_bases:
            actual = float(pair_daily[f"{basis}_pair_return"].sum())
            if abs(actual - expected[basis]) > tolerance:
                reconciliation_messages.append(
                    f"{profit['source_backtest']} window={profit['window_id']} pair={profit['pair_id']} "
                    f"basis={basis} expected={expected[basis]:.12g} actual={actual:.12g}"
                )
        if "num_trades" in profit:
            expected_trades = int(round(finite_float(profit["num_trades"], 0.0)))
            if expected_trades != len(selected_trades):
                trade_count_errors.append(
                    f"{profit['source_backtest']} {profit['window_id']} {profit['pair_id']} "
                    f"expected={expected_trades} actual={len(selected_trades)}"
                )
        if "forced_exits" in profit:
            expected_forced = int(round(finite_float(profit["forced_exits"], 0.0)))
            actual_forced = int(
                sum(
                    1
                    for trade in selected_trades
                    if str(trade.get("forced_exit", "")).strip().lower() in {"1", "1.0", "true", "yes", "y"}
                )
            )
            if expected_forced != actual_forced:
                forced_count_errors.append(
                    f"{profit['source_backtest']} {profit['window_id']} {profit['pair_id']} "
                    f"expected={expected_forced} actual={actual_forced}"
                )

        id_cols = {
            "selection_scope": profit["selection_scope"],
            "source_backtest": profit["source_backtest"],
            "comparison_family": profit["comparison_family"],
            "model": profit["model"],
            "optimization_cost_case": profit["optimization_cost_case"],
            "gamma_multiplier": profit["gamma_multiplier"],
            "window_id": profit["window_id"],
            "pair_id": profit["pair_id"],
            "trading_start_date": str(profit["trading_start"]),
            "trading_end_date": str(profit["trading_end"]),
        }
        group_id = len(id_records)
        id_records.append({"_result_group_id": group_id, **id_cols})
        pair_daily = pair_daily.copy()
        pair_daily["_result_group_id"] = group_id
        rows.append(pair_daily)

    if reconciliation_messages:
        raise SystemExit(
            "Daily marked-to-market returns do not reconcile to final backtest P&L. "
            f"First mismatches: {reconciliation_messages[:10]}"
        )
    if trade_count_errors:
        raise SystemExit(f"Trade-count reconciliation failed. First mismatches: {trade_count_errors[:10]}")
    if forced_count_errors:
        raise SystemExit(f"Forced-exit reconciliation failed. First mismatches: {forced_count_errors[:10]}")
    pair_daily_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not pair_daily_all.empty:
        pair_daily_all = pair_daily_all.merge(pd.DataFrame(id_records), on="_result_group_id", how="left")
        pair_daily_all = pair_daily_all.drop(columns=["_result_group_id"])
    vintage_rows: list[dict[str, Any]] = []
    group_cols = [
        "selection_scope",
        "source_backtest",
        "comparison_family",
        "model",
        "optimization_cost_case",
        "gamma_multiplier",
        "window_id",
        "trading_start_date",
        "trading_end_date",
    ]
    for key, group in pair_daily_all.groupby(group_cols + ["date"], dropna=False):
        key_dict = dict(zip(group_cols + ["date"], key))
        for basis in return_bases:
            vintage_rows.append(
                {
                    **key_dict,
                    "return_basis": basis,
                    "vintage_daily_return": float(group[f"{basis}_pair_return"].sum() / pair_slots),
                    "num_selected_pairs": int(group["pair_id"].nunique()),
                    "num_pairs_with_open_positions": int((group["num_open_positions"] > 0).sum()),
                    "num_trades_opened": int(group["num_trades_opened"].sum()),
                    "num_trades_closed": int(group["num_trades_closed"].sum()),
                }
            )
    vintage = pd.DataFrame(vintage_rows)
    diagnostics = ["daily P&L reconciliation passed"]
    if skipped_quote_rows:
        diagnostics.append(f"skipped quote-unavailable profit rows: {skipped_quote_rows}")
    return vintage, pair_daily_all, diagnostics


def add_zero_adf_cash_vintages(
    vintage: pd.DataFrame,
    zero_windows: pd.DataFrame,
    actual_calendar: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if vintage.empty or zero_windows.empty:
        return vintage, []
    group_cols = [
        "selection_scope",
        "source_backtest",
        "comparison_family",
        "model",
        "optimization_cost_case",
        "gamma_multiplier",
        "return_basis",
    ]
    missing_cols = [col for col in group_cols if col not in vintage.columns]
    if missing_cols:
        raise SystemExit(f"Cannot add ADF cash vintages; vintage daily frame missing columns: {missing_cols}")

    calendar = sorted(str(date) for date in actual_calendar)
    if not calendar:
        calendar = sorted(pd.to_datetime(vintage["date"]).dt.strftime("%Y-%m-%d").unique())
    zero_meta = {
        int(row.window_id): (str(row.trading_start), str(row.trading_end))
        for row in zero_windows.itertuples(index=False)
    }
    existing_by_group = {
        key: set(pd.to_numeric(group["window_id"], errors="coerce").dropna().astype(int).unique())
        for key, group in vintage.groupby(group_cols, dropna=False)
    }

    cash_rows: list[dict[str, Any]] = []
    for key, existing_windows in existing_by_group.items():
        key_dict = dict(zip(group_cols, key))
        if str(key_dict["selection_scope"]) != "adf_capped10":
            continue
        for window_id, (trading_start, trading_end) in zero_meta.items():
            if window_id in existing_windows:
                continue
            dates = calendar_between(calendar, trading_start, trading_end)
            if not dates:
                raise SystemExit(
                    "ADF zero-selected cash window has no dates on the actual trading calendar: "
                    f"window_id={window_id}, trading_start={trading_start}, trading_end={trading_end}, group={key_dict}"
                )
            for date in dates:
                cash_rows.append(
                    {
                        **key_dict,
                        "window_id": window_id,
                        "trading_start_date": trading_start,
                        "trading_end_date": trading_end,
                        "date": date,
                        "vintage_daily_return": 0.0,
                        "num_selected_pairs": 0,
                        "num_pairs_with_open_positions": 0,
                        "num_trades_opened": 0,
                        "num_trades_closed": 0,
                    }
                )
    if not cash_rows:
        return vintage, []
    out = pd.concat([vintage, pd.DataFrame(cash_rows)], ignore_index=True, sort=False)
    diagnostics = [
        f"added {len(cash_rows)} zero-return cash-vintage daily rows for "
        f"{len(zero_meta)} zero-selected ADF windows"
    ]
    return out, diagnostics


def calendar_between(calendar: list[str], start: str, end: str) -> list[str]:
    return [date for date in calendar if str(start) <= date <= str(end)]


def date_range_with_cash(dates: pd.Series, calendar: list[str] | None = None) -> list[str]:
    idx = pd.to_datetime(dates).dropna().sort_values().unique()
    if len(idx) == 0:
        return []
    start = pd.Timestamp(idx[0])
    end = pd.Timestamp(idx[-1])
    observed = sorted(pd.to_datetime(dates).dt.strftime("%Y-%m-%d").unique())
    if calendar is None:
        return observed
    return [date for date in calendar if start.strftime("%Y-%m-%d") <= date <= end.strftime("%Y-%m-%d")]


def aggregate_daily(
    vintage: pd.DataFrame,
    config: SummaryConfig,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = BASE_RESULT_GROUP_COLUMNS
    case = canonical_case(config.sector, config.year)
    actual_calendar = sorted(pd.to_datetime(vintage.attrs.get("actual_trading_calendar", [])).strftime("%Y-%m-%d"))
    if not actual_calendar:
        actual_calendar = sorted(pd.to_datetime(vintage["date"]).dt.strftime("%Y-%m-%d").unique())
    rows: list[dict[str, Any]] = []
    for key, group in vintage.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        for date, date_group in group.groupby("date", sort=True):
            live = int(date_group["window_id"].nunique())
            if mode == "overlapping":
                if config.boundary_policy == "active_average":
                    denominator = live
                else:
                    denominator = config.trading_window_days
                if denominator <= 0:
                    continue
                daily_return = float(date_group["vintage_daily_return"].sum() / denominator)
            else:
                daily_return = float(date_group["vintage_daily_return"].sum())
            rows.append(
                {
                    **key_dict,
                    "date": str(date),
                    "num_live_vintages": live,
                    "num_selected_pairs": int(date_group["num_selected_pairs"].sum()),
                    "num_pairs_with_open_positions": int(date_group["num_pairs_with_open_positions"].sum()),
                    "num_trades_opened": int(date_group["num_trades_opened"].sum()),
                    "num_trades_closed": int(date_group["num_trades_closed"].sum()),
                    "daily_return": daily_return,
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, pd.DataFrame(columns=BOUNDARY_DIAGNOSTIC_COLUMNS)

    filled_groups: list[pd.DataFrame] = []
    boundary_rows: list[dict[str, Any]] = []
    fill_zero = [
        "num_live_vintages",
        "num_selected_pairs",
        "num_pairs_with_open_positions",
        "num_trades_opened",
        "num_trades_closed",
        "daily_return",
    ]

    for key, group in daily.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        first_available = str(group["date"].min())
        last_available = str(group["date"].max())
        dates = calendar_between(actual_calendar, first_available, last_available)
        if not dates:
            raise SystemExit(f"No actual trading-calendar dates for result group {key_dict}.")
        base = pd.DataFrame({"date": dates})
        full = base.merge(group, on="date", how="left")
        for col, value in key_dict.items():
            full[col] = value
        full[fill_zero] = full[fill_zero].fillna(0)
        full["num_live_vintages"] = full["num_live_vintages"].astype(int)

        required = int(config.trading_window_days) if mode == "overlapping" else 1
        first_complete = ""
        last_complete = ""
        dates_excluded_start = 0
        dates_excluded_end = 0
        retained_complete_dates = len(full)
        validation_passed = True
        retained = full.copy()

        if mode == "overlapping" and config.boundary_policy == "full_only":
            complete_mask = full["num_live_vintages"].eq(required).to_numpy()
            complete_runs: list[tuple[int, int]] = []
            run_start: int | None = None
            for idx, is_complete in enumerate(complete_mask):
                if is_complete and run_start is None:
                    run_start = idx
                elif not is_complete and run_start is not None:
                    complete_runs.append((run_start, idx - 1))
                    run_start = None
            if run_start is not None:
                complete_runs.append((run_start, len(full) - 1))
            if not complete_runs:
                raise SystemExit(f"No full_only complete-committee dates found for result group {key_dict}.")
            calendar_midpoint = (len(full) - 1) / 2.0
            start_idx, end_idx = max(
                complete_runs,
                key=lambda item: (item[1] - item[0] + 1, -abs(((item[0] + item[1]) / 2.0) - calendar_midpoint)),
            )
            first_complete = str(full["date"].iloc[start_idx])
            last_complete = str(full["date"].iloc[end_idx])
            dates_excluded_start = int((full["date"] < first_complete).sum())
            dates_excluded_end = int((full["date"] > last_complete).sum())
            retained = full[full["date"].between(first_complete, last_complete)].copy()
            expected_dates = calendar_between(actual_calendar, first_complete, last_complete)
            if list(retained["date"]) != expected_dates:
                raise SystemExit(f"Retained full_only dates are not contiguous on the actual trading calendar for {key_dict}.")
            if not retained["num_live_vintages"].eq(required).all():
                bad = retained.loc[~retained["num_live_vintages"].eq(required), ["date", "num_live_vintages"]].head(10)
                raise SystemExit(f"Incomplete committee date inside retained full_only interval for {key_dict}: {bad.to_dict('records')}")
            retained_complete_dates = int(len(retained))
        elif mode == "overlapping" and config.boundary_policy in {"cash_padded", "active_average"}:
            first_complete = ""
            last_complete = ""
            retained = full
        else:
            retained = full

        boundary_rows.append(
            {
                "case": case,
                "sector": config.sector,
                "year": str(config.year),
                "aggregation_mode": mode,
                "boundary_policy": config.boundary_policy if mode == "overlapping" else "not_applicable",
                **key_dict,
                "required_live_vintages": required,
                "first_available_date": first_available,
                "last_available_date": last_available,
                "first_complete_date": first_complete,
                "last_complete_date": last_complete,
                "dates_excluded_at_start": dates_excluded_start,
                "dates_excluded_at_end": dates_excluded_end,
                "retained_complete_dates": retained_complete_dates,
                "minimum_live_vintages_before_filter": int(full["num_live_vintages"].min()),
                "maximum_live_vintages_before_filter": int(full["num_live_vintages"].max()),
                "minimum_live_vintages_after_filter": int(retained["num_live_vintages"].min()) if len(retained) else 0,
                "maximum_live_vintages_after_filter": int(retained["num_live_vintages"].max()) if len(retained) else 0,
                "complete_committee_validation_passed": bool(validation_passed),
                "retained_sample_contiguous_on_actual_calendar": True,
                "every_retained_date_has_required_unique_windows": bool(
                    retained["num_live_vintages"].eq(required).all()
                    if mode == "overlapping" and config.boundary_policy == "full_only"
                    else True
                ),
                "first_ten_unfiltered_live_counts": json.dumps(
                    full[["date", "num_live_vintages"]].head(10).to_dict("records")
                ),
                "final_ten_unfiltered_live_counts": json.dumps(
                    full[["date", "num_live_vintages"]].tail(10).to_dict("records")
                ),
            }
        )
        filled_groups.append(retained)

    daily = pd.concat(filled_groups, ignore_index=True)

    daily["daily_pnl_return_on_initial_committed_capital"] = daily["daily_return"].astype(float)
    if not np.isfinite(daily["daily_pnl_return_on_initial_committed_capital"].astype(float)).all():
        raise SystemExit("Daily P&L return increments must be finite.")

    boundary = pd.DataFrame(boundary_rows, columns=BOUNDARY_DIAGNOSTIC_COLUMNS)
    return daily.sort_values(group_cols + ["date"]).reset_index(drop=True), boundary


def attach_wealth(daily: pd.DataFrame, rf_daily: pd.DataFrame, config: SummaryConfig, mode: str) -> pd.DataFrame:
    if daily.empty:
        return daily
    out = daily.copy().merge(rf_daily, on="date", how="left", validate="many_to_one")
    if out["risk_free_daily_fama_french"].isna().any():
        missing = sorted(out.loc[out["risk_free_daily_fama_french"].isna(), "date"].unique().tolist())[:20]
        raise SystemExit(f"Missing Fama-French RF values for strategy dates; sample={missing}")
    group_cols = BASE_RESULT_GROUP_COLUMNS
    pieces: list[pd.DataFrame] = []
    case = canonical_case(config.sector, config.year)
    for _, group in out.groupby(group_cols, dropna=False):
        group = group.sort_values("date").copy()
        pnl = group["daily_pnl_return_on_initial_committed_capital"].astype(float)
        wealth = additive_wealth(pnl)
        running_max = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        drawdown = wealth / running_max - 1.0
        group["wealth_index"] = wealth
        group["drawdown"] = drawdown
        pieces.append(group)
    out = pd.concat(pieces, ignore_index=True)
    out["case"] = case
    out["sector"] = config.sector
    out["year"] = str(config.year)
    out["aggregation_mode"] = mode
    out["boundary_policy"] = config.boundary_policy if mode == "overlapping" else "not_applicable"
    return out[DAILY_COLUMNS].sort_values(
        [*BASE_RESULT_GROUP_COLUMNS, "date"]
    )


def performance_rows(
    daily: pd.DataFrame,
    vintage: pd.DataFrame,
    boundary: pd.DataFrame,
    config: SummaryConfig,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = BASE_RESULT_GROUP_COLUMNS
    case = canonical_case(config.sector, config.year)
    window_counts = (
        vintage.groupby(group_cols, dropna=False)["window_id"]
        .nunique()
        .rename("num_windows")
        .reset_index()
    )
    window_count_lookup = {
        tuple(row[col] for col in group_cols): int(row["num_windows"])
        for row in window_counts.to_dict("records")
    }
    boundary_lookup = {
        tuple(row[col] for col in group_cols): row
        for row in boundary.to_dict("records")
    }
    for key, group in daily.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        boundary_row = boundary_lookup.get(tuple(key), {})
        returns = group.sort_values("date")["daily_pnl_return_on_initial_committed_capital"].astype(float)
        n = int(len(returns))
        net = additive_test_period_return(returns)
        final_wealth = 1.0 + net
        ann = additive_annualized_return(final_wealth, n, config.annualisation_days)
        ann_additive = linear_additive_annualized_return(net, n, config.annualisation_days)
        excess_zero = returns
        excess_ff = returns - group.sort_values("date")["risk_free_daily_fama_french"].astype(float)
        vol = float(returns.std(ddof=1) * math.sqrt(config.annualisation_days)) if n > 1 else float("nan")
        sharpe_zero = (
            float(math.sqrt(config.annualisation_days) * excess_zero.mean() / excess_zero.std(ddof=1))
            if n > 1 and excess_zero.std(ddof=1) > 0
            else float("nan")
        )
        sharpe_ff = (
            float(math.sqrt(config.annualisation_days) * excess_ff.mean() / excess_ff.std(ddof=1))
            if n > 1 and excess_ff.std(ddof=1) > 0
            else float("nan")
        )
        tstat, pvalue = hac_mean_test(returns, config.hac_lags)
        rows.append(
            {
                "case": case,
                "sector": config.sector,
                "year": str(config.year),
                "aggregation_mode": mode,
                "boundary_policy": config.boundary_policy if mode == "overlapping" else "not_applicable",
                **key_dict,
                "sample_start_date": group["date"].min(),
                "sample_end_date": group["date"].max(),
                "num_daily_observations": n,
                "num_windows": int(window_count_lookup.get(tuple(key), 0)),
                "min_live_vintages": int(group["num_live_vintages"].min()),
                "max_live_vintages": int(group["num_live_vintages"].max()),
                "required_live_vintages": boundary_row.get("required_live_vintages", np.nan),
                "first_available_date": boundary_row.get("first_available_date", ""),
                "last_available_date": boundary_row.get("last_available_date", ""),
                "first_complete_date": boundary_row.get("first_complete_date", ""),
                "last_complete_date": boundary_row.get("last_complete_date", ""),
                "dates_excluded_at_start": boundary_row.get("dates_excluded_at_start", np.nan),
                "dates_excluded_at_end": boundary_row.get("dates_excluded_at_end", np.nan),
                "retained_complete_dates": boundary_row.get("retained_complete_dates", np.nan),
                "minimum_live_vintages_before_filter": boundary_row.get("minimum_live_vintages_before_filter", np.nan),
                "maximum_live_vintages_before_filter": boundary_row.get("maximum_live_vintages_before_filter", np.nan),
                "minimum_live_vintages_after_filter": boundary_row.get("minimum_live_vintages_after_filter", np.nan),
                "maximum_live_vintages_after_filter": boundary_row.get("maximum_live_vintages_after_filter", np.nan),
                "complete_committee_validation_passed": boundary_row.get("complete_committee_validation_passed", False),
                "committed_pair_slots_per_window": int(config.pair_slots),
                "test_period_net_return": net,
                "annualized_net_return": ann,
                "annualized_net_return_additive": ann_additive,
                "annualized_net_return_minus_additive": ann - ann_additive,
                "annualized_volatility": vol,
                "annualized_sharpe_rf_zero": sharpe_zero,
                "annualized_sharpe_fama_french_rf": sharpe_ff,
                "maximum_drawdown": float(group["drawdown"].min()),
                "mean_daily_return": safe_mean(returns),
                "median_daily_return": safe_median(returns),
                "daily_return_std": float(returns.std(ddof=1)) if n > 1 else float("nan"),
                "positive_day_rate": float((returns > 0).mean()) if n else float("nan"),
                "minimum_daily_return": float(returns.min()) if n else float("nan"),
                "maximum_daily_return": float(returns.max()) if n else float("nan"),
                "newey_west_lags": int(config.hac_lags),
                "newey_west_tstat": tstat,
                "newey_west_pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)[SUMMARY_LEADING_COLUMNS]


def window_metrics(vintage: pd.DataFrame, profits: pd.DataFrame, trades: pd.DataFrame, config: SummaryConfig, mode: str) -> pd.DataFrame:
    case = canonical_case(config.sector, config.year)
    valid = profits if "threshold_valid" not in profits.columns else profits[truthy(profits["threshold_valid"])].copy()
    rows: list[dict[str, Any]] = []
    group_cols = [*BASE_RESULT_GROUP_COLUMNS, "window_id"]
    trade_group_cols = ["selection_scope", "source_backtest", "comparison_family", "model", "optimization_cost_case", "gamma_multiplier", "window_id"]
    profit_lookup = {
        key: group
        for key, group in valid.groupby(trade_group_cols, dropna=False, sort=False)
    }
    trade_lookup_by_window = (
        {
            key: group
            for key, group in trades.groupby(trade_group_cols, dropna=False, sort=False)
        }
        if not trades.empty
        else {}
    )
    for key, group in vintage.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        basis = key_dict["return_basis"]
        lookup_key = tuple(key_dict[col] for col in trade_group_cols)
        profit_group = profit_lookup.get(lookup_key, pd.DataFrame())
        trade_group = trade_lookup_by_window.get(lookup_key, pd.DataFrame())
        trade_returns = pd.to_numeric(trade_group.get(RETURN_BASIS_TO_TRADE_COL[basis], pd.Series(dtype=float)), errors="coerce").dropna()
        forced = truthy(trade_group["forced_exit"]).sum() if "forced_exit" in trade_group.columns else 0
        selected_pair_count = int(profit_group["pair_id"].nunique()) if "pair_id" in profit_group.columns else 0
        traded_pair_count = int(trade_group["pair_id"].nunique()) if "pair_id" in trade_group.columns else 0
        window_returns = pd.to_numeric(group["vintage_daily_return"], errors="coerce").dropna()
        window_test_return = float(window_returns.sum()) if len(window_returns) else 0.0
        window_days = int(group["date"].nunique())
        rows.append(
            {
                "case": case,
                "sector": config.sector,
                "year": str(config.year),
                "aggregation_mode": mode,
                **key_dict,
                "trading_start_date": str(group["trading_start_date"].min()),
                "trading_end_date": str(group["trading_end_date"].max()),
                "number_of_trading_days": window_days,
                "committed_pair_slots": int(config.pair_slots),
                "actual_selected_pairs": selected_pair_count,
                "pairs_that_traded": traded_pair_count,
                "pair_participation_rate": float(traded_pair_count / selected_pair_count)
                if selected_pair_count
                else float("nan"),
                "completed_trades": int(len(trade_group)),
                "forced_exits": int(forced),
                "forced_exit_rate": float(forced / len(trade_group)) if len(trade_group) else float("nan"),
                "compounded_window_return": window_test_return,
                "annualized_window_return_additive": linear_additive_annualized_return(
                    window_test_return, window_days, config.annualisation_days
                ),
                "mean_daily_window_return": float(window_returns.mean()) if len(window_returns) else float("nan"),
                "median_daily_window_return": float(window_returns.median()) if len(window_returns) else float("nan"),
                "positive_day_rate": float((window_returns > 0).mean()) if len(window_returns) else float("nan"),
                "mean_trade_return": safe_mean(trade_returns),
                "median_trade_return": safe_median(trade_returns),
                "mean_holding_minutes": safe_mean(trade_group.get("duration_minutes", pd.Series(dtype=float))),
                "median_holding_minutes": safe_median(trade_group.get("duration_minutes", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows)[WINDOW_COLUMNS].sort_values(
        [*BASE_RESULT_GROUP_COLUMNS, "window_id"]
    )


def trade_metrics(profits: pd.DataFrame, trades: pd.DataFrame, return_bases: list[str], config: SummaryConfig, mode: str) -> pd.DataFrame:
    case = canonical_case(config.sector, config.year)
    valid = profits if "threshold_valid" not in profits.columns else profits[truthy(profits["threshold_valid"])].copy()
    base_cols = ["selection_scope", "source_backtest", "comparison_family", "model", "optimization_cost_case", "gamma_multiplier"]
    rows: list[dict[str, Any]] = []
    for key, profit_group in valid.groupby(base_cols, dropna=False):
        key_dict = dict(zip(base_cols, key))
        t_mask = pd.Series(False, index=trades.index) if trades.empty else pd.Series(True, index=trades.index)
        if not trades.empty:
            for col, value in key_dict.items():
                t_mask &= trades[col].eq(value)
            trade_group = trades[t_mask]
        else:
            trade_group = pd.DataFrame()
        for basis in return_bases:
            trade_returns = pd.to_numeric(
                trade_group.get(RETURN_BASIS_TO_TRADE_COL[basis], pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            gains = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]
            forced_mask = truthy(trade_group["forced_exit"]) if "forced_exit" in trade_group.columns else pd.Series(False, index=trade_group.index)
            normal_returns = pd.to_numeric(
                trade_group.loc[~forced_mask, RETURN_BASIS_TO_TRADE_COL[basis]] if len(trade_group) else pd.Series(dtype=float),
                errors="coerce",
            ).dropna()
            forced_returns = pd.to_numeric(
                trade_group.loc[forced_mask, RETURN_BASIS_TO_TRADE_COL[basis]] if len(trade_group) else pd.Series(dtype=float),
                errors="coerce",
            ).dropna()
            n_windows = int(profit_group["window_id"].nunique())
            rows.append(
                {
                    "case": case,
                    "sector": config.sector,
                    "year": str(config.year),
                    "aggregation_mode": mode,
                    **key_dict,
                    "return_basis": basis,
                    "committed_pair_slots": int(config.pair_slots),
                    "actual_selected_pairs": int(profit_group[["window_id", "pair_id"]].drop_duplicates().shape[0]),
                    "pairs_that_traded": int(trade_group[["window_id", "pair_id"]].drop_duplicates().shape[0]) if len(trade_group) else 0,
                    "pair_participation_rate": float(
                        (trade_group[["window_id", "pair_id"]].drop_duplicates().shape[0] if len(trade_group) else 0)
                        / profit_group[["window_id", "pair_id"]].drop_duplicates().shape[0]
                    )
                    if profit_group[["window_id", "pair_id"]].drop_duplicates().shape[0]
                    else float("nan"),
                    "num_completed_trades": int(len(trade_group)),
                    "trades_per_window": float(len(trade_group) / n_windows) if n_windows else float("nan"),
                    "trades_per_committed_pair_slot": float(len(trade_group) / (n_windows * config.pair_slots))
                    if n_windows
                    else float("nan"),
                    "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else float("nan"),
                    "mean_trade_return": safe_mean(trade_returns),
                    "median_trade_return": safe_median(trade_returns),
                    "average_gain": safe_mean(gains),
                    "median_gain": safe_median(gains),
                    "average_loss": safe_mean(losses),
                    "median_loss": safe_median(losses),
                    "mean_holding_minutes": safe_mean(trade_group.get("duration_minutes", pd.Series(dtype=float))),
                    "median_holding_minutes": safe_median(trade_group.get("duration_minutes", pd.Series(dtype=float))),
                    "num_forced_exits": int(forced_mask.sum()) if len(trade_group) else 0,
                    "forced_exit_rate": float(forced_mask.mean()) if len(trade_group) else float("nan"),
                    "mean_normal_exit_return": safe_mean(normal_returns),
                    "median_normal_exit_return": safe_median(normal_returns),
                    "mean_forced_exit_return": safe_mean(forced_returns),
                    "median_forced_exit_return": safe_median(forced_returns),
                }
            )
    return pd.DataFrame(rows)[TRADE_METRIC_COLUMNS].sort_values(
        BASE_RESULT_GROUP_COLUMNS
    )


def grid_bounds_for_source(backtest_dir: Path) -> tuple[float, float, list[str]]:
    lower, upper = float("nan"), float("nan")
    warnings: list[str] = []
    for name in ("run_summary.json", "repair_summary.json"):
        path = backtest_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        settings = payload.get("settings", payload)
        if isinstance(settings, dict):
            if "min_sigma_multiple" in settings:
                lower = finite_float(settings.get("min_sigma_multiple"), float("nan"))
            if "max_sigma_multiple" in settings:
                upper = finite_float(settings.get("max_sigma_multiple"), float("nan"))
    if not (np.isfinite(lower) and np.isfinite(upper)):
        warnings.append(
            f"threshold grid bounds missing in stored optimiser metadata for {backtest_dir}; "
            "boundary-rate columns set to NaN"
        )
    return lower, upper, warnings


def threshold_metrics(
    profits: pd.DataFrame,
    backtest_dirs: list[Path],
    config: SummaryConfig,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    case = canonical_case(config.sector, config.year)
    source_bounds: dict[str, tuple[float, float]] = {}
    warnings: list[str] = []
    for path in backtest_dirs:
        lower, upper, source_warnings = grid_bounds_for_source(path)
        source_bounds[path.name] = (lower, upper)
        warnings.extend(source_warnings)
    rows: list[dict[str, Any]] = []
    group_cols = ["selection_scope", "source_backtest", "comparison_family", "model", "optimization_cost_case", "gamma_multiplier"]
    for key, group in profits.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        valid = truthy(group["threshold_valid"]) if "threshold_valid" in group.columns else pd.Series(True, index=group.index)
        valid_group = group[valid].copy()
        scale = pd.to_numeric(valid_group.get("threshold_scale", pd.Series(np.nan, index=valid_group.index)), errors="coerce")
        if scale.isna().all() and "d_plus_sigma" in group.columns:
            d_plus = pd.to_numeric(valid_group["d_plus"], errors="coerce")
            d_plus_sigma = pd.to_numeric(valid_group["d_plus_sigma"], errors="coerce")
            scale = d_plus / d_plus_sigma.replace(0, np.nan)
        d_plus = pd.to_numeric(valid_group.get("d_plus", pd.Series(dtype=float)), errors="coerce")
        d_minus = pd.to_numeric(valid_group.get("d_minus", pd.Series(dtype=float)), errors="coerce")
        d_plus_over = d_plus / scale.replace(0, np.nan)
        d_minus_over = d_minus / scale.replace(0, np.nan)
        normalized_asym = (d_plus - d_minus) / scale.replace(0, np.nan)
        denominator = d_plus + d_minus
        asym = pd.Series(
            np.where(denominator > 0.0, (d_plus - d_minus) / denominator, np.nan),
            index=valid_group.index,
            dtype=float,
        )
        absolute_asym = asym.abs()
        lower, upper = source_bounds.get(str(key_dict["source_backtest"]), (float("nan"), float("nan")))
        tol = 1e-8
        comparable = np.isfinite(d_plus_over.to_numpy(dtype=float)) & np.isfinite(d_minus_over.to_numpy(dtype=float))
        symmetric = pd.Series(
            np.where(comparable, np.isclose(d_plus_over, d_minus_over, atol=tol, rtol=0.0), np.nan),
            index=valid_group.index,
            dtype=float,
        )
        upper_farther = pd.Series(
            np.where(comparable, d_plus_over > d_minus_over + tol, np.nan),
            index=valid_group.index,
            dtype=float,
        )
        lower_farther = pd.Series(
            np.where(comparable, d_minus_over > d_plus_over + tol, np.nan),
            index=valid_group.index,
            dtype=float,
        )
        has_bounds = np.isfinite(lower) and np.isfinite(upper)
        has_valid_thresholds = len(d_plus_over) > 0
        rows.append(
            {
                "case": case,
                "sector": config.sector,
                "year": str(config.year),
                "aggregation_mode": mode,
                **key_dict,
                "num_threshold_rows": int(len(group)),
                "num_valid_threshold_rows": int(valid.sum()),
                "valid_threshold_rate": float(valid.mean()) if len(valid) else float("nan"),
                "median_d_plus": safe_median(d_plus),
                "q25_d_plus": quantile(d_plus, 0.25),
                "q75_d_plus": quantile(d_plus, 0.75),
                "median_d_minus": safe_median(d_minus),
                "q25_d_minus": quantile(d_minus, 0.25),
                "q75_d_minus": quantile(d_minus, 0.75),
                "median_d_plus_over_scale": safe_median(d_plus_over),
                "q25_d_plus_over_scale": quantile(d_plus_over, 0.25),
                "q75_d_plus_over_scale": quantile(d_plus_over, 0.75),
                "median_d_minus_over_scale": safe_median(d_minus_over),
                "q25_d_minus_over_scale": quantile(d_minus_over, 0.25),
                "q75_d_minus_over_scale": quantile(d_minus_over, 0.75),
                "mean_threshold_scale": safe_mean(scale),
                "median_threshold_scale": safe_median(scale),
                "q25_threshold_scale": quantile(scale, 0.25),
                "q75_threshold_scale": quantile(scale, 0.75),
                "median_normalized_asymmetry": safe_median(normalized_asym),
                "q25_normalized_asymmetry": quantile(normalized_asym, 0.25),
                "q75_normalized_asymmetry": quantile(normalized_asym, 0.75),
                "symmetric_rate": safe_mean(symmetric),
                "upper_farther_rate": safe_mean(upper_farther),
                "lower_farther_rate": safe_mean(lower_farther),
                "median_asymmetry": safe_median(asym),
                "q25_asymmetry": quantile(asym, 0.25),
                "q75_asymmetry": quantile(asym, 0.75),
                "median_absolute_asymmetry": safe_median(absolute_asym),
                "q75_absolute_asymmetry": quantile(absolute_asym, 0.75),
                "d_plus_lower_boundary_rate": (
                    float(np.isclose(d_plus_over, lower, atol=tol, rtol=0).mean())
                    if has_bounds and has_valid_thresholds
                    else float("nan")
                ),
                "d_plus_upper_boundary_rate": (
                    float(np.isclose(d_plus_over, upper, atol=tol, rtol=0).mean())
                    if has_bounds and has_valid_thresholds
                    else float("nan")
                ),
                "d_minus_lower_boundary_rate": (
                    float(np.isclose(d_minus_over, lower, atol=tol, rtol=0).mean())
                    if has_bounds and has_valid_thresholds
                    else float("nan")
                ),
                "d_minus_upper_boundary_rate": (
                    float(np.isclose(d_minus_over, upper, atol=tol, rtol=0).mean())
                    if has_bounds and has_valid_thresholds
                    else float("nan")
                ),
            }
        )
    metadata = {
        source: {"min_sigma_multiple": bounds[0], "max_sigma_multiple": bounds[1]}
        for source, bounds in source_bounds.items()
    }
    return pd.DataFrame(rows)[THRESHOLD_COLUMNS].sort_values(group_cols), metadata, warnings


def model_comparisons(
    daily: pd.DataFrame,
    config: SummaryConfig,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    baseline_model = "gaussian_fixed_sigma_eq"
    available_models = set(daily["model"].astype(str))
    if baseline_model not in available_models:
        reason = "Gaussian fixed-sigma-eq baseline daily results are not present."
        return (
            pd.DataFrame(columns=MODEL_COMPARISON_COLUMNS),
            pd.DataFrame(columns=MODEL_DAILY_DIFFERENCE_COLUMNS),
            reason,
        )

    case = canonical_case(config.sector, config.year)
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    group_cols = ["selection_scope", "comparison_family", "optimization_cost_case", "gamma_multiplier", "return_basis"]
    dup_cols = [*group_cols, "model", "date"]
    dup = daily.duplicated(dup_cols, keep=False)
    if dup.any():
        sample = daily.loc[dup, dup_cols].head(10).to_dict("records")
        raise SystemExit(f"Model comparison would be many-to-many; duplicate family/model/date rows: {sample}")

    for key, group in daily.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        base = group[group["model"].eq(baseline_model)][
            ["date", "daily_pnl_return_on_initial_committed_capital"]
        ].rename(columns={"daily_pnl_return_on_initial_committed_capital": "baseline"})
        if base.empty:
            continue

        for model, alt in group[~group["model"].eq(baseline_model)].groupby("model"):
            merged = base.merge(
                alt[["date", "daily_pnl_return_on_initial_committed_capital"]].rename(
                    columns={"daily_pnl_return_on_initial_committed_capital": "comparison"}
                ),
                on="date",
                how="inner",
                validate="one_to_one",
            )
            if merged.empty:
                continue

            diff = merged["comparison"] - merged["baseline"]
            comparison_net = additive_test_period_return(merged["comparison"])
            baseline_net = additive_test_period_return(merged["baseline"])
            tstat, pvalue = hac_mean_test(diff, config.hac_lags)

            rows.append(
                {
                    "case": case,
                    "sector": config.sector,
                    "year": str(config.year),
                    "aggregation_mode": mode,
                    "boundary_policy": config.boundary_policy if mode == "overlapping" else "not_applicable",
                    **key_dict,
                    "baseline_model": baseline_model,
                    "comparison_model": model,
                    "number_of_matched_days": int(len(merged)),
                    "mean_daily_return_difference": safe_mean(diff),
                    "median_daily_return_difference": safe_median(diff),
                    "test_period_return_difference": comparison_net - baseline_net,
                    "annualized_return_difference": _annualized(comparison_net, len(merged), config)
                    - _annualized(baseline_net, len(merged), config),
                    "newey_west_tstat_difference": tstat,
                    "newey_west_pvalue_difference": pvalue,
                }
            )

            for date, baseline_return, comparison_return, difference in zip(
                merged["date"], merged["baseline"], merged["comparison"], diff
            ):
                daily_rows.append(
                    {
                        "case": case,
                        "sector": config.sector,
                        "year": str(config.year),
                        "aggregation_mode": mode,
                        "boundary_policy": config.boundary_policy if mode == "overlapping" else "not_applicable",
                        **key_dict,
                        "baseline_model": baseline_model,
                        "comparison_model": model,
                        "date": date,
                        "baseline_daily_return": float(baseline_return),
                        "comparison_daily_return": float(comparison_return),
                        "daily_return_difference": float(difference),
                    }
                )

    comparison_summary = pd.DataFrame(rows, columns=MODEL_COMPARISON_COLUMNS)
    daily_differences = pd.DataFrame(daily_rows, columns=MODEL_DAILY_DIFFERENCE_COLUMNS)
    if not daily_differences.empty:
        daily_differences = daily_differences.sort_values(
            [
                "selection_scope",
                "comparison_family",
                "optimization_cost_case",
                "gamma_multiplier",
                "return_basis",
                "comparison_model",
                "date",
            ]
        ).reset_index(drop=True)
    return comparison_summary, daily_differences, None


def _annualized(net: float, n: int, config: SummaryConfig) -> float:
    return additive_annualized_return(1.0 + float(net), n, config.annualisation_days)


def cost_impact(daily: pd.DataFrame, config: SummaryConfig, mode: str) -> tuple[pd.DataFrame, str | None]:
    required = set(RETURN_BASIS_TO_PROFIT_COL)
    if not required.issubset(set(daily["return_basis"].astype(str))):
        return pd.DataFrame(columns=COST_IMPACT_COLUMNS), "Not all return bases were requested or available."
    case = canonical_case(config.sector, config.year)
    rows: list[dict[str, Any]] = []
    pairs = [("midquote", "midquote_fixed_bps"), ("midquote", "bid_ask"), ("midquote_fixed_bps", "bid_ask")]
    group_cols = ["selection_scope", "source_backtest", "comparison_family", "model", "optimization_cost_case", "gamma_multiplier"]
    dup_cols = [*group_cols, "return_basis", "date"]
    dup = daily.duplicated(dup_cols, keep=False)
    if dup.any():
        sample = daily.loc[dup, dup_cols].head(10).to_dict("records")
        raise SystemExit(f"Cost-impact comparison would hide duplicate group/basis/date rows: {sample}")
    for key, group in daily.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        pivot = group.pivot(index="date", columns="return_basis", values="daily_pnl_return_on_initial_committed_capital")
        for b1, b2 in pairs:
            if b1 not in pivot.columns or b2 not in pivot.columns:
                continue
            matched = pivot[[b1, b2]].dropna()
            if matched.empty:
                continue
            net1 = additive_test_period_return(matched[b1])
            net2 = additive_test_period_return(matched[b2])
            diff = matched[b1] - matched[b2]
            rows.append(
                {
                    "case": case,
                    "sector": config.sector,
                    "year": str(config.year),
                    "aggregation_mode": mode,
                    "boundary_policy": config.boundary_policy if mode == "overlapping" else "not_applicable",
                    **key_dict,
                    "basis_1": b1,
                    "basis_2": b2,
                    "number_of_matched_days": int(len(matched)),
                    "test_period_return_basis_1": net1,
                    "test_period_return_basis_2": net2,
                    "test_period_return_difference": net1 - net2,
                    "annualized_return_basis_1": _annualized(net1, len(matched), config),
                    "annualized_return_basis_2": _annualized(net2, len(matched), config),
                    "annualized_return_difference": _annualized(net1, len(matched), config)
                    - _annualized(net2, len(matched), config),
                    "mean_daily_return_difference": safe_mean(diff),
                    "median_daily_return_difference": safe_median(diff),
                }
            )
    return pd.DataFrame(rows, columns=COST_IMPACT_COLUMNS), None


def selection_comparison(profits: pd.DataFrame, config: SummaryConfig, mode: str) -> tuple[pd.DataFrame, str | None, dict[str, Any]]:
    valid_profits = profits if "threshold_valid" not in profits.columns else profits[truthy(profits["threshold_valid"])].copy()
    scopes = set(valid_profits["selection_scope"].astype(str))
    diagnostics: dict[str, Any] = {"skipped_unmatched_group_count": 0, "skipped_unmatched_groups": []}
    if not {"gaussian_top10", "adf_capped10"}.issubset(scopes):
        return (
            pd.DataFrame(columns=SELECTION_COMPARISON_COLUMNS),
            "Both gaussian_top10 and adf_capped10 branches are required.",
            diagnostics,
        )
    case = canonical_case(config.sector, config.year)
    family_cols = ["comparison_family", "model", "optimization_cost_case", "gamma_multiplier", "window_id"]
    pairs = valid_profits[[*family_cols, "selection_scope", "pair_id"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for family_key, family_group in pairs.groupby(family_cols, dropna=False):
        family = dict(zip(family_cols, family_key))
        unrestricted = set(family_group[family_group["selection_scope"].eq("gaussian_top10")]["pair_id"])
        adf = set(family_group[family_group["selection_scope"].eq("adf_capped10")]["pair_id"])
        present_scopes = set(family_group["selection_scope"].astype(str))
        if not {"gaussian_top10", "adf_capped10"}.issubset(present_scopes):
            diagnostics["skipped_unmatched_group_count"] += 1
            if len(diagnostics["skipped_unmatched_groups"]) < 25:
                diagnostics["skipped_unmatched_groups"].append({**family, "present_scopes": sorted(present_scopes)})
            continue
        if not unrestricted and not adf:
            continue
        overlap = unrestricted & adf
        union = unrestricted | adf
        rows.append(
            {
                "case": case,
                "sector": config.sector,
                "year": str(config.year),
                "aggregation_mode": mode,
                "comparison_family": family["comparison_family"],
                "model": family["model"],
                "optimization_cost_case": family["optimization_cost_case"],
                "gamma_multiplier": family["gamma_multiplier"],
                "row_type": "window",
                "window_id": family["window_id"],
                "unrestricted_pair_count": len(unrestricted),
                "adf_pair_count": len(adf),
                "overlap_count": len(overlap),
                "unrestricted_retained_rate": len(overlap) / len(unrestricted) if unrestricted else float("nan"),
                "jaccard_similarity": len(overlap) / len(union) if union else float("nan"),
                "unrestricted_pairs_excluded_count": len(unrestricted - adf),
                "unrestricted_pairs_excluded_rate": len(unrestricted - adf) / len(unrestricted) if unrestricted else float("nan"),
                "adf_windows_with_fewer_than_10_pairs": int(len(adf) < config.pair_slots),
                "mean_adf_selected_pairs": np.nan,
                "proportion_of_adf_windows_with_fewer_than_10_pairs": np.nan,
                "mean_overlap_count": np.nan,
                "mean_unrestricted_retained_rate": np.nan,
                "mean_jaccard_similarity": np.nan,
                "mean_unrestricted_excluded_rate": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=SELECTION_COMPARISON_COLUMNS), "No matched windows for selection comparison.", diagnostics
    summary = {
        "case": case,
        "sector": config.sector,
        "year": str(config.year),
        "aggregation_mode": mode,
        "comparison_family": "all_matched_families",
        "model": "all",
        "optimization_cost_case": "all",
        "gamma_multiplier": "all",
        "row_type": "summary",
        "window_id": np.nan,
        "unrestricted_pair_count": np.nan,
        "adf_pair_count": np.nan,
        "overlap_count": np.nan,
        "unrestricted_retained_rate": np.nan,
        "jaccard_similarity": np.nan,
        "unrestricted_pairs_excluded_count": np.nan,
        "unrestricted_pairs_excluded_rate": np.nan,
        "adf_windows_with_fewer_than_10_pairs": np.nan,
        "mean_adf_selected_pairs": safe_mean(out["adf_pair_count"]),
        "proportion_of_adf_windows_with_fewer_than_10_pairs": safe_mean(out["adf_windows_with_fewer_than_10_pairs"]),
        "mean_overlap_count": safe_mean(out["overlap_count"]),
        "mean_unrestricted_retained_rate": safe_mean(out["unrestricted_retained_rate"]),
        "mean_jaccard_similarity": safe_mean(out["jaccard_similarity"]),
        "mean_unrestricted_excluded_rate": safe_mean(out["unrestricted_pairs_excluded_rate"]),
    }
    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
    diagnostics["matched_group_count"] = int((out["row_type"] == "window").sum())
    return out[SELECTION_COMPARISON_COLUMNS], None, diagnostics


def write_outputs(
    config: SummaryConfig,
    mode: str,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    windows: pd.DataFrame,
    trades: pd.DataFrame,
    thresholds: pd.DataFrame,
    threshold_rows: pd.DataFrame,
    model_cmp: pd.DataFrame,
    model_daily_differences: pd.DataFrame,
    selection_cmp: pd.DataFrame,
    cost_cmp: pd.DataFrame,
    boundary: pd.DataFrame,
    run_summary: dict[str, Any],
) -> Path:
    case = canonical_case(config.sector, config.year)
    base_out_dir = config.outputs_root / case / "results_overlapping"
    is_final_all_model_run = (
        config.selection_scope == "all"
        and config.model == "all"
        and config.return_basis == "all"
    )
    if is_final_all_model_run:
        out_dir = base_out_dir
    else:
        filtered_tag = (
            f"selection_{config.selection_scope}"
            f"__model_{config.model}"
            f"__return_{config.return_basis}"
            f"__boundary_{config.boundary_policy}"
        )
        out_dir = base_out_dir / "filtered_runs" / filtered_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "overlapping"
    files = {
        f"{prefix}_summary_metrics.csv": summary,
        f"{prefix}_daily_returns.csv": daily,
        f"{prefix}_window_metrics.csv": windows,
        f"{prefix}_trade_metrics.csv": trades,
        f"{prefix}_threshold_metrics.csv": thresholds,
        f"{prefix}_threshold_rows.csv": threshold_rows,
        f"{prefix}_model_comparisons.csv": model_cmp,
        f"{prefix}_model_daily_differences.csv": model_daily_differences,
        f"{prefix}_selection_comparison.csv": selection_cmp,
        f"{prefix}_cost_impact.csv": cost_cmp,
        f"{prefix}_boundary_diagnostics.csv": boundary,
    }
    output_paths: dict[str, str] = {}
    for filename, frame in files.items():
        path = out_dir / filename
        frame.to_csv(path, index=False)
        output_paths[filename] = str(path)
    summary_path = out_dir / f"{prefix}_run_summary.json"
    run_summary["output_paths"] = {**output_paths, summary_path.name: str(summary_path)}
    summary_path.write_text(json.dumps(run_summary, indent=2, default=str) + "\n")
    return out_dir


def run_summary(config: SummaryConfig, mode: str) -> Path:
    if mode != "overlapping":
        raise SystemExit("Only overlapping result summaries are supported.")
    case = canonical_case(config.sector, config.year)
    case_dir = config.outputs_root / case
    return_bases = requested_return_bases(config.return_basis)
    warnings: list[str] = []
    profits, trades, backtest_dirs = load_backtests(config, mode)
    diagnostics = validate_inputs(profits, trades)
    diagnostics.extend(validate_fixed_bps_execution_costs(trades))
    data_path = infer_data_path(case, case_dir, backtest_dirs, config.data_path)
    rf_daily = load_fama_french_daily_rf(config.fama_french_factors_path)
    quote_cache, actual_calendar, skipped_quote_windows, quote_warnings = load_daily_quote_cache(profits, data_path)
    warnings.extend(quote_warnings)
    zero_adf_windows, zero_adf_warnings = load_zero_adf_windows(config, mode, case_dir)
    warnings.extend(zero_adf_warnings)
    vintage, _pair_daily, reconciliation = build_vintage_daily(
        profits=profits,
        trades=trades,
        quote_cache=quote_cache,
        return_bases=return_bases,
        pair_slots=config.pair_slots,
        tolerance=config.reconciliation_tolerance,
    )
    vintage, cash_reconciliation = add_zero_adf_cash_vintages(vintage, zero_adf_windows, actual_calendar)
    vintage.attrs["actual_trading_calendar"] = actual_calendar
    diagnostics.extend(reconciliation)
    diagnostics.extend(cash_reconciliation)
    daily_raw, boundary = aggregate_daily(vintage, config, mode)
    vintage.attrs.clear()
    daily = attach_wealth(daily_raw, rf_daily, config, mode)
    summary = performance_rows(daily, vintage, boundary, config, mode)
    windows = window_metrics(vintage, profits, trades, config, mode)
    trade_summary = trade_metrics(profits, trades, return_bases, config, mode)
    threshold_summary, threshold_metadata, threshold_warnings = threshold_metrics(profits, backtest_dirs, config, mode)
    warnings.extend(threshold_warnings)
    threshold_rows, threshold_row_warnings = load_threshold_rows(config, mode, backtest_dirs)
    warnings.extend(threshold_row_warnings)
    model_cmp, model_daily_differences, model_cmp_reason = model_comparisons(daily, config, mode)
    selection_cmp, selection_cmp_reason, selection_cmp_diagnostics = selection_comparison(profits, config, mode)
    cost_cmp, cost_cmp_reason = cost_impact(daily, config, mode)
    numeric_windows = pd.to_numeric(vintage["window_id"], errors="coerce")
    if numeric_windows.notna().all():
        selected_window_ids = sorted(numeric_windows.astype(int).drop_duplicates().tolist())
    else:
        selected_window_ids = sorted(vintage["window_id"].astype(str).drop_duplicates().tolist())
    source_files = {
        path.name: {
            "profits_csv": str(path / "profits.csv"),
            "trades_csv": str(path / "trades.csv"),
            "thresholds_csv": str(path / "thresholds.csv") if (path / "thresholds.csv").exists() else "",
            "run_summary_json": str(path / "run_summary.json") if (path / "run_summary.json").exists() else "",
            "repair_summary_json": str(path / "repair_summary.json") if (path / "repair_summary.json").exists() else "",
        }
        for path in backtest_dirs
    }
    payload = {
        "case": case,
        "sector": config.sector,
        "year": str(config.year),
        "aggregation_mode": mode,
        "selection_scope": config.selection_scope,
        "model": config.model,
        "return_basis": config.return_basis,
        "return_bases_written": return_bases,
        "pair_slots": config.pair_slots,
        "trading_window_days": config.trading_window_days,
        "annualisation_days": config.annualisation_days,
        "hac_lags": config.hac_lags,
        "model_comparison_baseline": "gaussian_fixed_sigma_eq",
        "fama_french_factors_path": str(config.fama_french_factors_path),
        "boundary_policy": config.boundary_policy if mode == "overlapping" else "not_applicable",
        "cli_arguments": config.cli_args or {},
        "data_path": str(data_path),
        "source_file_paths": source_files,
        "source_backtests": [str(path.relative_to(case_dir)) for path in backtest_dirs],
        "skipped_quote_windows": skipped_quote_windows,
        "skipped_quote_window_count": len(skipped_quote_windows),
        "row_grain_detection": {
            "profits_csv": "one row per selected pair-window, model, optimisation cost case, gamma multiplier and threshold row; not a trade and not a portfolio row",
            "trades_csv": "one row per realised replayed trade event, including entry/exit timestamps and execution quotes",
            "summary_outputs": "daily portfolio series reconstructed from trades plus quote-panel daily close marks",
        },
        "column_mappings": {
            "return_basis_to_profit_column": RETURN_BASIS_TO_PROFIT_COL,
            "return_basis_to_trade_column": RETURN_BASIS_TO_TRADE_COL,
            "risk_free_column": {
                "source_file_column": "RF",
                "output_column": "risk_free_daily_fama_french",
                "unit_conversion": "Fama-French percentage units divided by 100",
            },
            "threshold_columns": {
                "d_plus": "d_plus",
                "d_minus": "d_minus",
                "scale": "threshold_scale, or d_plus / d_plus_sigma when threshold_scale is absent",
                "threshold_valid": "threshold_valid",
                "lower_grid_boundary": "stored optimiser setting min_sigma_multiple",
                "upper_grid_boundary": "stored optimiser setting max_sigma_multiple",
            },
            "grouping_keys": BASE_RESULT_GROUP_COLUMNS,
            "comparison_family": "source_backtest normalized by removing selection/model-family tokens",
        },
        "return_units": {
            "pair_return_columns": "return_on_gross pair capital; one dollar allocated to each leg and divided by two in execution.py",
            "daily_pnl_return_on_initial_committed_capital": "fixed-notional additive daily P&L increment divided by initial committed capital",
            "daily_return": "compatibility alias for daily_pnl_return_on_initial_committed_capital; do not geometrically compound it",
            "risk_free_daily_fama_french": "daily U.S. one-month Treasury bill return from Fama-French RF, divided by 100",
            "wealth_index": "1 + cumulative_sum(daily_pnl_return_on_initial_committed_capital)",
        },
        "capital_denominator": {
            "pair_slots": config.pair_slots,
            "convention": "fixed committed gross capital; unused ADF slots and non-traded selected pairs earn zero",
        },
        "threshold_mappings_and_grid_bounds": threshold_metadata,
        "selected_window_ids": selected_window_ids,
        "sample_dates": {
            "daily_first_date": str(daily["date"].min()) if not daily.empty else "",
            "daily_last_date": str(daily["date"].max()) if not daily.empty else "",
            "actual_trading_calendar_first_date": actual_calendar[0] if actual_calendar else "",
            "actual_trading_calendar_last_date": actual_calendar[-1] if actual_calendar else "",
        },
        "boundary_diagnostics": boundary.to_dict("records"),
        "selection_comparison_diagnostics": selection_cmp_diagnostics,
        "profit_rows_read": int(len(profits)),
        "trade_rows_read": int(len(trades)),
        "vintage_daily_rows": int(len(vintage)),
        "daily_rows_written": int(len(daily)),
        "summary_rows_written": int(len(summary)),
        "threshold_rows_written": int(len(threshold_rows)),
        "model_daily_difference_rows_written": int(len(model_daily_differences)),
        "warnings": warnings,
        "warning_count": len(warnings),
        "validation_checks": {
            "passed": diagnostics,
            "failed": [],
        },
        "reconciliation_examples": {
            "daily_pnl": "All checked pair-window daily marked P&L sums matched final backtest P&L within tolerance.",
            "fixed_bps": "Fixed-bps entry and exit cost formulas were checked directly against trades.csv fields.",
            "wealth_accounting": "Fixed-notional returns use additive wealth: wealth_t = 1 + cumulative_sum(daily P&L increments).",
        },
        "empty_comparison_reasons": {
            "model_comparisons": model_cmp_reason,
            "selection_comparison": selection_cmp_reason,
            "cost_impact": cost_cmp_reason,
        },
    }
    return write_outputs(
        config,
        mode,
        summary,
        daily,
        windows,
        trade_summary,
        threshold_summary,
        threshold_rows,
        model_cmp,
        model_daily_differences,
        selection_cmp,
        cost_cmp,
        boundary,
        payload,
    )
