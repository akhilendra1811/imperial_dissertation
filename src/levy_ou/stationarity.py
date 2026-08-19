"""Stationarity diagnostics for pair-window spreads."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from levy_ou.spreads import build_spread


def adf_test(
    series: pd.Series | np.ndarray,
    significance: float = 0.05,
    regression: str = "c",
    autolag: str = "AIC",
    maxlag: int | None = None,
    min_observations: int = 20,
) -> dict[str, Any]:
    """Run an ADF unit-root test on a spread in levels."""

    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    observations = int(len(clean))
    if observations < int(min_observations):
        return {
            "adf_valid": False,
            "adf_reason": f"too few observations; need at least {int(min_observations)}",
            "adf_observations": observations,
            "adf_significance": float(significance),
            "adf_regression": regression,
            "adf_autolag": autolag,
            "adf_maxlag": maxlag,
            "adf_reject_unit_root": False,
            "adf_pass": False,
        }
    if float(clean.std(ddof=0)) <= 0.0:
        return {
            "adf_valid": False,
            "adf_reason": "constant spread",
            "adf_observations": observations,
            "adf_significance": float(significance),
            "adf_regression": regression,
            "adf_autolag": autolag,
            "adf_maxlag": maxlag,
            "adf_reject_unit_root": False,
            "adf_pass": False,
        }

    try:
        result = adfuller(clean.to_numpy(dtype=float), maxlag=maxlag, regression=regression, autolag=autolag)
    except Exception as exc:  # statsmodels raises several data-dependent errors.
        return {
            "adf_valid": False,
            "adf_reason": f"adfuller failed: {exc}",
            "adf_observations": observations,
            "adf_significance": float(significance),
            "adf_regression": regression,
            "adf_autolag": autolag,
            "adf_maxlag": maxlag,
            "adf_reject_unit_root": False,
            "adf_pass": False,
        }

    critical_values = result[4]
    p_value = float(result[1])
    reject = bool(p_value < float(significance))
    return {
        "adf_valid": True,
        "adf_reason": "",
        "adf_statistic": float(result[0]),
        "adf_p_value": p_value,
        "adf_used_lags": int(result[2]),
        "adf_observations": int(result[3]),
        "adf_critical_1pct": float(critical_values.get("1%", np.nan)),
        "adf_critical_5pct": float(critical_values.get("5%", np.nan)),
        "adf_critical_10pct": float(critical_values.get("10%", np.nan)),
        "adf_significance": float(significance),
        "adf_regression": regression,
        "adf_autolag": autolag,
        "adf_maxlag": maxlag,
        "adf_reject_unit_root": reject,
        "adf_pass": reject,
    }


def run_adf_for_pair_window(
    price_a: pd.Series | np.ndarray,
    price_b: pd.Series | np.ndarray,
    significance: float = 0.05,
    regression: str = "c",
    autolag: str = "AIC",
    maxlag: int | None = None,
    min_observations: int = 100,
    spread_method: str = "normalized_log",
) -> dict[str, Any]:
    """Construct the project spread and run ADF on one formation window."""

    spread = build_spread(price_a, price_b, method=spread_method)
    return {
        "spread_observations": int(len(spread)),
        "spread_method": spread_method,
        **adf_test(
            spread,
            significance=significance,
            regression=regression,
            autolag=autolag,
            maxlag=maxlag,
            min_observations=min_observations,
        ),
    }


def build_pair_windows_from_panel_dates(
    panel: pd.DataFrame,
    formation_days: int = 30,
    trading_days: int = 10,
    step_days: int | None = None,
) -> pd.DataFrame:
    """Build all ticker-pair rolling windows from processed panel trade dates."""

    if "ticker" not in panel.columns or "trade_date" not in panel.columns:
        raise ValueError("panel must contain ticker and trade_date columns")

    tickers = sorted(pd.Series(panel["ticker"]).dropna().astype(str).str.upper().unique())
    dates = sorted(pd.Series(panel["trade_date"]).dropna().astype(str).unique())
    formation_days = int(formation_days)
    trading_days = int(trading_days)
    step = int(step_days if step_days is not None else 1)
    if formation_days <= 0 or trading_days <= 0 or step <= 0:
        raise ValueError("formation_days, trading_days, and step_days must be positive")

    rows: list[dict[str, Any]] = []
    window_id = 0
    last_start = len(dates) - formation_days - trading_days
    for start_idx in range(0, max(last_start + 1, 0), step):
        formation_start = dates[start_idx]
        formation_end = dates[start_idx + formation_days - 1]
        trading_start = dates[start_idx + formation_days]
        trading_end = dates[start_idx + formation_days + trading_days - 1]
        for ticker_a, ticker_b in combinations(tickers, 2):
            rows.append(
                {
                    "window_id": window_id,
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "formation_start": formation_start,
                    "formation_end": formation_end,
                    "trading_start": trading_start,
                    "trading_end": trading_end,
                }
            )
        window_id += 1
    return pd.DataFrame(rows)


__all__ = ["adf_test", "build_pair_windows_from_panel_dates", "run_adf_for_pair_window"]
