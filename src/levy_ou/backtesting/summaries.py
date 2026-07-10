"""Generate daily, summed, and annualised return summaries for pair backtests.
If a trade lasts several days, the code estimates daily PnL using daily close prices between entry and exit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ANNUAL_TRADING_DAYS = 252


def geometric_annualised_return(
    daily_returns: pd.Series | np.ndarray,
    trading_days: int = DEFAULT_ANNUAL_TRADING_DAYS,
) -> float:
    """Geometric annualised return from simple daily returns."""

    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    if np.any(r <= -1.0):
        return float("nan")
    return float(np.prod(1.0 + r) ** (float(trading_days) / len(r)) - 1.0)


def annualised_metrics(
    returns: pd.Series | np.ndarray,
    trading_days: int = DEFAULT_ANNUAL_TRADING_DAYS,
) -> dict[str, float | int]:
    """Daily return metrics using geometric annualisation."""

    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "return_days": 0,
            "mean_daily_return": np.nan,
            "annualised_return_geometric": np.nan,
            "annualised_return_arithmetic": np.nan,
            "annualised_volatility": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "positive_day_share": np.nan,
        }
    wealth = np.cumprod(1.0 + x)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    drawdowns = 1.0 - wealth / peaks
    return {
        "return_days": int(len(x)),
        "mean_daily_return": float(np.mean(x)),
        "annualised_return_geometric": geometric_annualised_return(x, trading_days=trading_days),
        "annualised_return_arithmetic": float(np.mean(x) * float(trading_days)),
        "annualised_volatility": float(np.std(x, ddof=1) * np.sqrt(float(trading_days))) if len(x) > 1 else np.nan,
        "cumulative_return": float(np.prod(1.0 + x) - 1.0),
        "max_drawdown": float(np.max(drawdowns)),
        "positive_day_share": float(np.mean(x > 0.0)),
    }


def daily_close_panel_from_combined(
    panel: pd.DataFrame,
    metadata_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract daily close prices from a combined model-ready panel."""

    metadata = metadata_cols or {"timestamp_utc", "timestamp_ny", "trade_date", "market_time"}
    tickers = [
        col
        for col in panel.columns
        if col not in metadata and not col.endswith("_bid_close") and not col.endswith("_ask_close")
    ]
    closes = panel.groupby("trade_date", sort=True)[tickers].last()
    closes.index = closes.index.astype(str)
    return closes, list(closes.index.astype(str))


def expand_denominators(
    windows: pd.DataFrame,
    trades: pd.DataFrame,
    calendar_dates: list[str],
) -> pd.DataFrame:
    """Build committed/employed pair denominators for daily returns."""

    if windows.empty:
        return pd.DataFrame(columns=["date", "committed_pairs", "employed_pairs"])
    active = set()
    if not trades.empty and {"window_id", "ticker_a", "ticker_b"}.issubset(trades.columns):
        active = set(
            zip(
                trades["window_id"].astype(int),
                trades["ticker_a"].astype(str),
                trades["ticker_b"].astype(str),
            )
        )

    rows: list[dict[str, Any]] = []
    key_cols = ["window_id", "ticker_a", "ticker_b", "trading_start", "trading_end"]
    for row in windows.drop_duplicates([col for col in key_cols if col in windows.columns]).itertuples(index=False):
        values = row._asdict()
        key = (int(values["window_id"]), str(values["ticker_a"]), str(values["ticker_b"]))
        for date in calendar_dates:
            if str(values["trading_start"]) <= date <= str(values["trading_end"]):
                rows.append({"date": date, "committed": 1, "employed": int(key in active)})
    if not rows:
        return pd.DataFrame(columns=["date", "committed_pairs", "employed_pairs"])
    return (
        pd.DataFrame(rows)
        .groupby("date", as_index=False)
        .agg(committed_pairs=("committed", "sum"), employed_pairs=("employed", "sum"))
    )


def expand_trade_marks_multi(
    trades: pd.DataFrame,
    closes: pd.DataFrame,
    calendar_dates: list[str],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Mark trades to daily closes and reconstruct daily PnL by execution case."""

    output_columns = [
        "date",
        "daily_profit_midquote",
        "daily_profit_bid_ask",
        "daily_profit_midquote_fixed_bps",
        "open_trade_marks",
        "trades_entered",
        "trades_exited",
    ]
    if trades.empty:
        return pd.DataFrame(columns=output_columns), {
            "trades": 0,
            "missing_intermediate_daily_marks": 0,
        }

    rows: list[dict[str, Any]] = []
    missing_marks = 0
    final_mid: list[float] = []
    final_bidask: list[float] = []
    final_fixed: list[float] = []

    for trade_id, trade in trades.reset_index(drop=True).iterrows():
        dates = [
            date
            for date in calendar_dates
            if str(trade.entry_trade_date) <= date <= str(trade.exit_trade_date)
        ]
        direction_a = 1.0 if trade.direction == "long_spread" else -1.0
        direction_b = -direction_a
        previous_mid = 0.0
        previous_bidask = 0.0
        previous_fixed = 0.0
        entry_cost = float(getattr(trade, "fixed_bps_entry_cost_pnl_per_dollar_leg", 0.0))
        exit_cost = float(getattr(trade, "fixed_bps_exit_cost_pnl_per_dollar_leg", 0.0))

        for date in dates:
            is_entry = date == str(trade.entry_trade_date)
            is_exit = date == str(trade.exit_trade_date)
            if is_exit:
                mark_mid_a = float(trade.exit_mid_a)
                mark_mid_b = float(trade.exit_mid_b)
                mark_exec_a = float(trade.exit_exec_a)
                mark_exec_b = float(trade.exit_exec_b)
            else:
                mark_mid_a = float(closes.at[date, trade.ticker_a])
                mark_mid_b = float(closes.at[date, trade.ticker_b])
                mark_exec_a = mark_mid_a
                mark_exec_b = mark_mid_b
                if not np.isfinite(mark_mid_a) or not np.isfinite(mark_mid_b):
                    missing_marks += 1

            cumulative_mid = direction_a * (mark_mid_a / float(trade.entry_mid_a) - 1.0)
            cumulative_mid += direction_b * (mark_mid_b / float(trade.entry_mid_b) - 1.0)
            cumulative_bidask = direction_a * (mark_exec_a / float(trade.entry_exec_a) - 1.0)
            cumulative_bidask += direction_b * (mark_exec_b / float(trade.entry_exec_b) - 1.0)
            cumulative_fixed = cumulative_mid - entry_cost - (exit_cost if is_exit else 0.0)

            rows.append(
                {
                    "trade_id": int(trade_id),
                    "date": date,
                    "daily_profit_midquote": float(cumulative_mid - previous_mid),
                    "daily_profit_bid_ask": float(cumulative_bidask - previous_bidask),
                    "daily_profit_midquote_fixed_bps": float(cumulative_fixed - previous_fixed),
                    "trade_entered": int(is_entry),
                    "trade_exited": int(is_exit),
                    "open_trade_mark": int(not is_exit),
                }
            )
            previous_mid = cumulative_mid
            previous_bidask = cumulative_bidask
            previous_fixed = cumulative_fixed

        final_mid.append(previous_mid)
        final_bidask.append(previous_bidask)
        final_fixed.append(previous_fixed)

    expanded = pd.DataFrame(rows)
    daily = expanded.groupby("date", as_index=False).agg(
        daily_profit_midquote=("daily_profit_midquote", "sum"),
        daily_profit_bid_ask=("daily_profit_bid_ask", "sum"),
        daily_profit_midquote_fixed_bps=("daily_profit_midquote_fixed_bps", "sum"),
        open_trade_marks=("open_trade_mark", "sum"),
        trades_entered=("trade_entered", "sum"),
        trades_exited=("trade_exited", "sum"),
    )
    reconciliation = {
        "trades": int(len(trades)),
        "stored_midquote_pnl": float(trades["midquote_pnl_per_dollar_leg"].sum()),
        "reconstructed_midquote_pnl": float(np.sum(final_mid)),
        "midquote_difference": float(np.sum(final_mid) - trades["midquote_pnl_per_dollar_leg"].sum()),
        "stored_bid_ask_pnl": float(trades["bid_ask_pnl_per_dollar_leg"].sum()),
        "reconstructed_bid_ask_pnl": float(np.sum(final_bidask)),
        "bid_ask_difference": float(np.sum(final_bidask) - trades["bid_ask_pnl_per_dollar_leg"].sum()),
        "stored_midquote_fixed_bps_pnl": float(trades["midquote_fixed_bps_pnl_per_dollar_leg"].sum()),
        "reconstructed_midquote_fixed_bps_pnl": float(np.sum(final_fixed)),
        "midquote_fixed_bps_difference": float(
            np.sum(final_fixed) - trades["midquote_fixed_bps_pnl_per_dollar_leg"].sum()
        ),
        "missing_intermediate_daily_marks": int(missing_marks),
    }
    return daily, reconciliation


DEFAULT_EXECUTION_SPECS = (
    ("midquote_c0_realized", "daily_profit_midquote", "return_committed_midquote", "return_employed_midquote"),
    (
        "midquote_fixed_bps_realized",
        "daily_profit_midquote_fixed_bps",
        "return_committed_midquote_fixed_bps",
        "return_employed_midquote_fixed_bps",
    ),
    ("lobster_bid_ask_realized", "daily_profit_bid_ask", "return_committed_bid_ask", "return_employed_bid_ask"),
)


def daily_outputs(
    windows: pd.DataFrame,
    trades: pd.DataFrame,
    closes: pd.DataFrame,
    calendar_dates: list[str],
    group_cols: list[str] | None = None,
    strategy: str = "ou_pair_backtest",
    trading_days: int = DEFAULT_ANNUAL_TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create daily returns, annualised metrics, and reconciliation tables."""

    group_cols = group_cols or [
        col
        for col in ["optimization_case", "optimization_cost_case", "gamma_multiplier"]
        if col in windows.columns
    ]
    if not group_cols:
        windows = windows.copy()
        trades = trades.copy()
        windows["_group"] = "all"
        if not trades.empty:
            trades["_group"] = "all"
        group_cols = ["_group"]

    daily_frames: list[pd.DataFrame] = []
    annual_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []

    for key, w in windows.groupby(group_cols, sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_values = dict(zip(group_cols, key_tuple, strict=False))
        if trades.empty:
            t = pd.DataFrame()
        else:
            mask = np.ones(len(trades), dtype=bool)
            for col, value in key_values.items():
                mask &= trades[col].eq(value).to_numpy() if col in trades.columns else False
            t = trades.loc[mask].copy()

        denominators = expand_denominators(w, t, calendar_dates)
        daily_pnl, reconciliation = expand_trade_marks_multi(t, closes, calendar_dates)
        daily = denominators.merge(daily_pnl, on="date", how="left")
        fill_cols = [
            "committed_pairs",
            "employed_pairs",
            "daily_profit_midquote",
            "daily_profit_bid_ask",
            "daily_profit_midquote_fixed_bps",
            "open_trade_marks",
            "trades_entered",
            "trades_exited",
        ]
        for fill_col in fill_cols:
            if fill_col in daily.columns:
                daily[fill_col] = pd.to_numeric(daily[fill_col], errors="coerce").fillna(0.0)
        daily["strategy"] = strategy
        for col, value in key_values.items():
            daily[col] = value
        for _case, profit_col, committed_col, employed_col in DEFAULT_EXECUTION_SPECS:
            daily[committed_col] = np.where(daily["committed_pairs"] > 0, daily[profit_col] / daily["committed_pairs"], np.nan)
            daily[employed_col] = np.where(daily["employed_pairs"] > 0, daily[profit_col] / daily["employed_pairs"], np.nan)
        daily_frames.append(daily)

        for execution_case, profit_col, committed_col, employed_col in DEFAULT_EXECUTION_SPECS:
            committed = annualised_metrics(daily[committed_col], trading_days=trading_days)
            employed = annualised_metrics(daily[employed_col], trading_days=trading_days)
            annual_rows.append(
                {
                    "strategy": strategy,
                    **key_values,
                    "execution_case": execution_case,
                    "trades": int(len(t)),
                    "trading_days": int(len(daily)),
                    "average_committed_pairs": float(daily["committed_pairs"].mean()) if len(daily) else np.nan,
                    "average_employed_pairs": float(daily["employed_pairs"].mean()) if len(daily) else np.nan,
                    "active_pair_share": float(daily["employed_pairs"].sum() / daily["committed_pairs"].sum())
                    if len(daily) and float(daily["committed_pairs"].sum())
                    else np.nan,
                    "total_profit_pair_capital_units": float(daily[profit_col].sum()) if len(daily) else 0.0,
                    **{f"committed_{metric}": value for metric, value in committed.items()},
                    **{f"employed_{metric}": value for metric, value in employed.items()},
                }
            )
        reconciliation_rows.append({**key_values, **reconciliation})

    return (
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        pd.DataFrame(annual_rows),
        pd.DataFrame(reconciliation_rows),
    )


def summarize_summed(
    windows: pd.DataFrame,
    trades: pd.DataFrame,
    group_cols: list[str] | None = None,
    strategy: str = "ou_pair_backtest",
) -> pd.DataFrame:
    """Trade-level summed summaries for realised execution cases."""

    group_cols = group_cols or [
        col
        for col in ["optimization_case", "optimization_cost_case", "gamma_multiplier"]
        if col in windows.columns
    ]
    if not group_cols:
        windows = windows.copy()
        trades = trades.copy()
        windows["_group"] = "all"
        if not trades.empty:
            trades["_group"] = "all"
        group_cols = ["_group"]

    execution_cases = {
        "midquote_c0_realized": "midquote_return_on_gross",
        "midquote_fixed_bps_realized": "midquote_fixed_bps_return_on_gross",
        "lobster_bid_ask_realized": "bid_ask_return_on_gross",
    }
    rows: list[dict[str, Any]] = []
    for key, w in windows.groupby(group_cols, sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_values = dict(zip(group_cols, key_tuple, strict=False))
        if trades.empty:
            t_all = pd.DataFrame()
        else:
            mask = np.ones(len(trades), dtype=bool)
            for col, value in key_values.items():
                mask &= trades[col].eq(value).to_numpy() if col in trades.columns else False
            t_all = trades.loc[mask].copy()
        for execution_case, return_col in execution_cases.items():
            total = int(len(t_all))
            rows.append(
                {
                    "strategy": strategy,
                    **key_values,
                    "execution_case": execution_case,
                    "pair_windows": int(len(w)),
                    "calendar_windows": int(w["window_id"].nunique()) if "window_id" in w else np.nan,
                    "successful_threshold_windows": int(w["success"].sum()) if "success" in w else np.nan,
                    "windows_with_trades": int((w["num_trades"] > 0).sum()) if "num_trades" in w else np.nan,
                    "total_trades": total,
                    "forced_exits": int(t_all["forced_exit"].sum()) if total and "forced_exit" in t_all else 0,
                    "forced_exit_rate": float(t_all["forced_exit"].mean()) if total and "forced_exit" in t_all else np.nan,
                    "winning_trades": int((t_all[return_col] > 0).sum()) if total and return_col in t_all else 0,
                    "win_rate": float((t_all[return_col] > 0).mean()) if total and return_col in t_all else np.nan,
                    "summed_return_on_gross": float(t_all[return_col].sum()) if total and return_col in t_all else 0.0,
                    "mean_trade_return_on_gross": float(t_all[return_col].mean()) if total and return_col in t_all else np.nan,
                    "median_trade_return_on_gross": float(t_all[return_col].median()) if total and return_col in t_all else np.nan,
                    "median_duration_minutes": float(t_all["duration_minutes"].median()) if total and "duration_minutes" in t_all else np.nan,
                    "median_threshold_c_bps": float(w["threshold_transaction_cost_c_bps"].median())
                    if "threshold_transaction_cost_c_bps" in w
                    else np.nan,
                    "median_full_band_bps": float(w["full_band_bps"].median()) if "full_band_bps" in w else np.nan,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_ANNUAL_TRADING_DAYS",
    "DEFAULT_EXECUTION_SPECS",
    "annualised_metrics",
    "daily_close_panel_from_combined",
    "daily_outputs",
    "expand_denominators",
    "expand_trade_marks_multi",
    "geometric_annualised_return",
    "summarize_summed",
]
