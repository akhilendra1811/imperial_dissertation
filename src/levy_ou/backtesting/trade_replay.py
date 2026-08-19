"""Generic threshold trade replay for OU pair-spread backtests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .execution import ENDRES_HALF_TURN_RATE, execution_pnl


def _row_value(row: pd.Series, name: str, default: Any = None) -> Any:
    return row[name] if name in row.index else default


def _quote(row: pd.Series, column: str, fallback: str) -> float:
    value = _row_value(row, column, None)
    if value is None or pd.isna(value):
        value = _row_value(row, fallback, np.nan)
    return float(value)


def _mean_for_row(
    row: pd.Series,
    mean: float,
    mean_col: str | None,
    mean_mode: str,
    entry_mean: float | None = None,
) -> float:
    if mean_mode == "constant":
        return float(mean)
    if mean_mode == "moving":
        if mean_col is None:
            return float(mean)
        return float(_row_value(row, mean_col, mean))
    if mean_mode == "frozen_entry":
        return float(entry_mean if entry_mean is not None else mean)
    raise ValueError("mean_mode must be 'constant', 'moving', or 'frozen_entry'.")


def trade_band_window(
    frame: pd.DataFrame,
    mean: float,
    d_plus: float,
    d_minus: float | None = None,
    exit_rule: str = "mean",
    mean_col: str | None = None,
    mean_mode: str = "constant",
    spread_col: str = "spread",
    fixed_bps_half_turn_rate: float = ENDRES_HALF_TURN_RATE,
    ticker_a: str | None = None,
    ticker_b: str | None = None,
) -> list[dict[str, Any]]:
    """Replay one pair-window using fixed upper/lower distances from the mean.

    The function is deliberately model-agnostic. Gaussian, NIG, CGMY, and symmetric BG
    runners should supply the fitted/formation mean and threshold distances.
    """

    if exit_rule not in {"mean", "opposite_band"}:
        raise ValueError("exit_rule must be 'mean' or 'opposite_band'.")
    if mean_mode not in {"constant", "moving", "frozen_entry"}:
        raise ValueError("mean_mode must be 'constant', 'moving', or 'frozen_entry'.")
    if spread_col not in frame.columns:
        raise ValueError(f"frame is missing spread column {spread_col!r}.")

    upper_distance = float(d_plus)
    lower_distance = float(d_minus if d_minus is not None else d_plus)
    if not (np.isfinite(upper_distance) and np.isfinite(lower_distance)):
        raise ValueError("threshold distances must be finite.")
    if upper_distance < 0 or lower_distance < 0:
        raise ValueError("threshold distances must be non-negative.")

    data = frame.reset_index(drop=True).copy()
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []

    def close_trade(i: int, forced: bool) -> None:
        nonlocal position
        assert position is not None
        row = data.iloc[i]
        direction = str(position["direction"])
        pnl = execution_pnl(
            direction=direction,
            entry_mid_a=float(position["entry_mid_a"]),
            entry_mid_b=float(position["entry_mid_b"]),
            exit_mid_a=_quote(row, "mid_a", "price_a"),
            exit_mid_b=_quote(row, "mid_b", "price_b"),
            entry_bid_a=float(position["entry_bid_a"]),
            entry_ask_a=float(position["entry_ask_a"]),
            entry_bid_b=float(position["entry_bid_b"]),
            entry_ask_b=float(position["entry_ask_b"]),
            exit_bid_a=_quote(row, "bid_a", "mid_a"),
            exit_ask_a=_quote(row, "ask_a", "mid_a"),
            exit_bid_b=_quote(row, "bid_b", "mid_b"),
            exit_ask_b=_quote(row, "ask_b", "mid_b"),
            fixed_bps_half_turn_rate=fixed_bps_half_turn_rate,
        )

        exit_spread = float(row[spread_col])
        entry_spread = float(position["entry_spread"])
        entry_mean = float(position["entry_mean"])
        current_mean = _mean_for_row(row, mean=float(mean), mean_col=mean_col, mean_mode=mean_mode, entry_mean=entry_mean)
        spread_pnl = exit_spread - entry_spread if direction == "long_spread" else entry_spread - exit_spread
        trades.append(
            {
                **position,
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "exit_i": int(i),
                "exit_timestamp_utc": _row_value(row, "timestamp_utc", i),
                "exit_timestamp_ny": _row_value(row, "timestamp_ny", _row_value(row, "timestamp_utc", i)),
                "exit_trade_date": _row_value(row, "trade_date", ""),
                "exit_market_time": _row_value(row, "market_time", ""),
                "exit_spread": exit_spread,
                "exit_mean": float(current_mean),
                "exit_spread_deviation": float(exit_spread - current_mean),
                "exit_mid_a": _quote(row, "mid_a", "price_a"),
                "exit_mid_b": _quote(row, "mid_b", "price_b"),
                "exit_reason": (
                    "forced exit at trading-window end"
                    if forced
                    else ("crossed mean" if exit_rule == "mean" else "crossed opposite band")
                ),
                "forced_exit": bool(forced),
                "exit_rule": exit_rule,
                "mean_mode": mean_mode,
                "duration_minutes": int(i - int(position["entry_i"])),
                "d_plus": upper_distance,
                "d_minus": lower_distance,
                "spread_pnl": float(spread_pnl),
                **pnl,
            }
        )
        position = None

    for i, row in data.iterrows():
        spread = float(row[spread_col])
        if not np.isfinite(spread):
            continue

        if position is None:
            current_mean = _mean_for_row(row, mean=float(mean), mean_col=mean_col, mean_mode="moving" if mean_mode == "moving" else "constant")
            deviation = spread - current_mean
            direction = None
            if deviation <= -lower_distance:
                direction = "long_spread"
            elif deviation >= upper_distance:
                direction = "short_spread"
            if direction is not None:
                mid_a = _quote(row, "mid_a", "price_a")
                mid_b = _quote(row, "mid_b", "price_b")
                position = {
                    "direction": direction,
                    "entry_i": int(i),
                    "entry_timestamp_utc": _row_value(row, "timestamp_utc", i),
                    "entry_timestamp_ny": _row_value(row, "timestamp_ny", _row_value(row, "timestamp_utc", i)),
                    "entry_trade_date": _row_value(row, "trade_date", ""),
                    "entry_market_time": _row_value(row, "market_time", ""),
                    "entry_spread": spread,
                    "entry_mean": float(current_mean),
                    "entry_spread_deviation": float(deviation),
                    "entry_mid_a": mid_a,
                    "entry_mid_b": mid_b,
                    "entry_bid_a": _quote(row, "bid_a", "mid_a"),
                    "entry_ask_a": _quote(row, "ask_a", "mid_a"),
                    "entry_bid_b": _quote(row, "bid_b", "mid_b"),
                    "entry_ask_b": _quote(row, "ask_b", "mid_b"),
                    "entry_reason": "crossed entry band",
                }
            continue

        entry_mean = float(position["entry_mean"])
        current_mean = _mean_for_row(row, mean=float(mean), mean_col=mean_col, mean_mode=mean_mode, entry_mean=entry_mean)
        deviation = spread - current_mean
        direction = str(position["direction"])
        if exit_rule == "mean":
            should_exit = (direction == "long_spread" and deviation >= 0.0) or (
                direction == "short_spread" and deviation <= 0.0
            )
        else:
            should_exit = (direction == "long_spread" and deviation >= upper_distance) or (
                direction == "short_spread" and deviation <= -lower_distance
            )
        if should_exit:
            close_trade(i, forced=False)

    if position is not None and len(data):
        close_trade(len(data) - 1, forced=True)
    return trades


def trade_real_window(
    trading: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    mu: float,
    d_plus: float,
    d_minus: float,
    exit_rule: str = "mean",
    fixed_bps_half_turn_rate: float = ENDRES_HALF_TURN_RATE,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for the NIG/CGMY simulated-threshold runners."""

    return trade_band_window(
        frame=trading,
        mean=float(mu),
        d_plus=float(d_plus),
        d_minus=float(d_minus),
        exit_rule=exit_rule,
        mean_mode="constant",
        fixed_bps_half_turn_rate=fixed_bps_half_turn_rate,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
    )


__all__ = ["trade_band_window", "trade_real_window"]
