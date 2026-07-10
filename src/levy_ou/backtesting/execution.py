"""Execution and transaction-cost helpers shared by pair backtests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ENDRES_HALF_TURN_BPS = 5.0
ENDRES_HALF_TURN_RATE = ENDRES_HALF_TURN_BPS / 10000.0


def log_return(entry_price: float, exit_price: float, side: int) -> float:
    """Signed log return, with side +1 for long and -1 for short."""

    if entry_price <= 0 or exit_price <= 0:
        return float("nan")
    return float(int(side) * np.log(exit_price / entry_price))


def spread_leg_directions(direction: str) -> tuple[float, float]:
    """Return leg signs for a spread direction.

    ``long_spread`` is long A and short B. ``short_spread`` is short A and
    long B. The signs are expressed from the perspective of price returns.
    """

    if direction == "long_spread":
        return 1.0, -1.0
    if direction == "short_spread":
        return -1.0, 1.0
    raise ValueError("direction must be 'long_spread' or 'short_spread'.")


def execution_prices(
    direction: str,
    entry_bid_a: float,
    entry_ask_a: float,
    entry_bid_b: float,
    entry_ask_b: float,
    exit_bid_a: float,
    exit_ask_a: float,
    exit_bid_b: float,
    exit_ask_b: float,
) -> dict[str, float]:
    """Return actual bid/ask fills for the pair entry and exit."""

    if direction == "long_spread":
        return {
            "entry_exec_a": float(entry_ask_a),
            "entry_exec_b": float(entry_bid_b),
            "exit_exec_a": float(exit_bid_a),
            "exit_exec_b": float(exit_ask_b),
        }
    if direction == "short_spread":
        return {
            "entry_exec_a": float(entry_bid_a),
            "entry_exec_b": float(entry_ask_b),
            "exit_exec_a": float(exit_ask_a),
            "exit_exec_b": float(exit_bid_b),
        }
    raise ValueError("direction must be 'long_spread' or 'short_spread'.")


def fixed_bps_round_trip_cost_pnl(
    entry_mid_a: float,
    entry_mid_b: float,
    exit_mid_a: float,
    exit_mid_b: float,
    half_turn_rate: float = ENDRES_HALF_TURN_RATE,
) -> dict[str, float]:
    """Endres-style fixed bps cost for a dollar-neutral pair trade.

    The convention is a cost of ``half_turn_rate * traded notional`` on every
    leg and every half-turn. With one dollar placed on each leg at entry, entry
    notional is 2.0. Exit notional is based on the same shares marked at exit.
    """

    prices = np.asarray([entry_mid_a, entry_mid_b, exit_mid_a, exit_mid_b], dtype=float)
    if (prices <= 0).any() or not np.isfinite(prices).all() or half_turn_rate < 0:
        return {
            "fixed_bps_entry_cost_pnl_per_dollar_leg": float("nan"),
            "fixed_bps_exit_cost_pnl_per_dollar_leg": float("nan"),
            "fixed_bps_cost_pnl_per_dollar_leg": float("nan"),
            "fixed_bps_cost_return_on_gross": float("nan"),
        }

    entry_notional = 2.0
    exit_notional = float(exit_mid_a / entry_mid_a + exit_mid_b / entry_mid_b)
    entry_cost = float(half_turn_rate * entry_notional)
    exit_cost = float(half_turn_rate * exit_notional)
    total_cost = entry_cost + exit_cost
    return {
        "fixed_bps_half_turn_rate": float(half_turn_rate),
        "fixed_bps_entry_cost_pnl_per_dollar_leg": entry_cost,
        "fixed_bps_exit_cost_pnl_per_dollar_leg": exit_cost,
        "fixed_bps_cost_pnl_per_dollar_leg": total_cost,
        "fixed_bps_cost_return_on_gross": float(total_cost / 2.0),
    }


def execution_pnl(
    direction: str,
    entry_mid_a: float,
    entry_mid_b: float,
    exit_mid_a: float,
    exit_mid_b: float,
    entry_bid_a: float,
    entry_ask_a: float,
    entry_bid_b: float,
    entry_ask_b: float,
    exit_bid_a: float,
    exit_ask_a: float,
    exit_bid_b: float,
    exit_ask_b: float,
    fixed_bps_half_turn_rate: float = ENDRES_HALF_TURN_RATE,
) -> dict[str, float]:
    """Compute midquote, actual bid/ask, and fixed-bps execution PnL.

    PnL is expressed in pair-capital units with one dollar initially allocated
    to each leg. ``return_on_gross`` divides that two-dollar gross by 2.
    """

    direction_a, direction_b = spread_leg_directions(direction)
    mids = np.asarray([entry_mid_a, entry_mid_b, exit_mid_a, exit_mid_b], dtype=float)
    if (mids <= 0).any() or not np.isfinite(mids).all():
        nan_fields = {
            "midquote_pnl_per_dollar_leg": float("nan"),
            "midquote_return_on_gross": float("nan"),
            "bid_ask_pnl_per_dollar_leg": float("nan"),
            "bid_ask_return_on_gross": float("nan"),
            "midquote_fixed_bps_pnl_per_dollar_leg": float("nan"),
            "midquote_fixed_bps_return_on_gross": float("nan"),
            "execution_cost_vs_midquote_pnl": float("nan"),
        }
        return {**nan_fields, **fixed_bps_round_trip_cost_pnl(*mids)}

    fills = execution_prices(
        direction=direction,
        entry_bid_a=entry_bid_a,
        entry_ask_a=entry_ask_a,
        entry_bid_b=entry_bid_b,
        entry_ask_b=entry_ask_b,
        exit_bid_a=exit_bid_a,
        exit_ask_a=exit_ask_a,
        exit_bid_b=exit_bid_b,
        exit_ask_b=exit_ask_b,
    )

    mid_pnl = direction_a * (float(exit_mid_a) / float(entry_mid_a) - 1.0)
    mid_pnl += direction_b * (float(exit_mid_b) / float(entry_mid_b) - 1.0)

    entry_exec_a = fills["entry_exec_a"]
    entry_exec_b = fills["entry_exec_b"]
    exit_exec_a = fills["exit_exec_a"]
    exit_exec_b = fills["exit_exec_b"]
    exec_prices = np.asarray([entry_exec_a, entry_exec_b, exit_exec_a, exit_exec_b], dtype=float)
    if (exec_prices <= 0).any() or not np.isfinite(exec_prices).all():
        bidask_pnl = float("nan")
    else:
        bidask_pnl = direction_a * (exit_exec_a / entry_exec_a - 1.0)
        bidask_pnl += direction_b * (exit_exec_b / entry_exec_b - 1.0)

    fixed_cost = fixed_bps_round_trip_cost_pnl(
        entry_mid_a=entry_mid_a,
        entry_mid_b=entry_mid_b,
        exit_mid_a=exit_mid_a,
        exit_mid_b=exit_mid_b,
        half_turn_rate=fixed_bps_half_turn_rate,
    )
    fixed_total = float(fixed_cost["fixed_bps_cost_pnl_per_dollar_leg"])
    fixed_pnl = float(mid_pnl - fixed_total) if np.isfinite(fixed_total) else float("nan")

    return {
        **fills,
        "midquote_pnl_per_dollar_leg": float(mid_pnl),
        "midquote_return_on_gross": float(mid_pnl / 2.0),
        "bid_ask_pnl_per_dollar_leg": float(bidask_pnl),
        "bid_ask_return_on_gross": float(bidask_pnl / 2.0),
        "execution_cost_vs_midquote_pnl": float(bidask_pnl - mid_pnl),
        **fixed_cost,
        "midquote_fixed_bps_pnl_per_dollar_leg": fixed_pnl,
        "midquote_fixed_bps_return_on_gross": float(fixed_pnl / 2.0),
    }


def pair_log_bidask_cost(
    ask_a: np.ndarray,
    bid_a: np.ndarray,
    ask_b: np.ndarray,
    bid_b: np.ndarray,
) -> np.ndarray:
    """Return log(ask_A/bid_A) + log(ask_B/bid_B)."""

    ask_a = np.asarray(ask_a, dtype=float)
    bid_a = np.asarray(bid_a, dtype=float)
    ask_b = np.asarray(ask_b, dtype=float)
    bid_b = np.asarray(bid_b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(ask_a / bid_a) + np.log(ask_b / bid_b)
    return out


def formation_log_bidask_costs(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    formation_start: str,
    formation_end: str,
    bid_suffix: str = "_bid_close",
    ask_suffix: str = "_ask_close",
) -> dict[str, Any]:
    """Summarise formation-window bid/ask log costs for one pair. cost-aware signal/threshold design"""
    

    ticker_a = str(ticker_a).upper()
    ticker_b = str(ticker_b).upper()
    needed = [
        f"{ticker_a}{bid_suffix}",
        f"{ticker_a}{ask_suffix}",
        f"{ticker_b}{bid_suffix}",
        f"{ticker_b}{ask_suffix}",
    ]
    missing = [col for col in needed + ["trade_date"] if col not in panel.columns]
    if missing:
        return {"valid_cost": False, "cost_reason": f"missing columns {missing}"}

    date_mask = panel["trade_date"].astype(str).between(
    str(formation_start),
    str(formation_end),
    )

    formation_panel = panel.loc[date_mask]
    frame = formation_panel[needed].copy()

    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[(frame[needed] > 0).all(axis=1)]
    if frame.empty:
        return {"valid_cost": False, "cost_reason": "empty positive formation bid/ask frame"}

    costs = pair_log_bidask_cost(
        ask_a=frame[f"{ticker_a}{ask_suffix}"],
        bid_a=frame[f"{ticker_a}{bid_suffix}"],
        ask_b=frame[f"{ticker_b}{ask_suffix}"],
        bid_b=frame[f"{ticker_b}{bid_suffix}"],
    )
    costs = costs[np.isfinite(costs) & (costs >= 0.0)]
    if len(costs) == 0:
        return {"valid_cost": False, "cost_reason": "no finite non-negative costs"}

    return {
        "valid_cost": True,
        "cost_reason": "",
        "c_median": float(np.median(costs)),
        "c_10x_median": float(10.0 * np.median(costs)),
        "c_worst": float(np.max(costs)),
        "c_mean": float(np.mean(costs)),
        "c_p75": float(np.quantile(costs, 0.75)),
        "c_p90": float(np.quantile(costs, 0.90)),
        "cost_observations": int(len(costs)),
        "cost_definition": "formation log(ask_A/bid_A)+log(ask_B/bid_B)",
    }


__all__ = [
    "ENDRES_HALF_TURN_BPS",
    "ENDRES_HALF_TURN_RATE",
    "execution_pnl",
    "execution_prices",
    "fixed_bps_round_trip_cost_pnl",
    "formation_log_bidask_costs",
    "log_return",
    "pair_log_bidask_cost",
    "spread_leg_directions",
]
