"""Real processed-LOBSTER helpers for small reproducible experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from levy_ou.backtesting.execution import pair_log_bidask_cost
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


def select_pair_windows(
    selection_csv: str | Path,
    n_pairs: int = 5,
    n_windows: int = 5,
    selection_mode: str = "first_window_pairs",
) -> pd.DataFrame:
    """Select pair-window rows for a real-data experiment.

    ``first_window_pairs`` keeps the historical smoke-test behavior: choose the
    first ``n_pairs`` from the first selected window and follow those pairs
    across the first ``n_windows``.

    ``as_is`` preserves the rows already present in ``selection_csv`` after an
    optional first-``n_windows`` filter. Use this when a previous step has
    already selected top-N pairs inside each window, for example ADF-filtered
    top-10 selections.
    """

    selected = pd.read_csv(selection_csv)
    required = {"window_id", "ticker_a", "ticker_b", "formation_start", "formation_end", "trading_start", "trading_end"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"selection CSV is missing columns: {missing}")

    selected["ticker_a"] = selected["ticker_a"].astype(str).str.upper()
    selected["ticker_b"] = selected["ticker_b"].astype(str).str.upper()
    window_ids = sorted(pd.unique(selected["window_id"]))[: int(n_windows)] if int(n_windows) > 0 else sorted(pd.unique(selected["window_id"]))
    if selection_mode == "as_is":
        return (
            selected[selected["window_id"].isin(window_ids)]
            .copy()
            .sort_values(["window_id", "ticker_a", "ticker_b"])
            .reset_index(drop=True)
        )
    if selection_mode != "first_window_pairs":
        raise ValueError("selection_mode must be 'first_window_pairs' or 'as_is'.")

    first_window = selected[selected["window_id"].eq(window_ids[0])]
    pairs = first_window[["ticker_a", "ticker_b"]].drop_duplicates().head(int(n_pairs))
    pair_keys = {(str(row.ticker_a).upper(), str(row.ticker_b).upper()) for row in pairs.itertuples(index=False)}

    out = selected[selected["window_id"].isin(window_ids)].copy()
    out = out[out[["ticker_a", "ticker_b"]].apply(tuple, axis=1).isin(pair_keys)]
    return out.sort_values(["window_id", "ticker_a", "ticker_b"]).reset_index(drop=True)


def load_lobster_panel(
    data_path: str | Path,
    tickers: list[str] | tuple[str, ...] | set[str],
    start_date: str,
    end_date: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load a filtered processed LOBSTER panel for selected tickers/dates."""

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


def _pivot_pair(
    panel: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
    anchor_a: float | None = None,
    anchor_b: float | None = None,
) -> pd.DataFrame:
    ticker_a = str(ticker_a).upper()
    ticker_b = str(ticker_b).upper()
    frame = panel[
        panel["ticker"].isin([ticker_a, ticker_b])
        & panel["trade_date"].astype(str).between(str(start), str(end))
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    base = frame[["timestamp_utc", "timestamp_ny", "trade_date", "market_time"]].drop_duplicates("timestamp_utc")
    base = base.sort_values("timestamp_utc").set_index("timestamp_utc")
    values = {}
    for col, prefix in [
        ("model_price_close", "price"),
        ("mid_close", "mid"),
        ("bid_close", "bid"),
        ("ask_close", "ask"),
    ]:
        pivot = frame.pivot_table(index="timestamp_utc", columns="ticker", values=col, aggfunc="last")
        values[f"{prefix}_a"] = pivot.get(ticker_a)
        values[f"{prefix}_b"] = pivot.get(ticker_b)

    out = base.join(pd.DataFrame(values))
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
    return out.reset_index()


def pair_formation_and_trading_frames(panel: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned formation and trading frames for one selected pair-window."""

    formation = _pivot_pair(panel, row["ticker_a"], row["ticker_b"], row["formation_start"], row["formation_end"])
    if formation.empty:
        return formation, pd.DataFrame()
    anchor_a = float(formation["price_a"].iloc[0])
    anchor_b = float(formation["price_b"].iloc[0])
    trading = _pivot_pair(
        panel,
        row["ticker_a"],
        row["ticker_b"],
        row["trading_start"],
        row["trading_end"],
        anchor_a=anchor_a,
        anchor_b=anchor_b,
    )
    return formation, trading


def formation_cost_cases(formation: pd.DataFrame) -> dict[str, float]:
    """Return optimisation cost cases from formation bid/ask quotes."""

    if formation.empty:
        return {"bidask_median_c": np.nan, "bidask_worst_c": np.nan}
    costs = pair_log_bidask_cost(
        ask_a=formation["ask_a"],
        bid_a=formation["bid_a"],
        ask_b=formation["ask_b"],
        bid_b=formation["bid_b"],
    )
    costs = costs[np.isfinite(costs) & (costs >= 0.0)]
    if len(costs) == 0:
        return {"bidask_median_c": np.nan, "bidask_worst_c": np.nan}
    return {
        "bidask_median_c": float(np.median(costs)),
        "bidask_worst_c": float(np.max(costs)),
    }


def scalar_trade_profit_summary(trades: pd.DataFrame) -> dict[str, Any]:
    """Summarise realised trade profits for one model/window/cost case."""

    if trades.empty:
        return {
            "num_trades": 0,
            "forced_exits": 0,
            "midquote_total_return_on_gross": 0.0,
            "midquote_fixed_bps_total_return_on_gross": 0.0,
            "bid_ask_total_return_on_gross": 0.0,
        }
    return {
        "num_trades": int(len(trades)),
        "forced_exits": int(pd.Series(trades["forced_exit"]).astype(bool).sum()),
        "midquote_total_return_on_gross": float(pd.to_numeric(trades["midquote_return_on_gross"], errors="coerce").sum()),
        "midquote_fixed_bps_total_return_on_gross": float(
            pd.to_numeric(trades["midquote_fixed_bps_return_on_gross"], errors="coerce").sum()
        ),
        "bid_ask_total_return_on_gross": float(pd.to_numeric(trades["bid_ask_return_on_gross"], errors="coerce").sum()),
    }


__all__ = [
    "formation_cost_cases",
    "load_lobster_panel",
    "pair_formation_and_trading_frames",
    "scalar_trade_profit_summary",
    "select_pair_windows",
]
