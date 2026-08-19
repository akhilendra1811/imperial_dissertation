"""Pair-window ranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_gaussian_endres_rank(
    frame: pd.DataFrame,
    *,
    window_col: str = "window_id",
    theta_col: str = "theta",
    sigma_col: str = "sigma_eq",
    ticker_a_col: str = "ticker_a",
    ticker_b_col: str = "ticker_b",
    valid_col: str = "valid",
    trading_eligible_col: str = "trading_eligible",
    top_n: int | None = 10,
    rank_col: str = "gaussian_endres_selection_rank",
    selected_col: str = "selected_gaussian_top10_unfiltered",
) -> pd.DataFrame:
    """Add Gaussian Endres-Stuebinger pair ranks within each formation window.

    Pairs are ranked separately by Gaussian OU mean-reversion speed and
    stationary spread standard deviation, with larger values preferred. The
    two parameter ranks are summed, then ties are resolved by higher speed,
    higher volatility, and finally alphabetical pair identifiers.
    """

    required = {window_col, theta_col, sigma_col, ticker_a_col, ticker_b_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing columns required for Gaussian Endres ranking: {missing}")

    out = frame.copy()
    theta = pd.to_numeric(out[theta_col], errors="coerce")
    sigma = pd.to_numeric(out[sigma_col], errors="coerce")
    rankable = theta.notna() & sigma.notna() & np.isfinite(theta) & np.isfinite(sigma)

    if valid_col in out.columns:
        rankable &= _truthy_series(out[valid_col])
    if trading_eligible_col in out.columns:
        rankable &= _truthy_series(out[trading_eligible_col])

    out["is_rankable_estimate"] = rankable
    out["r_theta"] = np.nan
    out["r_sigma"] = np.nan
    out["endres_parameter_rank_sum"] = np.nan
    out["endres_joint_model_rank"] = np.nan
    out[rank_col] = np.nan

    for _, index in out[rankable].groupby(window_col, sort=True).groups.items():
        idx = list(index)
        ranked = out.loc[idx].copy()
        ranked["_theta_numeric"] = theta.loc[idx].to_numpy(dtype=float)
        ranked["_sigma_numeric"] = sigma.loc[idx].to_numpy(dtype=float)
        ranked["r_theta"] = ranked["_theta_numeric"].rank(ascending=False, method="average")
        ranked["r_sigma"] = ranked["_sigma_numeric"].rank(ascending=False, method="average")
        ranked["endres_parameter_rank_sum"] = ranked["r_theta"] + ranked["r_sigma"]
        ranked = ranked.sort_values(
            [
                "endres_parameter_rank_sum",
                "_theta_numeric",
                "_sigma_numeric",
                ticker_a_col,
                ticker_b_col,
            ],
            ascending=[True, False, False, True, True],
            kind="mergesort",
        )
        ranks = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index, dtype=float)
        out.loc[ranked.index, "r_theta"] = ranked["r_theta"]
        out.loc[ranked.index, "r_sigma"] = ranked["r_sigma"]
        out.loc[ranked.index, "endres_parameter_rank_sum"] = ranked["endres_parameter_rank_sum"]
        out.loc[ranked.index, "endres_joint_model_rank"] = ranks
        out.loc[ranked.index, rank_col] = ranks

    if top_n is not None:
        selected = pd.to_numeric(out[rank_col], errors="coerce").le(int(top_n))
        out[selected_col] = selected.fillna(False)
    return out

def top_n_by_window(frame: pd.DataFrame, rank_col: str, top_n: int) -> pd.DataFrame:
    """Select rows with rank <= top_n inside each window."""

    out = frame.copy()
    ranks = pd.to_numeric(out[rank_col], errors="coerce")
    return out[ranks.le(int(top_n))].sort_values(["window_id", rank_col]).reset_index(drop=True)


def _truthy_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


__all__ = [
    "add_gaussian_endres_rank",
    "top_n_by_window",
]
