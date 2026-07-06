"""Pair-window ranking helpers."""

from __future__ import annotations

import pandas as pd


def top_n_by_window(frame: pd.DataFrame, rank_col: str, top_n: int) -> pd.DataFrame:
    """Keep rows with rank <= top_n, sorted by window and rank."""

    out = frame.copy()

    ranks = pd.to_numeric(out[rank_col], errors="coerce")
    is_top_n = ranks <= int(top_n)

    top_rows = out[is_top_n]
    top_rows = top_rows.sort_values(["window_id", rank_col])
    top_rows = top_rows.reset_index(drop=True)

    return top_rows

