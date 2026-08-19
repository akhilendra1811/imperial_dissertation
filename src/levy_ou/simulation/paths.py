"""Generic path diagnostics."""

from __future__ import annotations

import numpy as np


def path_summary(paths: np.ndarray) -> dict[str, float]:
    """Return simple diagnostics for simulated paths."""

    arr = np.asarray(paths, dtype=float)
    return {
        "paths": float(arr.shape[0]) if arr.ndim >= 2 else 1.0,
        "steps": float(arr.shape[-1]) if arr.ndim else 0.0,
        "finite_fraction": float(np.isfinite(arr).mean()) if arr.size else 0.0,
        "mean": float(np.nanmean(arr)) if arr.size else np.nan,
        "std": float(np.nanstd(arr)) if arr.size else np.nan,
    }

