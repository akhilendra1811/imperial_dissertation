"""FFT density helpers used by Levy-OU likelihoods."""

from __future__ import annotations

from typing import Any

import numpy as np


def fft_density_from_cf_real_line(
    values: np.ndarray,
    cf_values: Any,
    left: float,
    right: float,
    fft_grid_size: int,
    density_floor: float,
    interval_diagnostics: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """FFT density inversion on a real-line interval, then linear interpolation."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n_fft = int(fft_grid_size)
    if n_fft < 128:
        raise ValueError("fft_grid_size must be at least 128.")
    if n_fft % 2:
        n_fft += 1

    density = np.full(len(x), density_floor, dtype=float)
    diagnostics: dict[str, float] = {
        **interval_diagnostics,
        "density_count": int(len(x)),
        "fft_grid_size": int(n_fft),
        "outside_interval_count": np.nan,
        "outside_interval_fraction": np.nan,
        "negative_grid_density_count": np.nan,
        "negative_grid_density_fraction": np.nan,
        "negative_density_count": np.nan,
        "negative_density_fraction": np.nan,
        "floored_density_count": np.nan,
        "floored_density_fraction": np.nan,
        "raw_grid_density_min": np.nan,
        "raw_grid_density_max": np.nan,
        "raw_density_min": np.nan,
        "raw_density_max": np.nan,
        "raw_grid_density_area": np.nan,
        "fft_dx": np.nan,
        "fft_du": np.nan,
    }
    if len(x) == 0:
        return density, diagnostics
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        diagnostics.update(
            {
                "outside_interval_count": int(len(x)),
                "outside_interval_fraction": 1.0,
                "negative_grid_density_count": int(n_fft),
                "negative_grid_density_fraction": 1.0,
                "negative_density_count": int(len(x)),
                "negative_density_fraction": 1.0,
                "floored_density_count": int(len(x)),
                "floored_density_fraction": 1.0,
            }
        )
        return density, diagnostics

    dx = float((right - left) / n_fft)
    du = float(2.0 * np.pi / (n_fft * dx))
    x_grid = left + dx * np.arange(n_fft, dtype=float)
    k = np.arange(n_fft, dtype=float)
    u = (k - n_fft / 2.0) * du
    phi = cf_values(u)
    shifted_phi = phi * np.exp(-1j * u * left)
    raw_grid_density = np.real(
        (du / (2.0 * np.pi))
        * np.exp(1j * np.pi * np.arange(n_fft, dtype=float))
        * np.fft.fft(shifted_phi)
    )
    raw_density = np.interp(x, x_grid, raw_grid_density, left=np.nan, right=np.nan)
    outside = ~np.isfinite(raw_density)
    negative_grid = raw_grid_density < 0
    negative_observed = raw_density < 0
    floored = outside | (raw_density <= density_floor)
    density = np.where(floored, density_floor, np.maximum(raw_density, density_floor))
    finite_raw_density = raw_density[np.isfinite(raw_density)]
    diagnostics.update(
        {
            "outside_interval_count": int(outside.sum()),
            "outside_interval_fraction": float(outside.mean()),
            "negative_grid_density_count": int(negative_grid.sum()),
            "negative_grid_density_fraction": float(negative_grid.mean()),
            "negative_density_count": int(negative_observed.sum()),
            "negative_density_fraction": float(negative_observed.mean()),
            "floored_density_count": int(floored.sum()),
            "floored_density_fraction": float(floored.mean()),
            "raw_grid_density_min": float(np.nanmin(raw_grid_density)),
            "raw_grid_density_max": float(np.nanmax(raw_grid_density)),
            "raw_density_min": (
                float(np.nanmin(finite_raw_density))
                if len(finite_raw_density)
                else np.nan
            ),
            "raw_density_max": (
                float(np.nanmax(finite_raw_density))
                if len(finite_raw_density)
                else np.nan
            ),
            "raw_grid_density_area": float(np.trapezoid(raw_grid_density, x_grid)),
            "fft_dx": dx,
            "fft_du": du,
        }
    )
    return density, diagnostics


_fft_density_from_cf_real_line = fft_density_from_cf_real_line

__all__ = [
    "fft_density_from_cf_real_line",
    "_fft_density_from_cf_real_line",
]
