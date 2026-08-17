"""Plain formation mean/std thresholds for baseline pair trading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasicBaselineFit:
    """Fixed formation statistics used by the basic baseline strategy."""

    mean: float
    std: float
    observations: int
    ddof: int = 1

    def bands(self, sigma_multiple: float) -> tuple[float, float]:
        """Return the absolute lower and upper entry levels."""
        multiple = float(sigma_multiple)
        if not np.isfinite(multiple) or multiple <= 0.0:
            raise ValueError("sigma_multiple must be finite and positive")
        distance = multiple * self.std
        return self.mean - distance, self.mean + distance


def fit_basic_baseline(formation_spread: np.ndarray, *, ddof: int = 1) -> BasicBaselineFit:
    """Estimate an arithmetic mean and sample standard deviation once in formation."""
    values = np.asarray(formation_spread, dtype=float).reshape(-1)
    if values.size <= ddof:
        raise ValueError(f"need more than {ddof} formation observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("formation_spread must contain only finite values")
    std = float(np.std(values, ddof=ddof))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("formation spread standard deviation must be positive")
    return BasicBaselineFit(mean=float(np.mean(values)), std=std, observations=int(values.size), ddof=int(ddof))


__all__ = ["BasicBaselineFit", "fit_basic_baseline"]
