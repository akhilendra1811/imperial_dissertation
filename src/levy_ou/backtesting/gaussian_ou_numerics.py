"""Small numerical helpers for Gaussian OU threshold rules."""

from __future__ import annotations

import math
from typing import Callable

from scipy.optimize import minimize_scalar
from scipy.special import erfi as scipy_erfi


def erfi(x: float) -> float:
    """Return the imaginary error function as a Python float."""

    return float(scipy_erfi(float(x)))


def maximize_bounded(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    xatol: float = 1e-12,
    maxiter: int = 1000,
) -> dict[str, float | int | bool | str]:
    """Maximise a scalar objective on a bounded interval using SciPy."""

    a = float(lower)
    b = float(upper)
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        return {
            "success": False,
            "message": "invalid bounds",
            "x": math.nan,
            "fun": math.nan,
            "nfev": 0,
            "nit": 0,
        }

    result = minimize_scalar(
        lambda x: -float(objective(float(x))),
        bounds=(a, b),
        method="bounded",
        options={"xatol": float(xatol), "maxiter": int(maxiter)},
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "x": float(result.x),
        "fun": float(-result.fun),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
    }


__all__ = ["erfi", "maximize_bounded"]
