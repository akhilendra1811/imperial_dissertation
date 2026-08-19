"""Asymmetric finite-variation CGMY-OU simulator using shifted-CDF FFT inversion.

The observed spread is represented as

    X_t = long_run_mean + Y_t,

where Y_t has a zero-mean stationary asymmetric CGMY(C, G, M, Y) law.
The deterministic CGMY location correction required to centre the asymmetric
jump law is computed internally.

The primary sampler follows the Baviera-Manzoni direct shifted-CDF structure.
A density-FFT implementation is retained as an explicit fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from scipy import special

ShiftChoice = Literal["auto", "positive", "negative"]
RightTailRate = Literal["model", "paper", "survival_match"]
DEFAULT_SIMULATOR_N_FFT = 2**15  # 32768 points for CGMY simulation inversion.


@dataclass(frozen=True)
class AsymmetricCGMYOUParams:
    lam: float
    long_run_mean: float
    C: float
    G: float
    M: float
    Y: float


@dataclass(frozen=True)
class _CDFCandidate:
    a: float
    h: float
    dx: float
    x_full: np.ndarray
    cdf_full: np.ndarray
    x: np.ndarray
    cdf: np.ndarray
    left_rate: float
    right_rate: float
    tail_score: float


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _safe_cf_exp(log_phi: np.ndarray | complex) -> np.ndarray | complex:
    """Exponentiate a log transform safely at real or complex arguments."""
    values = np.asarray(log_phi, dtype=np.complex128)
    real = np.clip(np.real(values), -745.0, 700.0)
    result = np.exp(real + 1j * np.imag(values))
    return result.item() if np.ndim(log_phi) == 0 else result


def cgmy_jump_mean(C: float, G: float, M: float, Y: float) -> float:
    """Mean of the zero-location asymmetric CGMY law."""
    return float(
        C
        * special.gamma(1.0 - Y)
        * (M ** (Y - 1.0) - G ** (Y - 1.0))
    )


def cgmy_zero_mean_location(C: float, G: float, M: float, Y: float) -> float:
    """Deterministic location that makes the stationary CGMY component mean zero."""
    return -cgmy_jump_mean(C=C, G=G, M=M, Y=Y)


def stationary_log_cf_centered(
    u: np.ndarray | complex,
    C: float,
    G: float,
    M: float,
    Y: float,
) -> np.ndarray | complex:
    """Log CF of the zero-mean stationary asymmetric CGMY component."""
    values = np.asarray(u, dtype=np.complex128)
    location = cgmy_zero_mean_location(C=C, G=G, M=M, Y=Y)
    jump = C * special.gamma(-Y) * (
        (M - 1j * values) ** Y
        - M**Y
        + (G + 1j * values) ** Y
        - G**Y
    )
    result = 1j * location * values + jump
    return result.item() if np.ndim(u) == 0 else result


def stationary_cf_centered(
    u: np.ndarray | complex,
    C: float,
    G: float,
    M: float,
    Y: float,
) -> np.ndarray | complex:
    return _safe_cf_exp(
        stationary_log_cf_centered(u=u, C=C, G=G, M=M, Y=Y)
    )


def innovation_cf_centered(
    u: np.ndarray | complex,
    C: float,
    G: float,
    M: float,
    Y: float,
    rho: float,
) -> np.ndarray | complex:
    """CF of eps in X_next = mean + rho*(X-mean) + eps."""
    values = np.asarray(u, dtype=np.complex128)
    log_phi = stationary_log_cf_centered(values, C, G, M, Y) - stationary_log_cf_centered(
        rho * values, C, G, M, Y
    )
    return _safe_cf_exp(log_phi)


def stationary_cumulants(
    C: float,
    G: float,
    M: float,
    Y: float,
) -> dict[str, float]:
    """Cumulants of the zero-mean stationary asymmetric CGMY component."""
    return {
        "c1": 0.0,
        "c2": float(
            C
            * special.gamma(2.0 - Y)
            * (M ** (Y - 2.0) + G ** (Y - 2.0))
        ),
        "c3": float(
            C
            * special.gamma(3.0 - Y)
            * (M ** (Y - 3.0) - G ** (Y - 3.0))
        ),
        "c4": float(
            C
            * special.gamma(4.0 - Y)
            * (M ** (Y - 4.0) + G ** (Y - 4.0))
        ),
    }


def innovation_cumulants(
    C: float,
    G: float,
    M: float,
    Y: float,
    rho: float,
) -> dict[str, float]:
    stat = stationary_cumulants(C=C, G=G, M=M, Y=Y)
    return {
        f"c{order}": float((1.0 - rho**order) * stat[f"c{order}"])
        for order in range(1, 5)
    }


def cumulant_interval(
    cumulants: dict[str, float],
    truncation_l: float,
) -> tuple[float, float, dict[str, float]]:
    """Asymmetric cumulant interval used by the density-FFT fallback."""
    c1 = float(cumulants.get("c1", np.nan))
    c2 = float(cumulants.get("c2", np.nan))
    c4 = float(cumulants.get("c4", np.nan))
    diag = {
        **cumulants,
        "truncation_a": np.nan,
        "truncation_b": np.nan,
        "truncation_width": np.nan,
        "truncation_l": float(truncation_l),
        "interval_source": "cgmy_cumulant",
    }
    width_scale = c2 + np.sqrt(max(c4, 0.0))
    if not (
        np.isfinite(c1)
        and np.isfinite(c2)
        and np.isfinite(c4)
        and width_scale > 0.0
    ):
        return np.nan, np.nan, diag
    half_width = float(truncation_l) * float(np.sqrt(width_scale))
    left = c1 - half_width
    right = c1 + half_width
    diag.update(
        {
            "truncation_a": float(left),
            "truncation_b": float(right),
            "truncation_width": float(right - left),
        }
    )
    return float(left), float(right), diag


class AsymmetricCGMYOUFGMC:
    """Asymmetric CGMY-OU simulator with direct shifted-CDF inversion."""

    def __init__(
        self,
        C: float,
        G: float,
        M: float,
        Y: float,
        long_run_mean: float,
        lam: float,
        dt: float,
        n_fft: int = DEFAULT_SIMULATOR_N_FFT,
        du: float | None = None,
        sampler: str = "shifted_cdf",
        seed: int = 123,
        *,
        h: float | None = None,
        a: float | None = None,
        shift: ShiftChoice = "auto",
        shift_fraction: float = 0.95,
        tail_probability_tolerance: float = 1e-4,
        right_tail_rate: RightTailRate = "model",
        build_stationary: bool = True,
    ) -> None:
        self.C = float(C)
        self.G = float(G)
        self.M = float(M)
        self.Y = float(Y)
        self.long_run_mean = float(long_run_mean)
        self.mu = self.long_run_mean  # compatibility alias
        self.lam = float(lam)
        self.dt = float(dt)
        self.n_fft = int(n_fft)
        self.sampler = str(sampler)
        self.shift = shift
        self.shift_fraction = float(shift_fraction)
        self.tail_probability_tolerance = float(tail_probability_tolerance)
        self.right_tail_rate = right_tail_rate
        self.rng = np.random.default_rng(seed)

        if not _is_power_of_two(self.n_fft):
            raise ValueError("n_fft must be a power of two.")
        if not (
            self.C > 0.0
            and self.G > 0.0
            and self.M > 0.0
            and 0.0 < self.Y < 1.0
        ):
            raise ValueError("Need C>0, G>0, M>0 and 0<Y<1.")
        if self.lam <= 0.0 or self.dt <= 0.0:
            raise ValueError("Need lambda>0 and dt>0.")
        if self.sampler not in {"shifted_cdf", "density_fft"}:
            raise ValueError("sampler must be 'shifted_cdf' or 'density_fft'.")
        if h is not None and du is not None:
            raise ValueError("Supply only one of h and du.")
        if h is None:
            h = du
        self.user_h = None if h is None else float(h)
        if self.user_h is not None and self.user_h <= 0.0:
            raise ValueError("h/du must be positive.")
        self.du = (
            20.0
            if self.sampler == "density_fft" and self.user_h is None
            else (
                float(self.user_h)
                if self.user_h is not None
                else np.nan
            )
        )
        self.user_a = None if a is None else float(a)

        if shift not in {"auto", "positive", "negative"}:
            raise ValueError("shift must be 'auto', 'positive', or 'negative'.")
        if not 0.0 < self.shift_fraction < 1.0:
            raise ValueError("shift_fraction must lie in (0,1).")
        if not 0.0 < self.tail_probability_tolerance < 0.5:
            raise ValueError("tail_probability_tolerance must lie in (0,0.5).")
        if right_tail_rate not in {"model", "paper", "survival_match"}:
            raise ValueError(
                "right_tail_rate must be 'model', 'paper', or 'survival_match'."
            )

        self.rho = float(np.exp(-self.lam * self.dt))
        self.c = self.rho
        self.cgmy_location = cgmy_zero_mean_location(
            C=self.C, G=self.G, M=self.M, Y=self.Y
        )

        if self.sampler == "shifted_cdf":
            self._build_innovation_shifted_cdf()
        else:
            self._build_innovation_density_fft()

        self.has_stationary_sampler = False
        if build_stationary:
            if self.sampler == "shifted_cdf":
                self._build_stationary_shifted_cdf()
            else:
                self._build_stationary_density_fft()

    def stationary_mean(self) -> float:
        return self.long_run_mean

    def centered_stationary_mean(self) -> float:
        return 0.0

    def _validate_shift(self, a: float) -> None:
        if a == 0.0:
            raise ValueError("The shifted-CDF complex shift a must be non-zero.")
        # Same conservative half-strip convention used by the NIG simulator.
        lower = -0.5 * self.G
        upper = 0.5 * self.M
        if not lower < a < upper:
            raise ValueError(
                f"Shift a must satisfy {lower} < a < {upper}. Received a={a}."
            )

    def _automatic_shifts(self) -> list[float]:
        positive = self.shift_fraction * 0.5 * self.M
        negative = -self.shift_fraction * 0.5 * self.G
        if self.shift == "positive":
            return [positive]
        if self.shift == "negative":
            return [negative]
        return [positive, negative]

    def recommended_h(self, a: float, cumulants: dict[str, float]) -> float:
        """Baviera-Manzoni model-implied Fourier spacing for CGMY-OU.

        For the bilateral CGMY stationary law used here, the innovation CF has
        high-frequency decay

            |phi_eps(u - i a)| ~ exp(-ell * |u|**omega),

        with

            omega = Y,
            ell = -2 C Gamma(-Y) cos(pi Y / 2) (1 - rho**Y),
            rho = exp(-lambda * dt).

        The paper's exponential-decay spacing rule is

            h(N) = max(
                (2 pi |a| / (ell N**omega))**(1 / (omega + 1)),
                0.01,
            ).

        ``cumulants`` is retained in the signature only for compatibility with
        the existing sampler-building interface; it is not used in this rule.
        """
        self._validate_shift(float(a))

        omega = float(self.Y)
        ell = float(
            -2.0
            * self.C
            * special.gamma(-omega)
            * np.cos(0.5 * np.pi * omega)
            * (1.0 - self.rho**omega)
        )
        if not np.isfinite(ell) or ell <= 0.0:
            raise RuntimeError(
                "The CGMY innovation CF decay coefficient is not positive."
            )

        h = (
            2.0 * np.pi * abs(float(a))
            / (ell * self.n_fft**omega)
        ) ** (1.0 / (omega + 1.0))

        return float(max(h, 0.01))

    @staticmethod
    def _largest_valid_monotone_block(cdf: np.ndarray) -> tuple[int, int]:
        point_ok = np.isfinite(cdf) & (cdf >= 0.0) & (cdf <= 1.0)
        best_start = best_stop = 0
        current_start: int | None = None
        for index in range(cdf.size):
            if not point_ok[index]:
                if (
                    current_start is not None
                    and index - current_start > best_stop - best_start
                ):
                    best_start, best_stop = current_start, index
                current_start = None
                continue
            if current_start is None:
                current_start = index
                continue
            if cdf[index] <= cdf[index - 1]:
                if index - current_start > best_stop - best_start:
                    best_start, best_stop = current_start, index
                current_start = index
        if (
            current_start is not None
            and cdf.size - current_start > best_stop - best_start
        ):
            best_start, best_stop = current_start, cdf.size
        return best_start, best_stop

    def _fft_cdf_candidate(
        self,
        cf_func: Callable[[np.ndarray], np.ndarray],
        cumulants: dict[str, float],
        a: float,
        location_shift: float,
        h_override: float | None = None,
    ) -> _CDFCandidate:
        self._validate_shift(float(a))
        h = (
            float(h_override)
            if h_override is not None
            else (
                self.user_h
                if self.user_h is not None
                else self.recommended_h(float(a), cumulants)
            )
        )

        n = np.arange(self.n_fft, dtype=float)
        frequencies = (n + 0.5) * h
        shifted_frequencies = frequencies - 1j * a
        coefficients = cf_func(shifted_frequencies) / (1j * frequencies + a)
        if np.any(~np.isfinite(coefficients)):
            raise RuntimeError("Non-finite shifted-CDF Fourier coefficients.")

        dx = 2.0 * np.pi / (self.n_fft * h)
        x_centered = (
            np.arange(self.n_fft, dtype=float) - self.n_fft / 2.0
        ) * dx
        alternating = np.where(
            np.arange(self.n_fft) % 2 == 0, 1.0, -1.0
        )
        fft_values = np.fft.fft(coefficients * alternating)
        cdf_raw = (1.0 if a > 0.0 else 0.0) - (
            h / np.pi
        ) * np.exp(np.clip(-a * x_centered, -700.0, 700.0)) * np.real(
            np.exp(-0.5j * x_centered * h) * fft_values
        )

        start, stop = self._largest_valid_monotone_block(cdf_raw)
        if stop - start < 50:
            raise RuntimeError(
                f"Too few valid monotone CDF points for a={a}, h={h}."
            )

        x_valid = x_centered[start:stop]
        cdf_valid = cdf_raw[start:stop]
        interior = np.flatnonzero(
            (cdf_valid > 1e-15) & (cdf_valid < 1.0 - 1e-15)
        )
        if interior.size < 50:
            raise RuntimeError("Too few interior CDF points.")
        x_valid = x_valid[int(interior[0]) : int(interior[-1]) + 1]
        cdf_valid = cdf_valid[int(interior[0]) : int(interior[-1]) + 1]

        if self.right_tail_rate == "model":
            left_rate = self.G
            right_rate = self.M
        else:
            left_rate = (
                np.log(cdf_valid[1]) - np.log(cdf_valid[0])
            ) / (x_valid[1] - x_valid[0])
            if self.right_tail_rate == "paper":
                right_rate = (
                    np.log(cdf_valid[-1]) - np.log(cdf_valid[-2])
                ) / (x_valid[-1] - x_valid[-2])
            else:
                right_rate = (
                    np.log1p(-cdf_valid[-2])
                    - np.log1p(-cdf_valid[-1])
                ) / (x_valid[-1] - x_valid[-2])

        if not np.isfinite(left_rate) or left_rate <= 0.0:
            raise RuntimeError("Invalid left-tail rate.")
        if not np.isfinite(right_rate) or right_rate <= 0.0:
            raise RuntimeError("Invalid right-tail rate.")

        return _CDFCandidate(
            a=float(a),
            h=float(h),
            dx=float(dx),
            x_full=x_centered + float(location_shift),
            cdf_full=cdf_raw,
            x=x_valid + float(location_shift),
            cdf=cdf_valid,
            left_rate=float(left_rate),
            right_rate=float(right_rate),
            tail_score=float(
                max(cdf_valid[0], 1.0 - cdf_valid[-1])
            ),
        )

    def _build_shifted_cdf_sampler(
        self,
        cf_func: Callable[[np.ndarray], np.ndarray],
        cumulants: dict[str, float],
        location_shift: float,
        target: str,
    ) -> dict[str, Any]:
        shifts = (
            [self.user_a]
            if self.user_a is not None
            else self._automatic_shifts()
        )
        candidates: list[_CDFCandidate] = []
        failures: list[str] = []

        for shift_value in shifts:
            assert shift_value is not None
            # Use exactly one Fourier spacing for each admissible shift: the
            # Baviera-Manzoni model-implied h(N). No local h multipliers are
            # tested.
            trial_h = self.recommended_h(float(shift_value), cumulants)
            try:
                candidates.append(
                    self._fft_cdf_candidate(
                        cf_func=cf_func,
                        cumulants=cumulants,
                        a=float(shift_value),
                        location_shift=location_shift,
                        h_override=trial_h,
                    )
                )
            except (
                RuntimeError,
                ValueError,
                FloatingPointError,
            ) as exc:
                failures.append(
                    f"a={shift_value}, h={trial_h}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not candidates:
            raise RuntimeError(
                f"{target} shifted-CDF construction failed. "
                + "; ".join(failures)
            )

        candidate = min(candidates, key=lambda item: item.tail_score)
        diagnostics = {
            "sampler": "shifted_cdf",
            "method": "Baviera-Manzoni FGMC direct shifted-CDF inversion",
            "target": target,
            "inverse_sampler": "linear_np_interp_with_exponential_tails",
            "n_fft": self.n_fft,
            "selected_a": candidate.a,
            "selected_h": candidate.h,
            "du": candidate.h,
            "dx": candidate.dx,
            "tail_probability_tolerance": self.tail_probability_tolerance,
            "tail_score": candidate.tail_score,
            "tail_validation_passed": (
                candidate.tail_score <= self.tail_probability_tolerance
            ),
            "left_tail_rate": candidate.left_rate,
            "right_tail_rate": candidate.right_rate,
            "x_min_full": float(candidate.x_full[0]),
            "x_max_full": float(candidate.x_full[-1]),
            "x_min_used": float(candidate.x[0]),
            "x_max_used": float(candidate.x[-1]),
            "cdf_left_used": float(candidate.cdf[0]),
            "cdf_right_used": float(candidate.cdf[-1]),
            "num_inverse_points": int(len(candidate.cdf)),
            "candidate_failures": failures,
            "C": self.C,
            "G": self.G,
            "M": self.M,
            "Y": self.Y,
            "cgmy_location": self.cgmy_location,
            "long_run_mean": self.long_run_mean,
            **cumulants,
        }
        if candidate.tail_score > self.tail_probability_tolerance:
            raise RuntimeError(
                f"{target} shifted-CDF failed tail-domain check: "
                f"{candidate.tail_score:.3e} > "
                f"{self.tail_probability_tolerance:.3e}."
            )
        return {
            "x_full": candidate.x_full,
            "cdf_full": candidate.cdf_full,
            "x": candidate.x,
            "cdf": candidate.cdf,
            "left_rate": candidate.left_rate,
            "right_rate": candidate.right_rate,
            "diagnostics": diagnostics,
        }

    def _build_innovation_shifted_cdf(self) -> None:
        sampler = self._build_shifted_cdf_sampler(
            cf_func=lambda u: innovation_cf_centered(
                u,
                C=self.C,
                G=self.G,
                M=self.M,
                Y=self.Y,
                rho=self.rho,
            ),
            cumulants=innovation_cumulants(
                self.C, self.G, self.M, self.Y, self.rho
            ),
            location_shift=0.0,
            target="innovation",
        )
        self._store_innovation_sampler(sampler)

    def _build_stationary_shifted_cdf(self) -> None:
        sampler = self._build_shifted_cdf_sampler(
            cf_func=lambda u: stationary_cf_centered(
                u, self.C, self.G, self.M, self.Y
            ),
            cumulants=stationary_cumulants(
                self.C, self.G, self.M, self.Y
            ),
            location_shift=self.long_run_mean,
            target="stationary",
        )
        self._store_stationary_sampler(sampler)

    def _density_sampler(
        self,
        cf_func: Callable[[np.ndarray], np.ndarray],
        location_shift: float,
        target: str,
    ) -> dict[str, Any]:
        n_fft = self.n_fft
        du = float(self.du)
        k = np.arange(n_fft)
        u = (k - n_fft // 2) * du
        dx = 2.0 * np.pi / (n_fft * du)
        x_centered = (np.arange(n_fft) - n_fft // 2) * dx
        x = x_centered + location_shift

        phi = cf_func(u)
        fft_values = np.fft.fft(phi * ((-1.0) ** k))
        phase = (-1.0) ** (np.arange(n_fft) + n_fft // 2)
        density_raw = np.real(
            (du / (2.0 * np.pi)) * phase * fft_values
        )
        raw_area = float(np.trapezoid(density_raw, x))
        negative_mass = float(
            np.trapezoid(np.clip(-density_raw, 0.0, None), x)
        )
        density = np.clip(density_raw, 0.0, None)
        clipped_area = float(np.trapezoid(density, x))
        if not np.isfinite(clipped_area) or clipped_area <= 0.0:
            raise RuntimeError("Density FFT has non-positive clipped mass.")
        density /= clipped_area

        increments = (
            0.5 * (density[:-1] + density[1:]) * np.diff(x)
        )
        cdf = np.r_[0.0, np.cumsum(increments)]
        cdf = np.maximum.accumulate(
            np.clip(cdf / cdf[-1], 0.0, 1.0)
        )
        keep = np.r_[True, np.diff(cdf) > 1e-12]
        return {
            "x_full": x,
            "cdf_full": cdf,
            "x": x[keep],
            "cdf": cdf[keep],
            "left_rate": self.G,
            "right_rate": self.M,
            "diagnostics": {
                "sampler": "density_fft",
                "target": target,
                "n_fft": n_fft,
                "du": du,
                "dx": dx,
                "raw_density_area": raw_area,
                "negative_density_mass": negative_mass,
                "clipped_density_area": clipped_area,
                "num_inverse_points": int(np.sum(keep)),
            },
        }

    def _build_innovation_density_fft(self) -> None:
        sampler = self._density_sampler(
            cf_func=lambda u: innovation_cf_centered(
                u,
                self.C,
                self.G,
                self.M,
                self.Y,
                self.rho,
            ),
            location_shift=0.0,
            target="innovation",
        )
        self._store_innovation_sampler(sampler)

    def _build_stationary_density_fft(self) -> None:
        sampler = self._density_sampler(
            cf_func=lambda u: stationary_cf_centered(
                u, self.C, self.G, self.M, self.Y
            ),
            location_shift=self.long_run_mean,
            target="stationary",
        )
        self._store_stationary_sampler(sampler)

    def _store_innovation_sampler(self, sampler: dict[str, Any]) -> None:
        self.x_grid = sampler["x_full"]
        self.cdf_raw = sampler["cdf_full"]
        self.x_cdf = sampler["x"]
        self.cdf = sampler["cdf"]
        self.left_tail_rate = float(sampler["left_rate"])
        self.right_tail_rate_value = float(sampler["right_rate"])
        self.innovation_fft_diagnostics = sampler["diagnostics"]
        self.fft_diagnostics = self.innovation_fft_diagnostics

    def _store_stationary_sampler(self, sampler: dict[str, Any]) -> None:
        self.x_stationary_grid = sampler["x_full"]
        self.stationary_cdf_raw = sampler["cdf_full"]
        self.x_stationary_cdf = sampler["x"]
        self.stationary_cdf = sampler["cdf"]
        self.stationary_left_tail_rate = float(
            sampler["left_rate"]
        )
        self.stationary_right_tail_rate = float(
            sampler["right_rate"]
        )
        self.stationary_fft_diagnostics = sampler["diagnostics"]
        self.has_stationary_sampler = True

    @staticmethod
    def _cdf_from_grid(
        values: np.ndarray | float,
        x_grid: np.ndarray,
        cdf_grid: np.ndarray,
        left_rate: float,
        right_rate: float,
    ) -> np.ndarray | float:
        arr = np.asarray(values, dtype=float)
        scalar = arr.ndim == 0
        arr1 = np.atleast_1d(arr)
        out = np.empty_like(arr1)
        left = arr1 < x_grid[0]
        right = arr1 > x_grid[-1]
        middle = ~(left | right)
        out[left] = cdf_grid[0] * np.exp(
            left_rate * (arr1[left] - x_grid[0])
        )
        out[right] = 1.0 - (1.0 - cdf_grid[-1]) * np.exp(
            -right_rate * (arr1[right] - x_grid[-1])
        )
        out[middle] = np.interp(
            arr1[middle], x_grid, cdf_grid
        )
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if scalar else out

    @staticmethod
    def _quantile_from_grid(
        probs: np.ndarray | float,
        x_grid: np.ndarray,
        cdf_grid: np.ndarray,
        left_rate: float,
        right_rate: float,
    ) -> np.ndarray | float:
        arr = np.asarray(probs, dtype=float)
        scalar = arr.ndim == 0
        arr1 = np.atleast_1d(arr)
        if np.any((arr1 <= 0.0) | (arr1 >= 1.0)):
            raise ValueError("Probabilities must lie strictly in (0,1).")
        out = np.empty_like(arr1)
        left = arr1 < cdf_grid[0]
        right = arr1 > cdf_grid[-1]
        middle = ~(left | right)
        out[left] = x_grid[0] + np.log(
            arr1[left] / cdf_grid[0]
        ) / left_rate
        out[right] = x_grid[-1] - np.log(
            (1.0 - arr1[right]) / (1.0 - cdf_grid[-1])
        ) / right_rate
        out[middle] = np.interp(
            arr1[middle], cdf_grid, x_grid
        )
        if np.any(~np.isfinite(out)):
            raise RuntimeError("Inverse CDF returned NaN or inf.")
        return float(out[0]) if scalar else out

    def cdf_linear(
        self, values: np.ndarray | float
    ) -> np.ndarray | float:
        return self._cdf_from_grid(
            values,
            self.x_cdf,
            self.cdf,
            self.left_tail_rate,
            self.right_tail_rate_value,
        )

    def quantile_linear(
        self, probs: np.ndarray | float
    ) -> np.ndarray | float:
        return self._quantile_from_grid(
            probs,
            self.x_cdf,
            self.cdf,
            self.left_tail_rate,
            self.right_tail_rate_value,
        )

    def sample_innovations(self, n: int) -> np.ndarray:
        return np.asarray(
            self.quantile_linear(self.rng.random(int(n))),
            dtype=float,
        )

    def sample_stationary(self, n: int = 1) -> np.ndarray:
        if not self.has_stationary_sampler:
            raise RuntimeError("Stationary sampler was not built.")
        return np.asarray(
            self._quantile_from_grid(
                self.rng.random(int(n)),
                self.x_stationary_cdf,
                self.stationary_cdf,
                self.stationary_left_tail_rate,
                self.stationary_right_tail_rate,
            ),
            dtype=float,
        )

    def validate_innovations(
        self, n: int = 20_000
    ) -> dict[str, float]:
        sample = self.sample_innovations(int(n))
        centered = sample - float(np.mean(sample))
        sample_k2 = float(np.mean(centered**2))
        theory = innovation_cumulants(
            self.C, self.G, self.M, self.Y, self.rho
        )
        return {
            "sample_kappa_1": float(np.mean(sample)),
            "sample_kappa_2": sample_k2,
            "sample_kappa_3": float(np.mean(centered**3)),
            "sample_kappa_4": float(
                np.mean(centered**4) - 3.0 * sample_k2**2
            ),
            "theoretical_kappa_1": theory["c1"],
            "theoretical_kappa_2": theory["c2"],
            "theoretical_kappa_3": theory["c3"],
            "theoretical_kappa_4": theory["c4"],
        }

    def simulate(
        self,
        n: int,
        stationary_start: bool = True,
        burn_in: int = 0,
        x0: float | None = None,
    ) -> np.ndarray:
        total = int(n) + int(burn_in)
        if total <= 1:
            raise ValueError("Need n + burn_in > 1.")
        if burn_in < 0:
            raise ValueError("Need burn_in >= 0.")
        path = np.empty(total, dtype=float)
        if x0 is not None:
            path[0] = float(x0)
        elif stationary_start:
            path[0] = self.sample_stationary(1)[0]
        else:
            path[0] = self.long_run_mean
        eps = self.sample_innovations(total - 1)
        for t in range(1, total):
            path[t] = (
                self.long_run_mean
                + self.rho * (path[t - 1] - self.long_run_mean)
                + eps[t - 1]
            )
        return path[burn_in:] if burn_in else path

    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        x0: float | np.ndarray | None = None,
        stationary_start: bool = True,
        burn_in: int = 0,
    ) -> np.ndarray:
        n_paths = int(n_paths)
        n_steps = int(n_steps)
        burn_in = int(burn_in)
        if n_paths <= 0:
            raise ValueError("Need n_paths > 0.")
        if n_steps <= 0:
            raise ValueError("Need n_steps > 0.")
        if burn_in < 0:
            raise ValueError("Need burn_in >= 0.")

        total = n_steps + 1 + burn_in
        paths = np.empty((n_paths, total), dtype=float)
        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=float)
            if x0_arr.ndim == 0:
                paths[:, 0] = float(x0_arr)
            elif x0_arr.shape == (n_paths,):
                paths[:, 0] = x0_arr
            else:
                raise ValueError(
                    "x0 must be scalar or shape (n_paths,)."
                )
        elif stationary_start:
            paths[:, 0] = self.sample_stationary(n_paths)
        else:
            paths[:, 0] = self.long_run_mean

        eps = self.sample_innovations(
            n_paths * (total - 1)
        ).reshape(n_paths, total - 1)
        for t in range(1, total):
            paths[:, t] = (
                self.long_run_mean
                + self.rho
                * (paths[:, t - 1] - self.long_run_mean)
                + eps[:, t - 1]
            )
        return paths[:, burn_in:] if burn_in else paths


def build_simulator_from_estimate(
    est: dict[str, Any],
    dt: float,
    n_fft: int = DEFAULT_SIMULATOR_N_FFT,
    du: float | None = None,
    seed: int = 123,
    sampler: str = "shifted_cdf",
    build_stationary: bool = False,
) -> AsymmetricCGMYOUFGMC:
    """Build directly from the asymmetric Valdivieso estimator output.

    Stationary sampling is disabled by default because most project workflows
    simulate transitions from an observed spread level.
    """
    mean = est.get("gaussian_mean", est.get("long_run_mean", est.get("mu")))
    if mean is None:
        raise KeyError(
            "Estimate must contain gaussian_mean, long_run_mean, or mu."
        )
    lam = est.get("lambda_ou", est.get("lambda"))
    if lam is None:
        raise KeyError("Estimate must contain lambda_ou or lambda.")
    return AsymmetricCGMYOUFGMC(
        C=float(est["C"]),
        G=float(est["G"]),
        M=float(est["M"]),
        Y=float(est["Y"]),
        long_run_mean=float(mean),
        lam=float(lam),
        dt=float(dt),
        n_fft=int(n_fft),
        du=du,
        sampler=sampler,
        seed=int(seed),
        build_stationary=build_stationary,
    )


def build_cgmy_ou_simulator_with_fallback(
    C: float,
    G: float,
    M: float,
    Y: float,
    long_run_mean: float,
    lam: float,
    dt: float,
    seed: int = 123,
    shifted_attempts: tuple[dict[str, Any], ...] = (
        {"n_fft": DEFAULT_SIMULATOR_N_FFT, "shift_fraction": 0.95},
        {"n_fft": 2**16, "shift_fraction": 0.90},
    ),
    density_n_fft: int = DEFAULT_SIMULATOR_N_FFT,
    density_du: float = 20.0,
    density_negative_mass_tol: float = 1e-4,
    density_raw_area_tol: float = 1e-3,
    build_stationary: bool = False,
) -> tuple[AsymmetricCGMYOUFGMC, dict[str, Any]]:
    """Build a CGMY-OU transition simulator, falling back across FFT schemes.

    Backtests simulate forward from the observed spread level, so they only
    need the transition innovation sampler.  Stationary sampling is optional
    because validating a stationary CDF can reject parameter sets whose
    innovation sampler is numerically sound.
    """
    failures: list[str] = []
    for index, attempt in enumerate(shifted_attempts, start=1):
        try:
            sim = AsymmetricCGMYOUFGMC(
                C=C,
                G=G,
                M=M,
                Y=Y,
                long_run_mean=long_run_mean,
                lam=lam,
                dt=dt,
                seed=seed,
                sampler="shifted_cdf",
                n_fft=int(attempt.get("n_fft", DEFAULT_SIMULATOR_N_FFT)),
                shift_fraction=float(
                    attempt.get("shift_fraction", 0.95)
                ),
                build_stationary=build_stationary,
            )
            return sim, {
                "simulation_method": "shifted_cdf",
                "simulation_attempt": index,
                "simulation_validation_passed": True,
                "fallback_used": False,
                "attempt_failures": failures,
                **{
                    f"sim_{k}": v
                    for k, v in sim.fft_diagnostics.items()
                },
            }
        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
        ) as exc:
            failures.append(
                f"shifted_attempt_{index}: "
                f"{type(exc).__name__}: {exc}"
            )

    sim = AsymmetricCGMYOUFGMC(
        C=C,
        G=G,
        M=M,
        Y=Y,
        long_run_mean=long_run_mean,
        lam=lam,
        dt=dt,
        seed=seed,
        sampler="density_fft",
        n_fft=int(density_n_fft),
        du=float(density_du),
        build_stationary=build_stationary,
    )
    raw_area = float(
        sim.fft_diagnostics.get("raw_density_area", np.nan)
    )
    negative_mass = float(
        sim.fft_diagnostics.get("negative_density_mass", np.nan)
    )
    accepted = bool(
        np.isfinite(raw_area)
        and np.isfinite(negative_mass)
        and abs(raw_area - 1.0) < density_raw_area_tol
        and negative_mass < density_negative_mass_tol
    )
    diag = {
        "simulation_method": "density_fft_fallback",
        "simulation_attempt": len(shifted_attempts) + 1,
        "simulation_validation_passed": accepted,
        "fallback_used": True,
        "attempt_failures": failures,
        **{
            f"sim_{k}": v
            for k, v in sim.fft_diagnostics.items()
        },
    }
    if not accepted:
        raise RuntimeError(
            f"Density-FFT fallback failed diagnostics: {diag}"
        )
    return sim, diag


# Convenient short alias.
CGMYOUFGMC = AsymmetricCGMYOUFGMC


__all__ = [
    "AsymmetricCGMYOUFGMC",
    "AsymmetricCGMYOUParams",
    "CGMYOUFGMC",
    "build_cgmy_ou_simulator_with_fallback",
    "build_simulator_from_estimate",
    "cgmy_jump_mean",
    "cgmy_zero_mean_location",
    "cumulant_interval",
    "innovation_cf_centered",
    "innovation_cumulants",
    "stationary_cf_centered",
    "stationary_cumulants",
    "stationary_log_cf_centered",
]
