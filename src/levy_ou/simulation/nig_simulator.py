"""Stationary NIG-OU FFT simulators with explicit inversion diagnostics.

The primary simulator is the shifted-CDF FGMC inversion of the NIG-OU
innovation characteristic function. It uses monotone linear CDF and inverse-CDF
interpolation to avoid cubic-spline overshoot. A density-FFT sampler remains
available as an explicit fallback and diagnostic benchmark.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

ShiftChoice = Literal["auto", "positive", "negative"]
RightTailRate = Literal["paper", "survival_match"]
DEFAULT_SIMULATOR_N_FFT = 2**15  # 32768 points for NIG simulation inversion.


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


class NIGOUFGMC:
    """Stationary NIG-OU simulator using shifted-CDF or density-FFT inversion."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        mu: float,
        delta: float,
        lam: float,
        dt: float,
        n_fft: int = DEFAULT_SIMULATOR_N_FFT,
        du: float | None = None,
        eta: float | None = None,
        sampler: str = "shifted_cdf",
        seed: int = 123,
        *,
        h: float | None = None,
        a: float | None = None,
        shift: ShiftChoice = "auto",
        shift_fraction: float = 0.95,
        tail_probability_tolerance: float = 1e-4,
        right_tail_rate: RightTailRate = "survival_match",
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mu = float(mu)
        self.delta = float(delta)
        self.lam = float(lam)
        self.dt = float(dt)
        self.n_fft = int(n_fft)
        self.sampler = sampler
        self.shift = shift
        self.shift_fraction = float(shift_fraction)
        self.tail_probability_tolerance = float(tail_probability_tolerance)
        self.right_tail_rate = right_tail_rate

        if self.n_fft & (self.n_fft - 1) != 0:
            raise ValueError("n_fft must be a power of 2, for example 2**15.")
        if not (self.alpha > abs(self.beta)):
            raise ValueError("NIG condition failed: need alpha > abs(beta).")
        if self.delta <= 0:
            raise ValueError("Need delta > 0.")
        if self.lam <= 0:
            raise ValueError("Need lambda > 0.")
        if self.dt <= 0:
            raise ValueError("Need dt > 0.")
        if self.sampler not in {"shifted_cdf", "density_fft"}:
            raise ValueError("sampler must be 'shifted_cdf' or 'density_fft'.")
        if h is not None and du is not None:
            raise ValueError("Supply only one of h and du; they are aliases for shifted-CDF spacing.")
        if h is None:
            h = du
        self.user_h = None if h is None else float(h)
        if self.user_h is not None and self.user_h <= 0:
            raise ValueError("h/du must be positive.")
        if self.sampler == "density_fft":
            self.du = 20.0 if du is None and h is None else float(self.user_h)
        else:
            self.du = float(self.user_h) if self.user_h is not None else np.nan

        upper_strip = self.alpha + self.beta
        if a is not None and eta is not None:
            raise ValueError("Supply only one of a and eta.")
        if a is None and eta is not None and sampler == "shifted_cdf":
            a = -float(eta)
        self.user_a = None if a is None else float(a)

        if eta is None:
            eta = min(0.1, 0.5 * upper_strip)
        self.eta = float(eta)
        if self.sampler == "density_fft" and not (0 < self.eta < upper_strip):
            raise ValueError(
                f"Need 0 < eta < alpha + beta = {upper_strip}. "
                f"Current eta is {self.eta}."
            )
        if shift not in {"auto", "positive", "negative"}:
            raise ValueError("shift must be 'auto', 'positive', or 'negative'.")
        if not (0.0 < self.shift_fraction < 1.0):
            raise ValueError("shift_fraction must lie strictly between 0 and 1.")
        if not (0.0 < self.tail_probability_tolerance < 0.5):
            raise ValueError("tail_probability_tolerance must lie in (0, 0.5).")
        if right_tail_rate not in {"paper", "survival_match"}:
            raise ValueError("right_tail_rate must be 'paper' or 'survival_match'.")

        self.c = float(np.exp(-self.lam * self.dt))
        self.rng = np.random.default_rng(seed)
        if self.sampler == "shifted_cdf":
            self._build_shifted_cdf_sampler()
        elif self.sampler == "density_fft":
            self._build_density_sampler()
        else:
            raise ValueError("sampler must be 'density_fft' or 'shifted_cdf'.")

    def stationary_mean(self) -> float:
        return self.mu + self.delta * self.beta / np.sqrt(self.alpha**2 - self.beta**2)

    def nig_log_cf(self, u: np.ndarray | complex) -> np.ndarray | complex:
        """Log characteristic function of stationary NIG(alpha,beta,mu,delta)."""
        return (
            1j * self.mu * u
            + self.delta
            * (
                np.sqrt(self.alpha * self.alpha - self.beta * self.beta)
                - np.sqrt(self.alpha * self.alpha - (self.beta + 1j * u) ** 2)
            )
        )

    def innovation_cf(self, u: np.ndarray | complex) -> np.ndarray | complex:
        """Characteristic function of eps_t in X_{t+dt} = c X_t + eps_t."""
        return np.exp(self.nig_log_cf(u) - self.nig_log_cf(self.c * u))

    @property
    def p_minus(self) -> float:
        return self.alpha + self.beta

    @property
    def p_plus(self) -> float:
        return self.alpha - self.beta

    def stationary_variance(self) -> float:
        gamma = np.sqrt(self.alpha**2 - self.beta**2)
        return self.delta * self.alpha**2 / gamma**3

    def stationary_cumulants(self) -> dict[int, float]:
        gamma = np.sqrt(self.alpha**2 - self.beta**2)
        return {
            1: self.mu + self.delta * self.beta / gamma,
            2: self.delta * self.alpha**2 / gamma**3,
            3: 3.0 * self.delta * self.alpha**2 * self.beta / gamma**5,
            4: 3.0 * self.delta * self.alpha**2 * (self.alpha**2 + 4.0 * self.beta**2) / gamma**7,
        }

    def innovation_cumulants(self) -> dict[int, float]:
        return {order: (1.0 - self.c**order) * value for order, value in self.stationary_cumulants().items()}

    def _finalize_cdf_grid(self, x_unique: np.ndarray, cdf_unique: np.ndarray) -> None:
        """Store a monotone CDF grid for linear CDF and quantile evaluation."""
        if len(cdf_unique) < 50:
            raise RuntimeError("Too few usable CDF grid points.")
        if cdf_unique[0] > 0.0:
            cdf_unique[0] = 0.0
        if cdf_unique[-1] < 1.0:
            cdf_unique[-1] = 1.0
        if not np.all(np.diff(cdf_unique) > 0):
            raise RuntimeError("CDF grid is not strictly increasing.")
        if not np.all(np.diff(x_unique) > 0):
            raise RuntimeError("x grid is not strictly increasing.")

        self.x_cdf = np.asarray(x_unique, dtype=float)
        self.cdf = np.asarray(cdf_unique, dtype=float)

    def _validate_shift(self, a: float) -> None:
        if a == 0:
            raise ValueError("The shifted-CDF complex shift a must be non-zero.")
        lower = -0.5 * self.p_minus
        upper = 0.5 * self.p_plus
        if not (lower < a < upper):
            raise ValueError(f"Shift a must satisfy {lower} < a < {upper}. Received a={a}.")

    def _automatic_shifts(self) -> list[float]:
        positive = self.shift_fraction * 0.5 * self.p_plus
        negative = -self.shift_fraction * 0.5 * self.p_minus
        if self.shift == "positive":
            return [positive]
        if self.shift == "negative":
            return [negative]
        return [positive, negative]

    def recommended_h(self, a: float) -> float:
        """Baviera-Manzoni spacing rule for the NIG-OU innovation."""
        self._validate_shift(float(a))
        ell = self.delta * (1.0 - self.c)
        if ell <= 0:
            raise RuntimeError("The NIG innovation CF decay coefficient is not positive.")
        return float(max(np.sqrt(2.0 * np.pi * abs(a) / (ell * self.n_fft)), 0.01))

    @staticmethod
    def _largest_valid_monotone_block(cdf: np.ndarray) -> tuple[int, int]:
        point_ok = np.isfinite(cdf) & (cdf >= 0.0) & (cdf <= 1.0)
        best_start = 0
        best_stop = 0
        current_start: int | None = None
        for index in range(cdf.size):
            if not point_ok[index]:
                if current_start is not None and index - current_start > best_stop - best_start:
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
        if current_start is not None and cdf.size - current_start > best_stop - best_start:
            best_start, best_stop = current_start, cdf.size
        return best_start, best_stop

    def _fft_cdf_candidate(self, a: float) -> _CDFCandidate:
        self._validate_shift(float(a))
        h = self.user_h if self.user_h is not None else self.recommended_h(float(a))
        n = np.arange(self.n_fft, dtype=float)
        frequencies = (n + 0.5) * h
        shifted_frequencies = frequencies - 1j * a
        coefficients = self.innovation_cf(shifted_frequencies) / (1j * frequencies + a)
        dx = 2.0 * np.pi / (self.n_fft * h)
        x = (np.arange(self.n_fft, dtype=float) - self.n_fft / 2.0) * dx
        alternating = np.where((np.arange(self.n_fft) % 2) == 0, 1.0, -1.0)
        fft_values = np.fft.fft(coefficients * alternating)
        cdf_raw = (1.0 if a > 0 else 0.0) - (h / np.pi) * np.exp(np.clip(-a * x, -700.0, 700.0)) * np.real(
            np.exp(-0.5j * x * h) * fft_values
        )

        start, stop = self._largest_valid_monotone_block(cdf_raw)
        if stop - start < 50:
            raise RuntimeError(f"Shifted-CDF reconstruction has too few valid monotone points for a={a}, h={h}.")
        x_valid = x[start:stop]
        cdf_valid = cdf_raw[start:stop]
        interior = np.flatnonzero((cdf_valid > 1e-15) & (cdf_valid < 1.0 - 1e-15))
        if interior.size < 50:
            raise RuntimeError("Too few interior CDF points for tail fitting.")
        x_valid = x_valid[int(interior[0]) : int(interior[-1]) + 1]
        cdf_valid = cdf_valid[int(interior[0]) : int(interior[-1]) + 1]

        left_rate = (np.log(cdf_valid[1]) - np.log(cdf_valid[0])) / (x_valid[1] - x_valid[0])
        if self.right_tail_rate == "paper":
            right_rate = (np.log(cdf_valid[-1]) - np.log(cdf_valid[-2])) / (x_valid[-1] - x_valid[-2])
        else:
            right_rate = (np.log1p(-cdf_valid[-2]) - np.log1p(-cdf_valid[-1])) / (x_valid[-1] - x_valid[-2])
        if not np.isfinite(left_rate) or left_rate <= 0:
            raise RuntimeError("Invalid left exponential-tail rate.")
        if not np.isfinite(right_rate) or right_rate <= 0:
            raise RuntimeError("Invalid right exponential-tail rate.")
        return _CDFCandidate(
            a=float(a),
            h=float(h),
            dx=float(dx),
            x_full=x,
            cdf_full=cdf_raw,
            x=x_valid,
            cdf=cdf_valid,
            left_rate=float(left_rate),
            right_rate=float(right_rate),
            tail_score=float(max(cdf_valid[0], 1.0 - cdf_valid[-1])),
        )

    def _build_shifted_cdf_sampler(self) -> None:
        shifts = [self.user_a] if self.user_a is not None else self._automatic_shifts()
        candidates: list[_CDFCandidate] = []
        failures: list[str] = []
        for shift_value in shifts:
            assert shift_value is not None
            try:
                candidates.append(self._fft_cdf_candidate(float(shift_value)))
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                failures.append(f"a={shift_value}: {exc}")
        if not candidates:
            raise RuntimeError("Shifted-CDF construction failed. " + "; ".join(failures))
        candidate = min(candidates, key=lambda item: item.tail_score)

        self.a = candidate.a
        self.h = candidate.h
        self.dx = candidate.dx
        self.x_grid = candidate.x_full
        self.cdf_raw = candidate.cdf_full
        self.x_cdf = candidate.x
        self.cdf = candidate.cdf
        self.left_tail_rate = candidate.left_rate
        self.right_tail_rate_value = candidate.right_rate
        self.tail_score = candidate.tail_score
        self.fft_diagnostics = {
            "sampler": self.sampler,
            "method": "Baviera-Manzoni shifted-CDF equation (15)",
            "inverse_sampler": "linear_np_interp",
            "n_fft": self.n_fft,
            "selected_a": self.a,
            "selected_h": self.h,
            "du": self.h,
            "dx": self.dx,
            "right_tail_rate_method": self.right_tail_rate,
            "tail_probability_tolerance": self.tail_probability_tolerance,
            "tail_score": self.tail_score,
            "tail_validation_passed": self.tail_score <= self.tail_probability_tolerance,
            "left_tail_rate": self.left_tail_rate,
            "right_tail_rate": self.right_tail_rate_value,
            "x_min_full": float(self.x_grid[0]),
            "x_max_full": float(self.x_grid[-1]),
            "x_min_used": float(self.x_cdf[0]),
            "x_max_used": float(self.x_cdf[-1]),
            "cdf_left_used": float(self.cdf[0]),
            "cdf_right_used": float(self.cdf[-1]),
            "num_inverse_points": int(len(self.cdf)),
            "candidate_failures": failures,
        }
        if self.tail_score > self.tail_probability_tolerance:
            raise RuntimeError(
                "Shifted-CDF reconstruction failed the tail-domain check: "
                f"{self.tail_score:.3e} > {self.tail_probability_tolerance:.3e}."
            )

    def _build_density_sampler(self) -> None:
        """Build an FFT density grid and its monotone linear inverse CDF."""
        n_fft = self.n_fft
        du = self.du
        k = np.arange(n_fft)
        u = (k - n_fft // 2) * du
        dx = 2 * np.pi / (n_fft * du)
        x = (np.arange(n_fft) - n_fft // 2) * dx

        phi = self.innovation_cf(u)
        fft_values = np.fft.fft(phi * ((-1.0) ** k))
        phase = (-1.0) ** (np.arange(n_fft) + n_fft // 2)
        density_raw = np.real((du / (2 * np.pi)) * phase * fft_values)

        raw_area = float(np.trapezoid(density_raw, x))
        negative_fraction = float(np.mean(density_raw < 0))
        negative_mass = float(np.trapezoid(np.clip(-density_raw, 0.0, None), x))

        density = np.clip(density_raw, 0.0, None)
        clipped_area = float(np.trapezoid(density, x))
        if not np.isfinite(clipped_area) or clipped_area <= 0:
            raise RuntimeError("FFT density has non-positive mass after clipping.")
        density = density / clipped_area

        increments = 0.5 * (density[:-1] + density[1:]) * np.diff(x)
        cdf = np.r_[0.0, np.cumsum(increments)]
        cdf = cdf / cdf[-1]
        cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))

        keep = np.r_[True, np.diff(cdf) > 1e-12]
        x_unique = x[keep]
        cdf_unique = cdf[keep]
        self.x_grid = x
        self.density_raw = density_raw
        self.density = density
        self._finalize_cdf_grid(x_unique, cdf_unique)
        self.fft_diagnostics = {
            "sampler": self.sampler,
            "inverse_sampler": "linear_np_interp",
            "n_fft": self.n_fft,
            "du": self.du,
            "dx": float(dx),
            "eta": self.eta,
            "u_min": float(u[0]),
            "u_max": float(u[-1]),
            "x_min_full": float(x[0]),
            "x_max_full": float(x[-1]),
            "x_min_used": float(self.x_cdf[0]),
            "x_max_used": float(self.x_cdf[-1]),
            "raw_density_min": float(np.nanmin(density_raw)),
            "raw_density_max": float(np.nanmax(density_raw)),
            "raw_density_area": raw_area,
            "negative_density_fraction": negative_fraction,
            "negative_density_mass": negative_mass,
            "clipped_density_area": clipped_area,
            "num_inverse_points": int(len(self.cdf)),
        }

    def cdf_linear(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted CDF with linear interpolation and shifted-CDF tails."""
        values = np.asarray(x, dtype=float)
        scalar = values.ndim == 0
        values_1d = np.atleast_1d(values)
        if self.sampler != "shifted_cdf":
            out = np.interp(values_1d, self.x_cdf, self.cdf, left=0.0, right=1.0)
            return out[0] if scalar else out
        out = np.empty_like(values_1d)
        left = values_1d < self.x_cdf[0]
        right = values_1d > self.x_cdf[-1]
        middle = ~(left | right)
        out[left] = self.cdf[0] * np.exp(self.left_tail_rate * (values_1d[left] - self.x_cdf[0]))
        out[right] = 1.0 - (1.0 - self.cdf[-1]) * np.exp(
            -self.right_tail_rate_value * (values_1d[right] - self.x_cdf[-1])
        )
        out[middle] = np.interp(values_1d[middle], self.x_cdf, self.cdf)
        out = np.clip(out, 0.0, 1.0)
        return out[0] if scalar else out

    def quantile_linear(self, p: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted monotone linear inverse CDF."""
        probs = np.asarray(p, dtype=float)
        scalar = probs.ndim == 0
        probs_1d = np.atleast_1d(probs)
        if np.any((probs_1d <= 0.0) | (probs_1d >= 1.0)):
            raise ValueError("Probabilities must lie strictly between 0 and 1.")
        if self.sampler != "shifted_cdf":
            out = np.interp(probs_1d, self.cdf, self.x_cdf)
            return out[0] if scalar else out
        out = np.empty_like(probs_1d)
        left = probs_1d < self.cdf[0]
        right = probs_1d > self.cdf[-1]
        middle = ~(left | right)
        out[left] = self.x_cdf[0] + np.log(probs_1d[left] / self.cdf[0]) / self.left_tail_rate
        out[right] = self.x_cdf[-1] - np.log((1.0 - probs_1d[right]) / (1.0 - self.cdf[-1])) / self.right_tail_rate_value
        out[middle] = np.interp(probs_1d[middle], self.cdf, self.x_cdf)
        if np.any(~np.isfinite(out)):
            raise RuntimeError("Linear inverse CDF returned NaN or inf.")
        return out[0] if scalar else out

    def sample_innovations(self, n: int) -> np.ndarray:
        """Sample eps_t from the fitted FFT CDF using monotone linear inversion."""
        uniforms = self.rng.random(int(n))
        eps = self.quantile_linear(uniforms)
        if np.any(~np.isfinite(eps)):
            raise RuntimeError("Linear inverse CDF returned NaN or inf.")
        return np.asarray(eps, dtype=float)

    def sample_stationary(self, n: int = 1) -> np.ndarray:
        """Sample from stationary NIG(alpha,beta,mu,delta)."""
        return stats.norminvgauss.rvs(
            self.alpha * self.delta,
            self.beta * self.delta,
            loc=self.mu,
            scale=self.delta,
            size=int(n),
            random_state=self.rng,
        )

    def simulate(
        self,
        n: int,
        x0: float | None = None,
        stationary_start: bool = True,
        burn_in: int = 0,
    ) -> np.ndarray:
        """Simulate a stationary NIG-OU path of length n."""
        n = int(n)
        burn_in = int(burn_in)
        if n <= 1:
            raise ValueError("Need n > 1.")
        total = n + burn_in
        path = np.empty(total, dtype=float)
        if x0 is not None:
            path[0] = float(x0)
        elif stationary_start:
            path[0] = self.sample_stationary(1)[0]
        else:
            path[0] = self.stationary_mean()

        eps = self.sample_innovations(total - 1)
        for t in range(1, total):
            path[t] = self.c * path[t - 1] + eps[t - 1]
        return path[burn_in:] if burn_in > 0 else path

    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        x0: float | np.ndarray | None = None,
        stationary_start: bool = True,
        burn_in: int = 0,
    ) -> np.ndarray:
        """Simulate many paths for ``n_steps`` transitions after one FFT-CDF build."""

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
                raise ValueError("x0 must be scalar or have shape (n_paths,).")
        elif stationary_start:
            paths[:, 0] = self.sample_stationary(n_paths)
        else:
            paths[:, 0] = self.stationary_mean()

        eps = self.sample_innovations(n_paths * (total - 1)).reshape(n_paths, total - 1)
        for t in range(1, total):
            paths[:, t] = self.c * paths[:, t - 1] + eps[:, t - 1]
        return paths[:, burn_in:] if burn_in > 0 else paths

    def validate_innovations(self, n: int = 20_000) -> dict[str, float]:
        """Compare sampled innovation cumulants with theoretical cumulants."""

        sample = self.sample_innovations(int(n))
        centered = sample - float(np.mean(sample))
        sample_k2 = float(np.mean(centered**2))
        theoretical = self.innovation_cumulants()
        return {
            "sample_kappa_1": float(np.mean(sample)),
            "sample_kappa_2": sample_k2,
            "sample_kappa_3": float(np.mean(centered**3)),
            "sample_kappa_4": float(np.mean(centered**4) - 3.0 * sample_k2**2),
            "theoretical_kappa_1": float(theoretical[1]),
            "theoretical_kappa_2": float(theoretical[2]),
            "theoretical_kappa_3": float(theoretical[3]),
            "theoretical_kappa_4": float(theoretical[4]),
        }


def build_nig_ou_simulator_with_fallback(
    alpha: float,
    beta: float,
    mu: float,
    delta: float,
    lam: float,
    dt: float,
    seed: int = 123,
    shifted_attempts: tuple[dict[str, Any], ...] = (
        {"n_fft": DEFAULT_SIMULATOR_N_FFT, "shift_fraction": 0.95},
        {"n_fft": 2**16, "shift_fraction": 0.90},
    ),
    density_n_fft: int = DEFAULT_SIMULATOR_N_FFT,
    density_du: float = 20.0,
    right_tail_rate: RightTailRate = "survival_match",
    density_negative_mass_tol: float = 1e-4,
    density_raw_area_tol: float = 1e-3,
) -> tuple[NIGOUFGMC, dict[str, Any]]:
    """Build a NIG simulator using shifted-CDF first and density-FFT fallback."""

    failures: list[str] = []
    for attempt_index, attempt in enumerate(shifted_attempts, start=1):
        try:
            sim = NIGOUFGMC(
                alpha=alpha,
                beta=beta,
                mu=mu,
                delta=delta,
                lam=lam,
                dt=dt,
                seed=seed,
                sampler="shifted_cdf",
                n_fft=int(attempt.get("n_fft", DEFAULT_SIMULATOR_N_FFT)),
                du=None,
                shift_fraction=float(attempt.get("shift_fraction", 0.95)),
                right_tail_rate=right_tail_rate,
            )
            diag = {
                "simulation_method": "shifted_cdf",
                "simulation_attempt": attempt_index,
                "simulation_validation_passed": True,
                "fallback_used": False,
                "attempt_failures": failures,
                **{f"sim_{key}": value for key, value in sim.fft_diagnostics.items()},
            }
            return sim, diag
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failures.append(f"shifted_attempt_{attempt_index}: {type(exc).__name__}: {exc}")

    sim = NIGOUFGMC(
        alpha=alpha,
        beta=beta,
        mu=mu,
        delta=delta,
        lam=lam,
        dt=dt,
        seed=seed,
        sampler="density_fft",
        n_fft=int(density_n_fft),
        du=float(density_du),
    )
    raw_area = float(sim.fft_diagnostics.get("raw_density_area", np.nan))
    negative_mass = float(sim.fft_diagnostics.get("negative_density_mass", np.nan))
    accepted = bool(
        np.isfinite(raw_area)
        and np.isfinite(negative_mass)
        and abs(raw_area - 1.0) < float(density_raw_area_tol)
        and negative_mass < float(density_negative_mass_tol)
    )
    diag = {
        "simulation_method": "density_fft_fallback",
        "simulation_attempt": len(shifted_attempts) + 1,
        "simulation_validation_passed": accepted,
        "fallback_used": True,
        "attempt_failures": failures,
        "density_raw_area_tolerance": float(density_raw_area_tol),
        "density_negative_mass_tolerance": float(density_negative_mass_tol),
        **{f"sim_{key}": value for key, value in sim.fft_diagnostics.items()},
    }
    if not accepted:
        raise RuntimeError(f"Density-FFT fallback failed diagnostics: {diag}")
    return sim, diag


def simulate_nig_ou_process_fft_inversion(
    alpha: float,
    beta: float,
    delta: float,
    lam: float,
    Delta: float,
    n_steps: int,
    mu: float = 0.0,
    n_fft: int = DEFAULT_SIMULATOR_N_FFT,
    du: float | None = None,
    eta: float | None = None,
    sampler: str = "shifted_cdf",
    seed: int = 42,
    stationary_start: bool = True,
    burn_in: int = 0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    simulator = NIGOUFGMC(
        alpha=alpha,
        beta=beta,
        mu=mu,
        delta=delta,
        lam=lam,
        dt=Delta,
        n_fft=n_fft,
        du=du,
        eta=eta,
        sampler=sampler,
        seed=seed,
    )
    path = simulator.simulate(
        n=int(n_steps) + 1,
        stationary_start=stationary_start,
        burn_in=burn_in,
    )
    return path, simulator.fft_diagnostics
