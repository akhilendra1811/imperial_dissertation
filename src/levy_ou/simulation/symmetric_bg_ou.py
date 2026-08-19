"""Exact simulators for the symmetric bilateral-Gamma OU process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymmetricBGOUParams:
    """Parameters for a symmetric bilateral-Gamma OU process."""

    mu: float
    alpha: float
    beta: float
    kappa: float
    dt: float


class SymmetricBGOU:
    """Exact symmetric bilateral-Gamma OU simulator with two innovation methods."""

    def __init__(
        self,
        *,
        mu: float,
        alpha: float,
        beta: float,
        kappa: float,
        dt: float,
        seed: int | None = None,
    ) -> None:
        if not np.isfinite(mu):
            raise ValueError("mu must be finite.")
        if alpha <= 0 or beta <= 0 or kappa <= 0 or dt <= 0:
            raise ValueError("alpha, beta, kappa and dt must be positive.")

        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.dt = float(dt)
        self.c = float(np.exp(-self.kappa * self.dt))
        self.rng = np.random.default_rng(seed)

    def stationary_mean(self) -> float:
        return self.mu

    def stationary_variance(self) -> float:
        return 2.0 * self.alpha / self.beta**2

    def stationary_excess_kurtosis(self) -> float:
        return 3.0 / self.alpha

    def transition_atom_mass(self) -> float:
        return self.c ** (2.0 * self.alpha)

    def transition_atom_location(self) -> float:
        return (1.0 - self.c) * self.mu

    def sample_stationary(self, n: int = 1) -> np.ndarray:
        n = int(n)
        positive = self.rng.gamma(shape=self.alpha, scale=1.0 / self.beta, size=n)
        negative = self.rng.gamma(shape=self.alpha, scale=1.0 / self.beta, size=n)
        return self.mu + positive - negative

    def sample_remainders_polya(
        self,
        n: int,
        *,
        return_indices: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample exact innovations using the Polya/Erlang-mixture representation."""
        n = int(n)
        if n < 0:
            raise ValueError("n cannot be negative.")

        indices = self.rng.negative_binomial(n=self.alpha, p=self.c**2, size=n)
        positive = np.zeros(n, dtype=float)
        negative = np.zeros(n, dtype=float)
        active = indices > 0

        if np.any(active):
            positive[active] = self.rng.gamma(
                shape=indices[active],
                scale=self.c / self.beta,
            )
            negative[active] = self.rng.gamma(
                shape=indices[active],
                scale=self.c / self.beta,
            )

        remainder = self.transition_atom_location() + positive - negative
        if return_indices:
            return remainder, indices
        return remainder

    def _sample_gamma_c_remainder_compound_poisson(self, n: int) -> np.ndarray:
        """Sample a Gamma c-remainder as a compound Poisson sum."""
        n = int(n)
        intensity = self.alpha * np.log(1.0 / self.c)
        counts = self.rng.poisson(intensity, size=n)
        out = np.zeros(n, dtype=float)

        total = int(np.sum(counts))
        if total == 0:
            return out

        owners = np.repeat(np.arange(n), counts)
        uniform = self.rng.random(total)
        rates = self.beta * self.c ** (-uniform)
        jumps = self.rng.exponential(scale=1.0 / rates)
        np.add.at(out, owners, jumps)
        return out

    def sample_remainders_compound_poisson(self, n: int) -> np.ndarray:
        """Sample exact innovations using compound-Poisson random-rate jumps."""
        n = int(n)
        if n < 0:
            raise ValueError("n cannot be negative.")
        positive = self._sample_gamma_c_remainder_compound_poisson(n)
        negative = self._sample_gamma_c_remainder_compound_poisson(n)
        return self.transition_atom_location() + positive - negative

    def simulate(
        self,
        n_steps: int,
        *,
        x0: float | None = None,
        stationary_start: bool = True,
        method: str = "compound_poisson",
        return_indices: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate observations, including the initial observation."""
        n_steps = int(n_steps)
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2.")
        if method not in {"compound_poisson", "polya"}:
            raise ValueError("method must be 'compound_poisson' or 'polya'.")
        if return_indices and method != "polya":
            raise ValueError("return_indices is only available for method='polya'.")

        path = np.empty(n_steps, dtype=float)
        if x0 is not None:
            path[0] = float(x0)
        elif stationary_start:
            path[0] = float(self.sample_stationary(1)[0])
        else:
            path[0] = self.mu

        if method == "polya":
            remainder, indices = self.sample_remainders_polya(
                n_steps - 1,
                return_indices=True,
            )
        else:
            remainder = self.sample_remainders_compound_poisson(n_steps - 1)
            indices = np.empty(0, dtype=int)

        for index in range(1, n_steps):
            path[index] = self.c * path[index - 1] + remainder[index - 1]

        if return_indices:
            return path, indices
        return path
