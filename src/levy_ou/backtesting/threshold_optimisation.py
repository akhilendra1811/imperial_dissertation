"""Model-agnostic threshold optimisation from simulated OU paths."""

from __future__ import annotations

from typing import Any

import numpy as np


def choose_best_threshold(thresholds: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return the threshold with the highest finite score."""

    b = np.asarray(thresholds, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(b) & np.isfinite(s)
    if not np.any(mask):
        return float("nan"), float("nan")
    idx = int(np.argmax(s[mask]))
    return float(b[mask][idx]), float(s[mask][idx])


def next_true_indices(mask: np.ndarray) -> np.ndarray:
    """For every index, return the next index where ``mask`` is true."""

    values = np.asarray(mask, dtype=bool)
    out = np.full(len(values) + 1, -1, dtype=int)
    next_idx = -1
    for idx in range(len(values) - 1, -1, -1):
        if bool(values[idx]):
            next_idx = idx
        out[idx] = next_idx
    return out


def first_crossing_times(path: np.ndarray, thresholds: np.ndarray, side: str) -> np.ndarray:
    """First crossing index for each threshold."""

    x = np.asarray(path, dtype=float)
    levels = np.asarray(thresholds, dtype=float)
    if side == "above":
        crossed = x[:, None] >= levels[None, :]
    elif side == "below":
        crossed = x[:, None] <= levels[None, :]
    else:
        raise ValueError("side must be 'above' or 'below'.")
    any_cross = crossed.any(axis=0)
    first = np.where(any_cross, crossed.argmax(axis=0), len(x) + 1)
    return first.astype(int)


def threshold_arrays(
    paths: np.ndarray,
    mu: float,
    sigma: float,
    grid_points: int = 25,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 3.0,
    exit_rule: str = "mean",
) -> dict[str, np.ndarray]:
    """Precompute repeated-trade simulated profits for a threshold grid.

    This takes only simulated spread paths. It does not know whether the paths
    came from NIG-OU, CGMY-OU, Gaussian OU, symmetric BG-OU, or another simulator.
    Each simulated path is replayed with the same entry/exit convention used by
    the real trading replay: after a position exits, the path can enter again.
    """

    x = np.asarray(paths, dtype=float)
    if x.ndim != 2:
        raise ValueError("paths must be a 2D array with shape (n_paths, n_steps).")
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("mu must be finite and sigma must be positive.")
    if exit_rule not in {"mean", "opposite_band"}:
        raise ValueError("exit_rule must be 'mean' or 'opposite_band'.")

    distances = np.linspace(
        float(min_sigma_multiple) * float(sigma),
        float(max_sigma_multiple) * float(sigma),
        int(grid_points),
    )
    n_paths, n_steps = x.shape
    grid = len(distances)
    profits = np.zeros((n_paths, grid, grid), dtype=np.float64)
    trade_counts = np.zeros((n_paths, grid, grid), dtype=np.int32)
    forced_counts = np.zeros((n_paths, grid, grid), dtype=np.int32)
    holdings = np.zeros((n_paths, grid, grid), dtype=np.float64)

    upper = float(mu) + distances
    lower = float(mu) - distances
    upper_grid = upper[:, None]
    lower_grid = lower[None, :]

    for path_idx, path in enumerate(x):
        position = np.zeros((grid, grid), dtype=np.int8)
        entry_value = np.zeros((grid, grid), dtype=np.float64)
        entry_time = np.zeros((grid, grid), dtype=np.int32)
        total_profit = np.zeros((grid, grid), dtype=np.float64)
        total_holding = np.zeros((grid, grid), dtype=np.float64)
        total_trades = np.zeros((grid, grid), dtype=np.int32)
        total_forced = np.zeros((grid, grid), dtype=np.int32)

        for t, value in enumerate(path):
            if not np.isfinite(value):
                continue
            previous_position = position.copy()

            if exit_rule == "mean":
                close_long = (previous_position == 1) & (value >= float(mu))
                close_short = (previous_position == -1) & (value <= float(mu))
            else:
                close_long = (previous_position == 1) & (value >= upper_grid)
                close_short = (previous_position == -1) & (value <= lower_grid)

            close_any = close_long | close_short
            if np.any(close_any):
                trade_profit = np.zeros((grid, grid), dtype=np.float64)
                trade_profit[close_long] = float(value) - entry_value[close_long]
                trade_profit[close_short] = entry_value[close_short] - float(value)
                total_profit += trade_profit
                total_holding[close_any] += t - entry_time[close_any]
                total_trades[close_any] += 1
                position[close_any] = 0

            can_enter = previous_position == 0
            enter_long = can_enter & (value <= lower_grid)
            enter_short = can_enter & (value >= upper_grid)

            if np.any(enter_long | enter_short):
                position[enter_long] = 1
                position[enter_short] = -1
                entry_value[enter_long | enter_short] = float(value)
                entry_time[enter_long | enter_short] = int(t)

        if n_steps:
            final_value = float(path[-1])
            forced_long = position == 1
            forced_short = position == -1
            forced_any = forced_long | forced_short

            if np.any(forced_any) and np.isfinite(final_value):
                forced_profit = np.zeros((grid, grid), dtype=np.float64)
                forced_profit[forced_long] = final_value - entry_value[forced_long]
                forced_profit[forced_short] = entry_value[forced_short] - final_value
                total_profit += forced_profit
                total_holding[forced_any] += (n_steps - 1) - entry_time[forced_any]
                total_trades[forced_any] += 1
                total_forced[forced_any] += 1

        profits[path_idx] = total_profit
        holdings[path_idx] = total_holding
        trade_counts[path_idx] = total_trades
        forced_counts[path_idx] = total_forced

    return {
        "distances": distances,
        "profits": profits,
        "traded": trade_counts > 0,
        "trade_counts": trade_counts,
        "forced_counts": forced_counts,
        "holdings": holdings,
        "exit_rule": np.array(exit_rule),
    }


def threshold_arrays_one_trade(
    paths: np.ndarray,
    mu: float,
    sigma: float,
    grid_points: int = 25,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 3.0,
) -> dict[str, np.ndarray]:
    """Precompute one-trade simulated profits for a threshold grid.

    This is the older one-trade-cycle version. For each simulated path and each
    threshold pair, it enters only the first side crossed and exits once at the
    mean. It is kept for comparison with the repeated-trade optimiser.
    """

    x = np.asarray(paths, dtype=float)
    if x.ndim != 2:
        raise ValueError("paths must be a 2D array with shape (n_paths, n_steps).")
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("mu must be finite and sigma must be positive.")

    distances = np.linspace(
        float(min_sigma_multiple) * float(sigma),
        float(max_sigma_multiple) * float(sigma),
        int(grid_points),
    )
    n_paths, n_steps = x.shape
    grid = len(distances)

    profits = np.zeros((n_paths, grid, grid), dtype=np.float64)
    traded = np.zeros((n_paths, grid, grid), dtype=bool)
    forced = np.zeros((n_paths, grid, grid), dtype=bool)
    holdings = np.zeros((n_paths, grid, grid), dtype=np.float64)

    upper_thresholds = float(mu) + distances
    lower_thresholds = float(mu) - distances
    no_entry = n_steps + 1

    for path_idx, path in enumerate(x):
        upper_entry = first_crossing_times(path, upper_thresholds, "above")
        lower_entry = first_crossing_times(path, lower_thresholds, "below")

        next_below_mu = next_true_indices(path <= float(mu))
        next_above_mu = next_true_indices(path >= float(mu))

        short_forced = np.ones(grid, dtype=bool)
        short_profit = np.zeros(grid, dtype=float)
        short_holding = np.zeros(grid, dtype=float)

        for p_idx, entry_idx in enumerate(upper_entry):
            if entry_idx >= no_entry:
                continue

            exit_idx = next_below_mu[min(entry_idx + 1, n_steps)]
            if exit_idx < 0:
                exit_idx = n_steps - 1
            else:
                short_forced[p_idx] = False

            short_profit[p_idx] = float(path[entry_idx] - path[exit_idx])
            short_holding[p_idx] = float(exit_idx - entry_idx)

        long_forced = np.ones(grid, dtype=bool)
        long_profit = np.zeros(grid, dtype=float)
        long_holding = np.zeros(grid, dtype=float)

        for m_idx, entry_idx in enumerate(lower_entry):
            if entry_idx >= no_entry:
                continue

            exit_idx = next_above_mu[min(entry_idx + 1, n_steps)]
            if exit_idx < 0:
                exit_idx = n_steps - 1
            else:
                long_forced[m_idx] = False

            long_profit[m_idx] = float(path[exit_idx] - path[entry_idx])
            long_holding[m_idx] = float(exit_idx - entry_idx)

        upper_grid = upper_entry[:, None]
        lower_grid = lower_entry[None, :]

        has_short = upper_grid < no_entry
        has_long = lower_grid < no_entry

        choose_short = has_short & (~has_long | (upper_grid <= lower_grid))
        choose_long = has_long & ~choose_short
        did_trade = choose_short | choose_long

        p = np.zeros((grid, grid), dtype=float)
        p[choose_short] = np.broadcast_to(short_profit[:, None], (grid, grid))[choose_short]
        p[choose_long] = np.broadcast_to(long_profit[None, :], (grid, grid))[choose_long]

        h = np.zeros((grid, grid), dtype=float)
        h[choose_short] = np.broadcast_to(short_holding[:, None], (grid, grid))[choose_short]
        h[choose_long] = np.broadcast_to(long_holding[None, :], (grid, grid))[choose_long]

        f = np.zeros((grid, grid), dtype=bool)
        f[choose_short] = np.broadcast_to(short_forced[:, None], (grid, grid))[choose_short]
        f[choose_long] = np.broadcast_to(long_forced[None, :], (grid, grid))[choose_long]

        profits[path_idx] = p
        holdings[path_idx] = h
        traded[path_idx] = did_trade
        forced[path_idx] = f

    trade_counts = traded.astype(np.int32)
    forced_counts = (forced & traded).astype(np.int32)

    return {
        "distances": distances,
        "profits": profits,
        "traded": traded,
        "trade_counts": trade_counts,
        "forced": forced,
        "forced_counts": forced_counts,
        "holdings": holdings,
        "exit_rule": np.array("one_trade_mean"),
    }


def threshold_with_variance_penalty(
    arr: dict[str, np.ndarray],
    mu: float,
    sigma: float,
    c: float,
    gamma: float = 0.0,
    gamma_multiplier: float = 0.0,
    gamma_base: float = 0.0,
) -> dict[str, Any]:
    """Choose asymmetric thresholds maximising mean(P-c) - gamma * var(P-c).

    ``P`` is the total repeated-trade profit over each simulated path. Costs are
    charged once per simulated trade, so paths with more entries/exits pay more.
    """

    distances = np.asarray(arr["distances"], dtype=float)
    profits = np.asarray(arr["profits"], dtype=float)
    trade_counts_by_path = np.asarray(arr.get("trade_counts", arr["traded"]), dtype=float)
    forced_counts_by_path = np.asarray(
        arr.get("forced_counts", np.zeros_like(trade_counts_by_path)),
        dtype=float,
    )
    holdings = np.asarray(arr["holdings"], dtype=float)
    n_paths = int(profits.shape[0])

    net_path_profit = profits - float(c) * trade_counts_by_path
    mean_net = net_path_profit.mean(axis=0)
    var_net = net_path_profit.var(axis=0, ddof=1) if n_paths > 1 else np.zeros_like(mean_net)
    std_net = np.sqrt(np.maximum(var_net, 0.0))

    traded_paths = trade_counts_by_path > 0
    traded_path_count = traded_paths.sum(axis=0)
    trade_count = trade_counts_by_path.sum(axis=0)
    forced_count = forced_counts_by_path.sum(axis=0)
    positive_path_count = ((net_path_profit > 0.0) & traded_paths).sum(axis=0)
    holding_sum = holdings.sum(axis=0)
    gross_mean = profits.mean(axis=0)

    score = mean_net - float(gamma) * var_net
    feasible = trade_count > 0
    score = np.where(feasible & np.isfinite(score), score, -np.inf)

    if not np.isfinite(score).any():
        return {
            "threshold_valid": False,
            "threshold_reason": "no feasible threshold grid cell",
            "gamma_multiplier": float(gamma_multiplier),
            "gamma": float(gamma),
            "gamma_base": float(gamma_base),
        }

    p_idx, m_idx = np.unravel_index(int(np.nanargmax(score)), score.shape)
    trades = int(trade_count[p_idx, m_idx])
    paths_with_trades = int(traded_path_count[p_idx, m_idx])
    d_plus = float(distances[p_idx])
    d_minus = float(distances[m_idx])

    return {
        "threshold_valid": True,
        "threshold_reason": "",
        "d_plus": d_plus,
        "d_minus": d_minus,
        "d_plus_bps": d_plus * 10000.0,
        "d_minus_bps": d_minus * 10000.0,
        "full_band": d_plus + d_minus,
        "full_band_bps": (d_plus + d_minus) * 10000.0,
        "d_plus_sigma": float(d_plus / sigma),
        "d_minus_sigma": float(d_minus / sigma),
        "upper_entry": float(mu + d_plus),
        "lower_entry": float(mu - d_minus),
        "objective_score": float(score[p_idx, m_idx]),
        "simulated_mean_profit": float(mean_net[p_idx, m_idx]),
        "simulated_mean_net_profit": float(mean_net[p_idx, m_idx]),
        "simulated_mean_gross_profit": float(gross_mean[p_idx, m_idx]),
        "simulated_total_trade_count": trades,
        "simulated_paths_with_trades": paths_with_trades,
        "simulated_profit_variance": float(var_net[p_idx, m_idx]),
        "simulated_profit_std": float(std_net[p_idx, m_idx]),
        "simulated_trade_fraction": float(paths_with_trades / n_paths),
        "simulated_forced_exit_fraction": float(forced_count[p_idx, m_idx] / trades) if trades else 0.0,
        "simulated_positive_path_fraction": float(positive_path_count[p_idx, m_idx] / paths_with_trades)
        if paths_with_trades
        else 0.0,
        "simulated_win_rate": float(positive_path_count[p_idx, m_idx] / paths_with_trades)
        if paths_with_trades
        else 0.0,
        "simulated_mean_holding_minutes": float(holding_sum[p_idx, m_idx] / trades) if trades else np.nan,
        "simulated_trade_count": trades,
        "simulated_mean_trades_per_path": float(trades / n_paths),
        "simulated_mean_trades_per_traded_path": float(trades / paths_with_trades) if paths_with_trades else 0.0,
        "gamma_multiplier": float(gamma_multiplier),
        "gamma": float(gamma),
        "gamma_base": float(gamma_base),
    }


def gamma_base_for_cost(arr: dict[str, np.ndarray], c: float) -> tuple[float, dict[str, Any]]:
    """Scale gamma from the no-penalty optimum for one transaction cost."""

    baseline = threshold_with_variance_penalty(
        arr=arr,
        mu=0.0,
        sigma=1.0,
        c=float(c),
        gamma=0.0,
        gamma_multiplier=0.0,
        gamma_base=0.0,
    )
    m0 = float(baseline.get("simulated_mean_profit", np.nan))
    s20 = float(baseline.get("simulated_profit_variance", np.nan))
    gamma_base = abs(m0) / s20 if np.isfinite(m0) and np.isfinite(s20) and s20 > 1e-18 else 0.0
    return float(gamma_base), baseline


def optimize_thresholds_from_paths(
    paths: np.ndarray,
    mu: float,
    sigma: float,
    transaction_cost_c: float = 0.0,
    gamma_multiplier: float = 0.0,
    gamma: float | None = None,
    grid_points: int = 25,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 3.0,
    exit_rule: str = "mean",
) -> dict[str, Any]:
    """Convenience wrapper for one simulated path matrix and one cost/gamma case."""

    arr = threshold_arrays(
        paths=paths,
        mu=mu,
        sigma=sigma,
        grid_points=grid_points,
        min_sigma_multiple=min_sigma_multiple,
        max_sigma_multiple=max_sigma_multiple,
        exit_rule=exit_rule,
    )
    gamma_base, _baseline = gamma_base_for_cost(arr, c=float(transaction_cost_c))
    raw_gamma = float(gamma) if gamma is not None else float(gamma_multiplier) * gamma_base

    return threshold_with_variance_penalty(
        arr=arr,
        mu=mu,
        sigma=sigma,
        c=float(transaction_cost_c),
        gamma=raw_gamma,
        gamma_multiplier=float(gamma_multiplier),
        gamma_base=gamma_base,
    )


def optimize_thresholds_from_paths_one_trade(
    paths: np.ndarray,
    mu: float,
    sigma: float,
    transaction_cost_c: float = 0.0,
    gamma_multiplier: float = 0.0,
    gamma: float | None = None,
    grid_points: int = 25,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 3.0,
) -> dict[str, Any]:
    """Convenience wrapper using the older one-trade-cycle optimiser."""

    arr = threshold_arrays_one_trade(
        paths=paths,
        mu=mu,
        sigma=sigma,
        grid_points=grid_points,
        min_sigma_multiple=min_sigma_multiple,
        max_sigma_multiple=max_sigma_multiple,
    )
    gamma_base, _baseline = gamma_base_for_cost(arr, c=float(transaction_cost_c))
    raw_gamma = float(gamma) if gamma is not None else float(gamma_multiplier) * gamma_base

    return threshold_with_variance_penalty(
        arr=arr,
        mu=mu,
        sigma=sigma,
        c=float(transaction_cost_c),
        gamma=raw_gamma,
        gamma_multiplier=float(gamma_multiplier),
        gamma_base=gamma_base,
    )


def optimize_cost_gamma_grid(
    paths: np.ndarray,
    mu: float,
    sigma: float,
    cost_cases: dict[str, float],
    gamma_multipliers: tuple[float, ...] | list[float] = (0.0,),
    grid_points: int = 25,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 3.0,
    exit_rule: str = "mean",
) -> list[dict[str, Any]]:
    """Optimise thresholds for many cost cases and gamma multipliers."""

    arr = threshold_arrays(
        paths=paths,
        mu=mu,
        sigma=sigma,
        grid_points=grid_points,
        min_sigma_multiple=min_sigma_multiple,
        max_sigma_multiple=max_sigma_multiple,
        exit_rule=exit_rule,
    )
    rows: list[dict[str, Any]] = []

    for cost_case, c in cost_cases.items():
        if not np.isfinite(float(c)) or float(c) < 0.0:
            rows.append(
                {
                    "optimization_cost_case": cost_case,
                    "threshold_valid": False,
                    "threshold_reason": "invalid transaction cost",
                    "threshold_transaction_cost_c": float(c),
                }
            )
            continue

        gamma_base, baseline = gamma_base_for_cost(arr, c=float(c))

        for multiplier in gamma_multipliers:
            gamma_value = float(multiplier) * gamma_base
            result = threshold_with_variance_penalty(
                arr=arr,
                mu=mu,
                sigma=sigma,
                c=float(c),
                gamma=gamma_value,
                gamma_multiplier=float(multiplier),
                gamma_base=gamma_base,
            )
            rows.append(
                {
                    "optimization_cost_case": cost_case,
                    "threshold_transaction_cost_c": float(c),
                    "threshold_transaction_cost_c_bps": float(c) * 10000.0,
                    "baseline_mean_profit": baseline.get("simulated_mean_profit"),
                    "baseline_profit_variance": baseline.get("simulated_profit_variance"),
                    **result,
                }
            )

    return rows


def optimize_cost_gamma_grid_one_trade(
    paths: np.ndarray,
    mu: float,
    sigma: float,
    cost_cases: dict[str, float],
    gamma_multipliers: tuple[float, ...] | list[float] = (0.0,),
    grid_points: int = 25,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 3.0,
) -> list[dict[str, Any]]:
    """Optimise thresholds using the older one-trade-cycle simulation logic."""

    arr = threshold_arrays_one_trade(
        paths=paths,
        mu=mu,
        sigma=sigma,
        grid_points=grid_points,
        min_sigma_multiple=min_sigma_multiple,
        max_sigma_multiple=max_sigma_multiple,
    )
    rows: list[dict[str, Any]] = []

    for cost_case, c in cost_cases.items():
        if not np.isfinite(float(c)) or float(c) < 0.0:
            rows.append(
                {
                    "optimization_cost_case": cost_case,
                    "threshold_valid": False,
                    "threshold_reason": "invalid transaction cost",
                    "threshold_transaction_cost_c": float(c),
                }
            )
            continue

        gamma_base, baseline = gamma_base_for_cost(arr, c=float(c))

        for multiplier in gamma_multipliers:
            gamma_value = float(multiplier) * gamma_base
            result = threshold_with_variance_penalty(
                arr=arr,
                mu=mu,
                sigma=sigma,
                c=float(c),
                gamma=gamma_value,
                gamma_multiplier=float(multiplier),
                gamma_base=gamma_base,
            )
            rows.append(
                {
                    "optimization_cost_case": cost_case,
                    "threshold_transaction_cost_c": float(c),
                    "threshold_transaction_cost_c_bps": float(c) * 10000.0,
                    "baseline_mean_profit": baseline.get("simulated_mean_profit"),
                    "baseline_profit_variance": baseline.get("simulated_profit_variance"),
                    **result,
                }
            )

    return rows


__all__ = [
    "choose_best_threshold",
    "first_crossing_times",
    "gamma_base_for_cost",
    "next_true_indices",
    "optimize_cost_gamma_grid",
    "optimize_cost_gamma_grid_one_trade",
    "optimize_thresholds_from_paths",
    "optimize_thresholds_from_paths_one_trade",
    "threshold_arrays",
    "threshold_arrays_one_trade",
    "threshold_with_variance_penalty",
]
