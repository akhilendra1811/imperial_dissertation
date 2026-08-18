"""Reusable estimation and backtest pipelines for command-line runners."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from levy_ou.backtesting.threshold_optimisation import optimize_cost_gamma_grid
from levy_ou.backtesting.trade_replay import trade_real_window
from levy_ou.backtesting.optimal_gaussian import endres_5bps_round_trip_c
from levy_ou.data import DEFAULT_LOBSTER_DATA, load_pair_prices_from_lobster
from levy_ou.experiments.models import (
    DEFAULT_SIM_FFT_GRID_SIZE,
    fit_models_for_spread,
    model_result_row,
    model_scale,
    simulate_paths_from_fit,
)
from levy_ou.experiments.outputs import scalarize_mapping
from levy_ou.experiments.real_data import (
    formation_cost_cases,
    load_lobster_panel,
    pair_formation_and_trading_frames,
    scalar_trade_profit_summary,
    select_pair_windows,
)
from levy_ou.experiments.synthetic import synthetic_ou_spread, synthetic_trading_frame
from levy_ou.spreads import build_spread


_SAVED_BACKTEST_CONTEXT: dict[str, Any] = {}


def _finite_first(*values: Any) -> float:
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate):
            return candidate
    return float("nan")


def _parse_models(models: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(models, str):
        return [item.strip().lower() for item in models.split(",") if item.strip()]
    return [str(item).strip().lower() for item in models if str(item).strip()]


def _parse_floats(values: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(values, str):
        return tuple(float(item.strip()) for item in values.split(",") if item.strip())
    return tuple(float(item) for item in values)


def default_cost_cases(names: str = "c0,midquote_5bps") -> dict[str, float]:
    """Return threshold cost cases in spread/log-return units."""

    out: dict[str, float] = {}
    for raw_name in [item.strip() for item in str(names).split(",") if item.strip()]:
        name = raw_name.lower()
        if name in {"c0", "zero", "none"}:
            out["c0"] = 0.0
        elif name in {"midquote_5bps", "5bps", "endres_5bps"}:
            out["midquote_5bps"] = endres_5bps_round_trip_c()
        elif name in {"bidask_median_c", "bidask_worst_c"}:
            out[name] = np.nan
        else:
            out[raw_name] = float(raw_name)
    return out


def _window_cost_cases(names: str, formation: pd.DataFrame) -> dict[str, float]:
    requested = default_cost_cases(names)
    bidask_costs = formation_cost_cases(formation)
    out: dict[str, float] = {}
    for name, value in requested.items():
        if name in bidask_costs:
            out[name] = float(bidask_costs[name])
        else:
            out[name] = float(value)
    return out


def _init_saved_backtest_worker(context: dict[str, Any]) -> None:
    global _SAVED_BACKTEST_CONTEXT
    _SAVED_BACKTEST_CONTEXT = context


def _saved_backtest_row_worker(item: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    context = _SAVED_BACKTEST_CONTEXT
    row_idx, row_dict = item
    row = pd.Series(row_dict)
    panel = context["panel"]
    formation, trading = pair_formation_and_trading_frames(panel, row)
    metadata = {
        "dataset": row.get("dataset", "lobster"),
        "window_id": int(row["window_id"]),
        "pair_index": int(row.get("pair_index", row_idx + 1)) if pd.notna(row.get("pair_index", np.nan)) else int(row_idx + 1),
        "ticker_a": str(row["ticker_a"]).upper(),
        "ticker_b": str(row["ticker_b"]).upper(),
        "formation_start": row["formation_start"],
        "formation_end": row["formation_end"],
        "trading_start": row["trading_start"],
        "trading_end": row["trading_end"],
        "model": row.get("model", "unknown"),
        "formation_rows": int(len(formation)),
        "trading_rows": int(len(trading)),
    }
    if len(formation) < int(context["min_observations"]) or trading.empty:
        reason = "insufficient formation or trading rows"
        bad = {**metadata, "threshold_valid": False, "threshold_reason": reason}
        return bad, [bad], [bad], []

    fit = row.to_dict()
    if context.get("simulation_method"):
        fit["simulation_method"] = context["simulation_method"]
    spread = formation["spread"].to_numpy(dtype=float)
    window_costs = _window_cost_cases(context["cost_cases"], formation)
    try:
        paths = simulate_paths_from_fit(
            fit,
            x0=float(formation["spread"].iloc[-1]),
            n_paths=int(context["n_paths"]),
            n_steps=int(context["sim_steps"]),
            seed=int(context["seed"]) + 1000 + row_idx * 17,
            sim_fft_grid_size=int(context.get("sim_fft_grid_size", DEFAULT_SIM_FFT_GRID_SIZE)),
            nig_du=float(context.get("nig_du", 20.0)),
            cgmy_du=float(context.get("cgmy_du", 0.5)),
        )
        simulator_diagnostics = getattr(simulate_paths_from_fit, "last_simulator_diagnostics", {})
        mu = _finite_first(fit.get("u_form"), fit.get("mu"), np.mean(spread))
        scale = model_scale(fit, fallback_spread=spread)
        thresholds = optimize_cost_gamma_grid(
            paths,
            mu=mu,
            sigma=scale,
            cost_cases=window_costs,
            gamma_multipliers=context["gammas"],
            grid_points=int(context["grid_points"]),
            min_sigma_multiple=float(context["min_sigma_multiple"]),
            max_sigma_multiple=float(context["max_sigma_multiple"]),
            exit_rule=context["exit_rule"],
        )
    except Exception as exc:
        reason = repr(exc)
        bad = {**metadata, "threshold_valid": False, "threshold_reason": reason}
        return {**bad, **scalarize_mapping(fit)}, [bad], [bad], []

    estimate_out = {
        **metadata,
        **scalarize_mapping(fit),
        **{f"sim_{key}": value for key, value in scalarize_mapping(simulator_diagnostics).items()},
    }
    threshold_rows: list[dict[str, Any]] = []
    profit_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for threshold_idx, threshold in enumerate(thresholds):
        threshold_row = {
            **metadata,
            "threshold_row": int(threshold_idx),
            "threshold_scale": scale,
            **threshold,
        }
        threshold_rows.append(threshold_row)
        if not bool(threshold.get("threshold_valid", False)):
            profit_rows.append(threshold_row)
            continue
        trades = pd.DataFrame(
            trade_real_window(
                trading,
                ticker_a=str(row["ticker_a"]).upper(),
                ticker_b=str(row["ticker_b"]).upper(),
                mu=mu,
                d_plus=float(threshold["d_plus"]),
                d_minus=float(threshold["d_minus"]),
                exit_rule=context["exit_rule"],
            )
        )
        profit_rows.append({**threshold_row, **scalar_trade_profit_summary(trades)})
        if not trades.empty:
            for trade_id, trade in trades.reset_index(drop=True).iterrows():
                trade_rows.append(
                    {
                        **metadata,
                        "threshold_row": int(threshold_idx),
                        "trade_id": int(trade_id) + 1,
                        "optimization_cost_case": threshold["optimization_cost_case"],
                        "gamma_multiplier": threshold["gamma_multiplier"],
                        **scalarize_mapping(trade.to_dict()),
                    }
                )
    return estimate_out, threshold_rows, profit_rows, trade_rows


def run_synthetic_estimation(
    models: str | list[str] | tuple[str, ...],
    observations: int = 120,
    seed: int = 123,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Fit requested models to one synthetic formation spread."""

    spread = synthetic_ou_spread(observations, seed=seed)
    fits = fit_models_for_spread(spread, models=_parse_models(models), seed=seed, **fit_kwargs)
    rows = [
        model_result_row(
            fit,
            dataset="synthetic",
            window_id=1,
            ticker_a="AAA",
            ticker_b="BBB",
            formation_observations=int(len(spread)),
            formation_spread_std=float(np.std(spread, ddof=1)),
        )
        for fit in fits
    ]
    return pd.DataFrame(rows)


def run_window_estimation(
    pair_windows_csv: str | Path,
    models: str | list[str] | tuple[str, ...],
    data_path: str | Path = DEFAULT_LOBSTER_DATA,
    price_col: str = "model_price_close",
    min_observations: int = 100,
    max_windows: int | None = None,
    seed: int = 123,
    n_jobs: int = 1,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Fit requested models for rows in a pair-window CSV."""

    windows = pd.read_csv(pair_windows_csv)
    required = {"ticker_a", "ticker_b", "formation_start", "formation_end"}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"pair window CSV is missing columns: {missing}")
    if max_windows is not None:
        windows = windows.head(int(max_windows))

    price_wide: pd.DataFrame | None = None
    if price_col == "model_price_close" and not windows.empty:
        tickers = sorted(set(windows["ticker_a"].astype(str).str.upper()).union(windows["ticker_b"].astype(str).str.upper()))
        panel = load_lobster_panel(
            data_path=data_path,
            tickers=tickers,
            start_date=str(windows["formation_start"].min()),
            end_date=str(windows["formation_end"].max()),
        )
        if not panel.empty:
            panel = panel[["trade_date", "timestamp_utc", "ticker", "model_price_close"]].copy()
            panel["trade_date"] = panel["trade_date"].astype(str)
            panel["ticker"] = panel["ticker"].astype(str).str.upper()
            price_wide = (
                panel.pivot_table(
                    index=["trade_date", "timestamp_utc"],
                    columns="ticker",
                    values="model_price_close",
                    aggfunc="last",
                )
                .sort_index()
            )

    model_list = _parse_models(models)

    def estimate_row(item: tuple[int, pd.Series]) -> list[dict[str, Any]]:
        row_idx, row = item
        window_id = int(row.get("window_id", row_idx + 1))
        ticker_a = str(row["ticker_a"]).upper()
        ticker_b = str(row["ticker_b"]).upper()
        metadata = {
            "dataset": "lobster",
            "window_id": window_id,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "formation_start": row["formation_start"],
            "formation_end": row["formation_end"],
            "trading_start": row.get("trading_start"),
            "trading_end": row.get("trading_end"),
        }
        if price_wide is not None:
            try:
                pair_prices = price_wide.loc[
                    (slice(str(row["formation_start"]), str(row["formation_end"])), slice(None)),
                    [ticker_a, ticker_b],
                ].dropna()
            except KeyError:
                pair_prices = pd.DataFrame(columns=[ticker_a, ticker_b])
            pair_prices = pair_prices[(pair_prices[ticker_a] > 0.0) & (pair_prices[ticker_b] > 0.0)]
            if pair_prices.empty or len(pair_prices) < min_observations:
                reason = (
                    "No model-ready rows found for this pair and formation period."
                    if pair_prices.empty
                    else f"Too few overlapping observations; need at least {min_observations}."
                )
                return [
                        {
                            **metadata,
                            "model": model,
                            "valid": False,
                            "reason": reason,
                            "observations": int(len(pair_prices)),
                            "spread_definition": "log(M_A(t)/M_A(0)) - log(M_B(t)/M_B(0))",
                            "price_column": price_col,
                        }
                    for model in model_list
                ]
            spread = build_spread(
                pair_prices[ticker_a].to_numpy(dtype=float),
                pair_prices[ticker_b].to_numpy(dtype=float),
            )
        else:
            loaded = load_pair_prices_from_lobster(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                formation_start=str(row["formation_start"]),
                formation_end=str(row["formation_end"]),
                data_path=data_path,
                price_col=price_col,
                min_observations=min_observations,
            )
            if not loaded.get("valid", False):
                return [{**metadata, "model": model, **scalarize_mapping(loaded)} for model in model_list]
            spread = build_spread(loaded["price_a"], loaded["price_b"])

        fits = fit_models_for_spread(spread, models=model_list, seed=seed + row_idx, **fit_kwargs)
        return [
                model_result_row(
                    fit,
                    **metadata,
                    formation_observations=int(len(spread)),
                    formation_spread_std=float(np.std(spread, ddof=1)),
                )
            for fit in fits
        ]

    items = list(windows.reset_index(drop=True).iterrows())
    rows: list[dict[str, Any]] = []
    if int(n_jobs) <= 1 or len(items) <= 1:
        for item in items:
            rows.extend(estimate_row(item))
    else:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            for item_rows in executor.map(estimate_row, items):
                rows.extend(item_rows)
    return pd.DataFrame(rows)


def run_synthetic_backtest(
    models: str | list[str] | tuple[str, ...],
    observations: int = 120,
    trading_steps: int = 120,
    seed: int = 123,
    n_paths: int = 16,
    sim_steps: int = 80,
    cost_cases: str | dict[str, float] = "c0,midquote_5bps",
    gamma_multipliers: str | list[float] | tuple[float, ...] = "0",
    grid_points: int = 5,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 1.5,
    exit_rule: str = "mean",
    **fit_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Fit, simulate, optimise thresholds, and replay synthetic trading windows."""

    formation_spread = synthetic_ou_spread(observations, seed=seed)
    trading_spread = synthetic_ou_spread(trading_steps, seed=seed + 10, rho=0.93, innovation_sigma=0.003)
    trading = synthetic_trading_frame(trading_spread, seed=seed + 20)
    fits = fit_models_for_spread(formation_spread, models=_parse_models(models), seed=seed, **fit_kwargs)
    costs = default_cost_cases(cost_cases) if isinstance(cost_cases, str) else dict(cost_cases)
    gammas = _parse_floats(gamma_multipliers)

    estimate_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for fit_idx, fit in enumerate(fits):
        metadata = {
            "dataset": "synthetic",
            "window_id": 1,
            "ticker_a": "AAA",
            "ticker_b": "BBB",
            "model": fit.get("model"),
        }
        estimate_rows.append(model_result_row(fit, **metadata))
        if not bool(fit.get("valid", False)):
            threshold_rows.append({**metadata, "threshold_valid": False, "threshold_reason": fit.get("reason")})
            continue

        paths = simulate_paths_from_fit(
            fit,
            x0=float(formation_spread[-1]),
            n_paths=n_paths,
            n_steps=sim_steps,
            seed=seed + 100 + fit_idx,
            sim_fft_grid_size=int(fit_kwargs.get("sim_fft_grid_size", DEFAULT_SIM_FFT_GRID_SIZE)),
            nig_du=float(fit_kwargs.get("nig_du", 20.0)),
            cgmy_du=float(fit_kwargs.get("cgmy_du", 0.5)),
        )
        mu = _finite_first(fit.get("u_form"), fit.get("mu"), np.mean(formation_spread))
        scale = model_scale(fit, fallback_spread=formation_spread)
        thresholds = optimize_cost_gamma_grid(
            paths,
            mu=mu,
            sigma=scale,
            cost_cases=costs,
            gamma_multipliers=gammas,
            grid_points=grid_points,
            min_sigma_multiple=min_sigma_multiple,
            max_sigma_multiple=max_sigma_multiple,
            exit_rule=exit_rule,
        )
        for threshold_idx, threshold in enumerate(thresholds):
            threshold_row = {
                **metadata,
                "threshold_row": threshold_idx,
                "threshold_scale": scale,
                **threshold,
            }
            threshold_rows.append(threshold_row)
            if not bool(threshold.get("threshold_valid", False)):
                continue
            trades = trade_real_window(
                trading,
                ticker_a="AAA",
                ticker_b="BBB",
                mu=mu,
                d_plus=float(threshold["d_plus"]),
                d_minus=float(threshold["d_minus"]),
                exit_rule=exit_rule,
            )
            for trade_id, trade in enumerate(trades, start=1):
                trade_rows.append(
                    {
                        **metadata,
                        "threshold_row": threshold_idx,
                        "trade_id": trade_id,
                        "optimization_cost_case": threshold["optimization_cost_case"],
                        "gamma_multiplier": threshold["gamma_multiplier"],
                        **scalarize_mapping(trade),
                    }
                )

    return {
        "estimates": pd.DataFrame(estimate_rows),
        "thresholds": pd.DataFrame(threshold_rows),
        "trades": pd.DataFrame(trade_rows),
    }


def run_window_backtest(
    pair_windows_csv: str | Path,
    data_path: str | Path,
    models: str | list[str] | tuple[str, ...],
    n_pairs: int = 5,
    n_windows: int = 5,
    selection_mode: str = "first_window_pairs",
    seed: int = 123,
    n_paths: int = 16,
    sim_steps: int = 3900,
    cost_cases: str = "c0,midquote_5bps,bidask_median_c,bidask_worst_c",
    gamma_multipliers: str | list[float] | tuple[float, ...] = "0",
    grid_points: int = 5,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 1.5,
    exit_rule: str = "mean",
    min_observations: int = 100,
    **fit_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Fit, simulate, optimise, and replay real selected LOBSTER windows."""

    selected = select_pair_windows(
        pair_windows_csv,
        n_pairs=n_pairs,
        n_windows=n_windows,
        selection_mode=selection_mode,
    )
    if selected.empty:
        raise ValueError("No pair windows selected.")
    tickers = sorted(set(selected["ticker_a"]).union(set(selected["ticker_b"])))
    panel = load_lobster_panel(
        data_path=data_path,
        tickers=tickers,
        start_date=str(selected["formation_start"].min()),
        end_date=str(selected["trading_end"].max()),
    )
    if panel.empty:
        raise ValueError("Filtered LOBSTER panel is empty.")

    model_list = _parse_models(models)
    gammas = _parse_floats(gamma_multipliers)
    estimate_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    profit_rows: list[dict[str, Any]] = []

    for row_idx, row in selected.reset_index(drop=True).iterrows():
        formation, trading = pair_formation_and_trading_frames(panel, row)
        metadata = {
            "dataset": "lobster",
            "window_id": int(row["window_id"]),
            "pair_index": int(row.get("pair_index", row_idx + 1)),
            "ticker_a": str(row["ticker_a"]).upper(),
            "ticker_b": str(row["ticker_b"]).upper(),
            "formation_start": row["formation_start"],
            "formation_end": row["formation_end"],
            "trading_start": row["trading_start"],
            "trading_end": row["trading_end"],
            "formation_rows": int(len(formation)),
            "trading_rows": int(len(trading)),
        }
        if len(formation) < int(min_observations) or trading.empty:
            for model in model_list:
                bad = {
                    **metadata,
                    "model": model,
                    "valid": False,
                    "reason": "insufficient formation or trading rows",
                }
                estimate_rows.append(bad)
                threshold_rows.append({**bad, "threshold_valid": False, "threshold_reason": bad["reason"]})
            continue

        spread = formation["spread"].to_numpy(dtype=float)
        fits = fit_models_for_spread(spread, models=model_list, seed=seed + row_idx, **fit_kwargs)
        window_costs = _window_cost_cases(cost_cases, formation)
        for fit_idx, fit in enumerate(fits):
            model_meta = {**metadata, "model": fit.get("model")}
            estimate_rows.append(model_result_row(fit, **model_meta))
            if not bool(fit.get("valid", False)):
                reason = fit.get("reason", "invalid fit")
                threshold_rows.append({**model_meta, "threshold_valid": False, "threshold_reason": reason})
                profit_rows.append({**model_meta, "threshold_valid": False, "threshold_reason": reason})
                continue

            try:
                paths = simulate_paths_from_fit(
                    fit,
                    x0=float(formation["spread"].iloc[-1]),
                    n_paths=n_paths,
                    n_steps=sim_steps,
                    seed=seed + 1000 + row_idx * 17 + fit_idx,
                    sim_fft_grid_size=int(fit_kwargs.get("sim_fft_grid_size", DEFAULT_SIM_FFT_GRID_SIZE)),
                    nig_du=float(fit_kwargs.get("nig_du", 20.0)),
                    cgmy_du=float(fit_kwargs.get("cgmy_du", 0.5)),
                )
                mu = _finite_first(fit.get("u_form"), fit.get("mu"), np.mean(spread))
                scale = model_scale(fit, fallback_spread=spread)
                thresholds = optimize_cost_gamma_grid(
                    paths,
                    mu=mu,
                    sigma=scale,
                    cost_cases=window_costs,
                    gamma_multipliers=gammas,
                    grid_points=grid_points,
                    min_sigma_multiple=min_sigma_multiple,
                    max_sigma_multiple=max_sigma_multiple,
                    exit_rule=exit_rule,
                )
            except Exception as exc:
                threshold_rows.append({**model_meta, "threshold_valid": False, "threshold_reason": repr(exc)})
                profit_rows.append({**model_meta, "threshold_valid": False, "threshold_reason": repr(exc)})
                continue

            for threshold_idx, threshold in enumerate(thresholds):
                threshold_row = {
                    **model_meta,
                    "threshold_row": threshold_idx,
                    "threshold_scale": scale,
                    **threshold,
                }
                threshold_rows.append(threshold_row)
                if not bool(threshold.get("threshold_valid", False)):
                    profit_rows.append(threshold_row)
                    continue
                trades = pd.DataFrame(
                    trade_real_window(
                        trading,
                        ticker_a=str(row["ticker_a"]).upper(),
                        ticker_b=str(row["ticker_b"]).upper(),
                        mu=mu,
                        d_plus=float(threshold["d_plus"]),
                        d_minus=float(threshold["d_minus"]),
                        exit_rule=exit_rule,
                    )
                )
                trade_summary = scalar_trade_profit_summary(trades)
                profit_rows.append({**threshold_row, **trade_summary})
                if not trades.empty:
                    for trade_id, trade in trades.reset_index(drop=True).iterrows():
                        trade_rows.append(
                            {
                                **model_meta,
                                "threshold_row": threshold_idx,
                                "trade_id": int(trade_id) + 1,
                                "optimization_cost_case": threshold["optimization_cost_case"],
                                "gamma_multiplier": threshold["gamma_multiplier"],
                                **scalarize_mapping(trade.to_dict()),
                            }
                        )

    return {
        "selected_windows": selected,
        "estimates": pd.DataFrame(estimate_rows),
        "thresholds": pd.DataFrame(threshold_rows),
        "profits": pd.DataFrame(profit_rows),
        "trades": pd.DataFrame(trade_rows),
    }


def run_window_backtest_from_estimates(
    estimates_csv: str | Path,
    data_path: str | Path,
    seed: int = 123,
    n_paths: int = 16,
    sim_steps: int = 3900,
    cost_cases: str = "c0,midquote_5bps,bidask_median_c,bidask_worst_c",
    gamma_multipliers: str | list[float] | tuple[float, ...] = "0",
    grid_points: int = 5,
    min_sigma_multiple: float = 0.25,
    max_sigma_multiple: float = 1.5,
    exit_rule: str = "mean",
    min_observations: int = 100,
    n_jobs: int = 1,
    max_rows: int | None = None,
    simulation_method: str | None = None,
    parallel_backend: str = "thread",
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = True,
    **sim_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Simulate thresholds and replay real windows from saved model estimates."""

    estimates = pd.read_csv(estimates_csv)
    required = {"window_id", "ticker_a", "ticker_b", "formation_start", "formation_end", "trading_start", "trading_end"}
    missing = sorted(required - set(estimates.columns))
    if missing:
        raise ValueError(f"estimates CSV is missing columns: {missing}")
    if "valid" in estimates.columns:
        valid_mask = estimates["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        estimates = estimates[valid_mask].copy()
    if max_rows is not None:
        estimates = estimates.head(int(max_rows)).copy()
    if estimates.empty:
        raise ValueError("No valid estimate rows to backtest.")

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    existing_outputs: dict[str, pd.DataFrame] = {}
    completed_keys: set[tuple[int, str, str, str]] = set()
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if resume and (checkpoint_path / "estimates_checkpoint.csv").exists():
            existing_estimates = pd.read_csv(checkpoint_path / "estimates_checkpoint.csv")
            existing_outputs["estimates"] = existing_estimates
            if not existing_estimates.empty:
                for done in existing_estimates.itertuples(index=False):
                    completed_keys.add(
                        (
                            int(getattr(done, "window_id")),
                            str(getattr(done, "ticker_a")).upper(),
                            str(getattr(done, "ticker_b")).upper(),
                            str(getattr(done, "model", "unknown")),
                        )
                    )
            for name in ("thresholds", "profits", "trades"):
                path = checkpoint_path / f"{name}_checkpoint.csv"
                if path.exists():
                    existing_outputs[name] = pd.read_csv(path)

    if completed_keys:
        keys = list(
            zip(
                estimates["window_id"].astype(int),
                estimates["ticker_a"].astype(str).str.upper(),
                estimates["ticker_b"].astype(str).str.upper(),
                estimates.get("model", pd.Series(["unknown"] * len(estimates))).astype(str),
            )
        )
        estimates = estimates[[key not in completed_keys for key in keys]].copy()

    if estimates.empty:
        return {
            "estimates": existing_outputs.get("estimates", pd.DataFrame()),
            "thresholds": existing_outputs.get("thresholds", pd.DataFrame()),
            "profits": existing_outputs.get("profits", pd.DataFrame()),
            "trades": existing_outputs.get("trades", pd.DataFrame()),
        }

    tickers = sorted(set(estimates["ticker_a"].astype(str).str.upper()).union(estimates["ticker_b"].astype(str).str.upper()))
    panel = load_lobster_panel(
        data_path=data_path,
        tickers=tickers,
        start_date=str(estimates["formation_start"].min()),
        end_date=str(estimates["trading_end"].max()),
    )
    if panel.empty:
        raise ValueError("Filtered LOBSTER panel is empty.")

    gammas = _parse_floats(gamma_multipliers)

    def backtest_row(item: tuple[int, pd.Series]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        row_idx, estimate_row = item
        row = estimate_row.copy()
        row["ticker_a"] = str(row["ticker_a"]).upper()
        row["ticker_b"] = str(row["ticker_b"]).upper()
        formation, trading = pair_formation_and_trading_frames(panel, row)
        metadata = {
            "dataset": row.get("dataset", "lobster"),
            "window_id": int(row["window_id"]),
            "pair_index": int(row.get("pair_index", row_idx + 1)) if pd.notna(row.get("pair_index", np.nan)) else int(row_idx + 1),
            "ticker_a": str(row["ticker_a"]),
            "ticker_b": str(row["ticker_b"]),
            "formation_start": row["formation_start"],
            "formation_end": row["formation_end"],
            "trading_start": row["trading_start"],
            "trading_end": row["trading_end"],
            "model": row.get("model", "unknown"),
            "formation_rows": int(len(formation)),
            "trading_rows": int(len(trading)),
        }
        if len(formation) < int(min_observations) or trading.empty:
            reason = "insufficient formation or trading rows"
            bad = {**metadata, "threshold_valid": False, "threshold_reason": reason}
            return bad, [bad], [bad], []

        fit = row.to_dict()
        if simulation_method:
            fit["simulation_method"] = simulation_method
        spread = formation["spread"].to_numpy(dtype=float)
        window_costs = _window_cost_cases(cost_cases, formation)
        try:
            paths = simulate_paths_from_fit(
                fit,
                x0=float(formation["spread"].iloc[-1]),
                n_paths=n_paths,
                n_steps=sim_steps,
                seed=seed + 1000 + row_idx * 17,
                sim_fft_grid_size=int(sim_kwargs.get("sim_fft_grid_size", DEFAULT_SIM_FFT_GRID_SIZE)),
                nig_du=float(sim_kwargs.get("nig_du", 20.0)),
                cgmy_du=float(sim_kwargs.get("cgmy_du", 0.5)),
            )
            simulator_diagnostics = getattr(simulate_paths_from_fit, "last_simulator_diagnostics", {})
            mu = _finite_first(fit.get("u_form"), fit.get("mu"), np.mean(spread))
            scale = model_scale(fit, fallback_spread=spread)
            thresholds = optimize_cost_gamma_grid(
                paths,
                mu=mu,
                sigma=scale,
                cost_cases=window_costs,
                gamma_multipliers=gammas,
                grid_points=grid_points,
                min_sigma_multiple=min_sigma_multiple,
                max_sigma_multiple=max_sigma_multiple,
                exit_rule=exit_rule,
            )
        except Exception as exc:
            reason = repr(exc)
            bad = {**metadata, "threshold_valid": False, "threshold_reason": reason}
            return {**bad, **scalarize_mapping(fit)}, [bad], [bad], []

        estimate_out = {
            **metadata,
            **scalarize_mapping(fit),
            **{f"sim_{key}": value for key, value in scalarize_mapping(simulator_diagnostics).items()},
        }
        threshold_rows: list[dict[str, Any]] = []
        profit_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        for threshold_idx, threshold in enumerate(thresholds):
            threshold_row = {
                **metadata,
                "threshold_row": int(threshold_idx),
                "threshold_scale": scale,
                **threshold,
            }
            threshold_rows.append(threshold_row)
            if not bool(threshold.get("threshold_valid", False)):
                profit_rows.append(threshold_row)
                continue
            trades = pd.DataFrame(
                trade_real_window(
                    trading,
                    ticker_a=str(row["ticker_a"]),
                    ticker_b=str(row["ticker_b"]),
                    mu=mu,
                    d_plus=float(threshold["d_plus"]),
                    d_minus=float(threshold["d_minus"]),
                    exit_rule=exit_rule,
                )
            )
            profit_rows.append({**threshold_row, **scalar_trade_profit_summary(trades)})
            if not trades.empty:
                for trade_id, trade in trades.reset_index(drop=True).iterrows():
                    trade_rows.append(
                        {
                            **metadata,
                            "threshold_row": int(threshold_idx),
                            "trade_id": int(trade_id) + 1,
                            "optimization_cost_case": threshold["optimization_cost_case"],
                            "gamma_multiplier": threshold["gamma_multiplier"],
                            **scalarize_mapping(trade.to_dict()),
                        }
                    )
        return estimate_out, threshold_rows, profit_rows, trade_rows

    items = list(estimates.reset_index(drop=True).iterrows())
    estimate_rows: list[dict[str, Any]] = existing_outputs.get("estimates", pd.DataFrame()).to_dict("records")
    threshold_rows: list[dict[str, Any]] = existing_outputs.get("thresholds", pd.DataFrame()).to_dict("records")
    profit_rows: list[dict[str, Any]] = existing_outputs.get("profits", pd.DataFrame()).to_dict("records")
    trade_rows: list[dict[str, Any]] = existing_outputs.get("trades", pd.DataFrame()).to_dict("records")

    pending_estimate_rows: list[dict[str, Any]] = []
    pending_threshold_rows: list[dict[str, Any]] = []
    pending_profit_rows: list[dict[str, Any]] = []
    pending_trade_rows: list[dict[str, Any]] = []

    def append_checkpoint(name: str, records: list[dict[str, Any]]) -> None:
        if checkpoint_path is None or not records:
            return
        path = checkpoint_path / f"{name}_checkpoint.csv"
        new_frame = pd.DataFrame(records)
        if path.exists():
            try:
                existing_frame = pd.read_csv(path)
                frame = pd.concat([existing_frame, new_frame], ignore_index=True, sort=False)
            except pd.errors.ParserError:
                backup = path.with_suffix(path.suffix + ".malformed_backup")
                path.replace(backup)
                frame = new_frame
        else:
            frame = new_frame
        frame.to_csv(path, index=False)

    def record_result(result: tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]) -> None:
        estimate_out, row_thresholds, row_profits, row_trades = result
        if estimate_out is not None:
            estimate_rows.append(estimate_out)
            pending_estimate_rows.append(estimate_out)
        threshold_rows.extend(row_thresholds)
        profit_rows.extend(row_profits)
        trade_rows.extend(row_trades)
        pending_threshold_rows.extend(row_thresholds)
        pending_profit_rows.extend(row_profits)
        pending_trade_rows.extend(row_trades)

    def flush_checkpoint(force: bool = False) -> None:
        if checkpoint_path is None:
            return
        if not force and len(pending_estimate_rows) < max(1, int(checkpoint_every)):
            return
        append_checkpoint("estimates", pending_estimate_rows)
        append_checkpoint("thresholds", pending_threshold_rows)
        append_checkpoint("profits", pending_profit_rows)
        append_checkpoint("trades", pending_trade_rows)
        pending_estimate_rows.clear()
        pending_threshold_rows.clear()
        pending_profit_rows.clear()
        pending_trade_rows.clear()

    if parallel_backend not in {"thread", "process"}:
        raise ValueError("parallel_backend must be 'thread' or 'process'.")
    completed = len(completed_keys)
    total = completed + len(items)
    if int(n_jobs) <= 1 or len(items) <= 1:
        for item in items:
            record_result(backtest_row(item))
            completed += 1
            flush_checkpoint()
            print(f"backtest progress: {completed}/{total}", flush=True)
    elif parallel_backend == "process":
        process_context = {
            "panel": panel,
            "min_observations": int(min_observations),
            "simulation_method": simulation_method,
            "cost_cases": cost_cases,
            "gammas": gammas,
            "n_paths": int(n_paths),
            "sim_steps": int(sim_steps),
            "seed": int(seed),
            "grid_points": int(grid_points),
            "min_sigma_multiple": float(min_sigma_multiple),
            "max_sigma_multiple": float(max_sigma_multiple),
            "exit_rule": exit_rule,
            "sim_fft_grid_size": int(sim_kwargs.get("sim_fft_grid_size", DEFAULT_SIM_FFT_GRID_SIZE)),
            "nig_du": float(sim_kwargs.get("nig_du", 20.0)),
            "cgmy_du": float(sim_kwargs.get("cgmy_du", 0.5)),
        }
        process_items = [(row_idx, row.to_dict()) for row_idx, row in items]
        with ProcessPoolExecutor(
            max_workers=int(n_jobs),
            initializer=_init_saved_backtest_worker,
            initargs=(process_context,),
        ) as executor:
            futures = [executor.submit(_saved_backtest_row_worker, item) for item in process_items]
            for future in as_completed(futures):
                record_result(future.result())
                completed += 1
                flush_checkpoint()
                print(f"backtest progress: {completed}/{total}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            futures = [executor.submit(backtest_row, item) for item in items]
            for future in as_completed(futures):
                record_result(future.result())
                completed += 1
                flush_checkpoint()
                print(f"backtest progress: {completed}/{total}", flush=True)
    flush_checkpoint(force=True)

    return {
        "estimates": pd.DataFrame(estimate_rows),
        "thresholds": pd.DataFrame(threshold_rows),
        "profits": pd.DataFrame(profit_rows),
        "trades": pd.DataFrame(trade_rows),
    }


__all__ = [
    "default_cost_cases",
    "run_synthetic_backtest",
    "run_synthetic_estimation",
    "run_window_backtest",
    "run_window_backtest_from_estimates",
    "run_window_estimation",
]
