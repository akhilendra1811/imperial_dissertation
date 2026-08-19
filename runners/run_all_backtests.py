#!/usr/bin/env python3
"""Run final pair-window backtests for Gaussian baselines and Levy OU models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from levy_ou.backtesting.basic_baseline import fit_basic_baseline
from levy_ou.backtesting.optimal_gaussian import endres_5bps_round_trip_c
from levy_ou.backtesting.trade_replay import trade_real_window
from levy_ou.backtesting.zeng_lee_gaussian import solve_zeng_lee_gaussian_ou_from_fit
from levy_ou.estimators.gaussian_ou import fit_brownian_ou_from_spread
from levy_ou.experiments.outputs import scalarize_mapping
from levy_ou.experiments.pipeline import run_window_backtest_from_estimates, run_window_estimation
from levy_ou.experiments.real_data import (
    formation_cost_cases,
    load_lobster_panel,
    pair_formation_and_trading_frames,
    scalar_trade_profit_summary,
)


LEVY_MODELS = {"symmetric_bg", "nig", "cgmy"}
DEFAULT_COST_CASES = "c0,midquote_5bps,bidask_median_c,bidask_worst_c"
DEFAULT_GAMMAS = "0,0.5,1,2,4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sector", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--selection-scope", choices=["gaussian_top10", "adf_capped10"], default="gaussian_top10")
    parser.add_argument("--pair-windows-csv")
    parser.add_argument("--estimates-csv", help="Optional existing BG/NIG/CGMY model estimates CSV. If supplied, Levy estimation is skipped.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--models", default="gaussian_fixed_sigma_eq,formation_mean_std,zeng_lee_gaussian_conventional,symmetric_bg,nig,cgmy")
    parser.add_argument("--sigma-multiple", type=float, default=1.5)
    parser.add_argument("--cost-cases", default=DEFAULT_COST_CASES)
    parser.add_argument("--gamma-multipliers", default=DEFAULT_GAMMAS)
    parser.add_argument("--n-paths", type=int, default=1000)
    parser.add_argument("--sim-steps", type=int, default=3900)
    parser.add_argument("--grid-points", type=int, default=10)
    parser.add_argument("--min-sigma-multiple", type=float, default=0.25)
    parser.add_argument("--max-sigma-multiple", type=float, default=3.0)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fft-grid-size", type=int, default=512)
    parser.add_argument("--sim-fft-grid-size", type=int, default=2**15)
    parser.add_argument("--truncation-l", type=float, default=6.0)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--max-optimizer-starts", type=int, default=1)
    parser.add_argument("--max-candidate-starts-to-score", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--parallel-backend", choices=["thread", "process"], default="thread")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def case_name(sector: str, year: str) -> str:
    return f"{str(sector).strip().lower()}_{str(year).strip()}"


def parse_list(values: str) -> list[str]:
    return [item.strip().lower() for item in str(values).split(",") if item.strip()]


def parse_floats(values: str) -> list[float]:
    return [float(item.strip()) for item in str(values).split(",") if item.strip()]


def default_pair_windows_path(case: str, selection_scope: str) -> Path:
    candidates = [
        Path("data/selections") / f"{case}_{selection_scope}_pair_windows.csv",
        Path("data/selections") / f"{case}_{selection_scope}_selected_pair_windows.csv",
        Path("data/selections") / f"{case}_{selection_scope}_gaussian_ranked_pair_windows.csv",
    ]
    if selection_scope == "gaussian_top10":
        candidates.extend(
            [
                Path("data/selections") / f"{case}_gaussian_top10_pair_windows.csv",
                Path("data/selections") / f"{case}_gaussian_top10_selected_pair_windows.csv",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def cost_cases(names: str, formation: pd.DataFrame) -> dict[str, float]:
    formation_costs = formation_cost_cases(formation)
    out: dict[str, float] = {}
    for raw in [item.strip() for item in str(names).split(",") if item.strip()]:
        name = raw.lower()
        if name in {"c0", "zero", "none"}:
            out["c0"] = 0.0
        elif name in {"midquote_5bps", "endres_5bps", "5bps"}:
            out["midquote_5bps"] = endres_5bps_round_trip_c()
        elif name in {"bidask_median_c", "bidask_worst_c"}:
            out[name] = float(formation_costs.get(name, np.nan))
        else:
            out[raw] = float(raw)
    return out


def write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


def run_deterministic_gaussian_backtests(args: argparse.Namespace, pair_windows: Path, output_dir: Path) -> None:
    requested = set(parse_list(args.models))
    needed = {"gaussian_fixed_sigma_eq", "formation_mean_std", "zeng_lee_gaussian_conventional"} & requested
    if not needed:
        return

    windows = pd.read_csv(pair_windows)
    if args.max_rows is not None:
        windows = windows.head(int(args.max_rows)).copy()
    tickers = sorted(set(windows["ticker_a"].astype(str).str.upper()).union(windows["ticker_b"].astype(str).str.upper()))
    panel = load_lobster_panel(
        data_path=args.data_path,
        tickers=tickers,
        start_date=str(windows["formation_start"].min()),
        end_date=str(windows["trading_end"].max()),
    )
    gammas = parse_floats(args.gamma_multipliers)
    estimate_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    profit_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for row_idx, row in windows.reset_index(drop=True).iterrows():
        row = row.copy()
        row["ticker_a"] = str(row["ticker_a"]).upper()
        row["ticker_b"] = str(row["ticker_b"]).upper()
        formation, trading = pair_formation_and_trading_frames(panel, row)
        metadata = {
            "dataset": case_name(args.sector, args.year),
            "window_id": int(row["window_id"]),
            "pair_index": int(row.get("pair_index", row_idx + 1)) if pd.notna(row.get("pair_index", np.nan)) else int(row_idx + 1),
            "ticker_a": row["ticker_a"],
            "ticker_b": row["ticker_b"],
            "formation_start": row["formation_start"],
            "formation_end": row["formation_end"],
            "trading_start": row["trading_start"],
            "trading_end": row["trading_end"],
            "formation_rows": int(len(formation)),
            "trading_rows": int(len(trading)),
        }
        if len(formation) < int(args.min_observations) or trading.empty:
            for model in sorted(needed):
                bad = {**metadata, "model": model, "threshold_valid": False, "threshold_reason": "insufficient formation or trading rows"}
                estimate_rows.append(bad)
                threshold_rows.append(bad)
                profit_rows.append(bad)
            continue

        spread = formation["spread"].to_numpy(dtype=float)
        gaussian_fit = fit_brownian_ou_from_spread(spread)
        basic_fit = fit_basic_baseline(spread)
        row_costs = cost_cases(args.cost_cases, formation)

        for cost_name, c_value in row_costs.items():
            for gamma in gammas:
                common = {**metadata, "optimization_cost_case": cost_name, "optimization_cost": c_value, "gamma_multiplier": gamma}
                if "gaussian_fixed_sigma_eq" in needed:
                    model = "gaussian_fixed_sigma_eq"
                    mu = float(gaussian_fit["mu"])
                    distance = float(args.sigma_multiple) * float(gaussian_fit["sigma_eq"])
                    threshold = {
                        **common,
                        "model": model,
                        "threshold_row": 0,
                        "threshold_valid": bool(gaussian_fit.get("valid", False)),
                        "threshold_reason": "",
                        "mu": mu,
                        "d_plus": distance,
                        "d_minus": distance,
                        "sigma_multiple": float(args.sigma_multiple),
                        "threshold_scale": float(gaussian_fit["sigma_eq"]),
                    }
                    trades = pd.DataFrame(trade_real_window(trading, ticker_a=row["ticker_a"], ticker_b=row["ticker_b"], mu=mu, d_plus=distance, d_minus=distance, exit_rule="mean"))
                    estimate_rows.append({**common, "model": model, **scalarize_mapping(gaussian_fit)})
                    threshold_rows.append(threshold)
                    profit_rows.append({**threshold, **scalar_trade_profit_summary(trades)})
                    trade_rows.extend(trade_records(trades, common, model))

                if "formation_mean_std" in needed:
                    model = "formation_mean_std"
                    mu = float(basic_fit.mean)
                    distance = float(args.sigma_multiple) * float(basic_fit.std)
                    threshold = {
                        **common,
                        "model": model,
                        "threshold_row": 0,
                        "threshold_valid": True,
                        "threshold_reason": "",
                        "mu": mu,
                        "d_plus": distance,
                        "d_minus": distance,
                        "sigma_multiple": float(args.sigma_multiple),
                        "threshold_scale": float(basic_fit.std),
                    }
                    trades = pd.DataFrame(trade_real_window(trading, ticker_a=row["ticker_a"], ticker_b=row["ticker_b"], mu=mu, d_plus=distance, d_minus=distance, exit_rule="mean"))
                    estimate_rows.append({**common, "model": model, "mu": mu, "sigma_eq": float(basic_fit.std), "observations": int(basic_fit.observations), "valid": True})
                    threshold_rows.append(threshold)
                    profit_rows.append({**threshold, **scalar_trade_profit_summary(trades)})
                    trade_rows.extend(trade_records(trades, common, model))

                if "zeng_lee_gaussian_conventional" in needed:
                    model = "zeng_lee_gaussian_conventional"
                    threshold_fit = solve_zeng_lee_gaussian_ou_from_fit(gaussian_fit, c=c_value, rule="conventional")
                    threshold = {
                        **common,
                        "model": model,
                        "threshold_row": 0,
                        "zeng_lee_rule": "conventional",
                        "exit_rule": "mean",
                        **scalarize_mapping(threshold_fit),
                    }
                    trades = pd.DataFrame()
                    if bool(threshold_fit.get("threshold_valid", False)):
                        trades = pd.DataFrame(
                            trade_real_window(
                                trading,
                                ticker_a=row["ticker_a"],
                                ticker_b=row["ticker_b"],
                                mu=float(threshold_fit["mu"]),
                                d_plus=float(threshold_fit["b_star"]),
                                d_minus=float(threshold_fit["b_star"]),
                                exit_rule="mean",
                            )
                        )
                    estimate_rows.append({**common, "model": model, **scalarize_mapping(gaussian_fit)})
                    threshold_rows.append(threshold)
                    profit_rows.append({**threshold, **scalar_trade_profit_summary(trades)})
                    trade_rows.extend(trade_records(trades, common, model))

    write_outputs(
        output_dir,
        {
            "estimates": pd.DataFrame(estimate_rows),
            "thresholds": pd.DataFrame(threshold_rows),
            "profits": pd.DataFrame(profit_rows),
            "trades": pd.DataFrame(trade_rows),
        },
        {"runner": "deterministic_gaussian_backtests", **vars(args)},
    )


def trade_records(trades: pd.DataFrame, metadata: dict[str, Any], model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return rows
    for trade_id, trade in trades.reset_index(drop=True).iterrows():
        rows.append(
            {
                **metadata,
                "model": model,
                "threshold_row": 0,
                "trade_id": int(trade_id) + 1,
                **scalarize_mapping(trade.to_dict()),
            }
        )
    return rows


def run_levy_backtests(args: argparse.Namespace, pair_windows: Path, output_dir: Path) -> None:
    requested = set(parse_list(args.models))
    levy_models = sorted(LEVY_MODELS & requested)
    if not levy_models:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    estimates_csv = output_dir / "model_estimates.csv"
    if args.estimates_csv:
        source_estimates = Path(args.estimates_csv)
        if not source_estimates.exists():
            raise SystemExit(f"estimates CSV not found: {source_estimates}")
        estimates = pd.read_csv(source_estimates, low_memory=False)
        if "model" not in estimates.columns:
            raise SystemExit(f"estimates CSV is missing required column: model")
        estimates = estimates[estimates["model"].astype(str).str.lower().isin(levy_models)].copy()
        if args.max_rows is not None:
            estimates = estimates.head(int(args.max_rows)).copy()
        if estimates.empty:
            raise SystemExit(f"no requested Levy model rows found in estimates CSV: {source_estimates}")
    else:
        estimates = run_window_estimation(
            pair_windows_csv=pair_windows,
            models=",".join(levy_models),
            data_path=args.data_path,
            min_observations=args.min_observations,
            max_windows=None,
            seed=args.seed,
            n_jobs=args.n_jobs,
            fft_grid_size=args.fft_grid_size,
            truncation_l=args.truncation_l,
            maxiter=args.maxiter,
            max_optimizer_starts=args.max_optimizer_starts,
            max_candidate_starts_to_score=args.max_candidate_starts_to_score,
        )
        if args.max_rows is not None:
            estimates = estimates.head(int(args.max_rows)).copy()
    estimates.to_csv(estimates_csv, index=False)
    outputs = run_window_backtest_from_estimates(
        estimates_csv=estimates_csv,
        data_path=args.data_path,
        seed=args.seed,
        n_paths=args.n_paths,
        sim_steps=args.sim_steps,
        cost_cases=args.cost_cases,
        gamma_multipliers=args.gamma_multipliers,
        grid_points=args.grid_points,
        min_sigma_multiple=args.min_sigma_multiple,
        max_sigma_multiple=args.max_sigma_multiple,
        exit_rule="mean",
        min_observations=args.min_observations,
        n_jobs=args.n_jobs,
        max_rows=None,
        parallel_backend=args.parallel_backend,
        checkpoint_dir=output_dir / "checkpoints",
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        sim_fft_grid_size=args.sim_fft_grid_size,
    )
    write_outputs(
        output_dir,
        outputs,
        {
            "runner": "levy_simulation_backtests",
            "models": levy_models,
            "estimation_source": str(Path(args.estimates_csv).resolve()) if args.estimates_csv else "estimated_by_runner",
            **vars(args),
        },
    )


def main() -> int:
    args = parse_args()
    case = case_name(args.sector, args.year)
    pair_windows = Path(args.pair_windows_csv) if args.pair_windows_csv else default_pair_windows_path(case, args.selection_scope)
    if not pair_windows.exists():
        raise SystemExit(f"pair-window CSV not found: {pair_windows}")

    case_dir = Path(args.outputs_root) / case
    backtest_root = case_dir / ("adf_capped10/backtests" if args.selection_scope == "adf_capped10" else "backtests")
    prefix = f"{case}_{args.selection_scope}"

    run_deterministic_gaussian_backtests(args, pair_windows, backtest_root / f"{prefix}_gaussian_fixed_sigma_eq_formation_mean_std_zeng_lee_gaussian_conventional_backtest")
    run_levy_backtests(args, pair_windows, backtest_root / f"{prefix}_levy_backtest")
    print(f"wrote final backtests under {backtest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
