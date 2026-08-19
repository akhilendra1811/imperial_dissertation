#!/usr/bin/env python3
"""Run ADF stationarity filters on pair-window formation spreads."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from levy_ou.experiments.real_data import PANEL_COLUMNS, load_lobster_panel
from levy_ou.stationarity import build_pair_windows_from_panel_dates, run_adf_for_pair_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset label used in output rows and folders.")
    parser.add_argument("--data-path", required=True, help="Processed model-ready LOBSTER CSV/CSV.GZ.")
    parser.add_argument("--output-dir", default="outputs/stationarity")
    parser.add_argument("--pair-windows-csv", help="Optional existing pair-window selection CSV.")
    parser.add_argument("--formation-days", type=int, default=30)
    parser.add_argument("--trading-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--regression", choices=["c", "ct", "ctt", "n"], default="c")
    parser.add_argument("--autolag", choices=["AIC", "BIC", "t-stat"], default="AIC")
    parser.add_argument("--maxlag", type=int)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every-windows", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-passing", action="store_true")
    return parser.parse_args()


def _load_pair_windows(args: argparse.Namespace) -> pd.DataFrame:
    if args.pair_windows_csv:
        windows = pd.read_csv(args.pair_windows_csv)
    else:
        meta_cols = ["trade_date", "ticker", "model_ready_price"]
        meta = pd.read_csv(args.data_path, usecols=meta_cols, compression="infer")
        meta["ticker"] = meta["ticker"].astype(str).str.upper()
        ready = meta["model_ready_price"].astype(str).str.lower().eq("true")
        windows = build_pair_windows_from_panel_dates(
            meta.loc[ready, ["trade_date", "ticker"]],
            formation_days=args.formation_days,
            trading_days=args.trading_days,
            step_days=args.step_days,
        )

    required = {"window_id", "ticker_a", "ticker_b", "formation_start", "formation_end", "trading_start", "trading_end"}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"pair-window data is missing columns: {missing}")

    windows = windows.copy()
    windows["ticker_a"] = windows["ticker_a"].astype(str).str.upper()
    windows["ticker_b"] = windows["ticker_b"].astype(str).str.upper()
    windows = windows.sort_values(["window_id", "ticker_a", "ticker_b"]).reset_index(drop=True)
    if args.max_windows is not None:
        keep_windows = sorted(pd.unique(windows["window_id"]))[: int(args.max_windows)]
        windows = windows[windows["window_id"].isin(keep_windows)].copy()
    if args.max_pairs is not None:
        pairs = windows[["ticker_a", "ticker_b"]].drop_duplicates().head(int(args.max_pairs))
        pair_keys = {(row.ticker_a, row.ticker_b) for row in pairs.itertuples(index=False)}
        windows = windows[windows[["ticker_a", "ticker_b"]].apply(tuple, axis=1).isin(pair_keys)].copy()
    return windows.reset_index(drop=True)


def _empty_adf_result(args: argparse.Namespace, reason: str) -> dict[str, object]:
    return {
        "spread_observations": 0,
        "spread_method": "normalized_log",
        "adf_valid": False,
        "adf_reason": reason,
        "adf_observations": 0,
        "adf_significance": float(args.significance),
        "adf_regression": args.regression,
        "adf_autolag": args.autolag,
        "adf_maxlag": args.maxlag,
        "adf_reject_unit_root": False,
        "adf_pass": False,
    }


def _run_one_adf(row_dict: dict[str, object], price_a: pd.Series, price_b: pd.Series, args: argparse.Namespace) -> dict[str, object]:
    result = run_adf_for_pair_window(
        price_a,
        price_b,
        significance=args.significance,
        regression=args.regression,
        autolag=args.autolag,
        maxlag=args.maxlag,
        min_observations=args.min_observations,
    )
    return {"dataset": args.dataset, **row_dict, **result}


def _append_checkpoint(rows: list[dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _completed_window_ids(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    completed = pd.read_csv(csv_path, usecols=["window_id"])
    return {int(value) for value in completed["window_id"].dropna().unique()}


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def run_adf_filters(args: argparse.Namespace, csv_path: Path | None = None) -> pd.DataFrame:
    windows = _load_pair_windows(args)
    if windows.empty:
        return pd.DataFrame()
    if csv_path is not None and args.resume:
        done = _completed_window_ids(csv_path)
        if done:
            windows = windows[~windows["window_id"].astype(int).isin(done)].copy()
            print(f"resuming {args.dataset}: skipping {len(done)} completed windows")
    if windows.empty:
        return pd.DataFrame()

    tickers = sorted(set(windows["ticker_a"]).union(set(windows["ticker_b"])))
    start_date = str(windows["formation_start"].min())
    end_date = str(windows["formation_end"].max())
    panel = load_lobster_panel(
        data_path=args.data_path,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        chunksize=args.chunksize,
    )
    if panel.empty:
        raise ValueError("no model-ready panel rows loaded for requested pair-windows")
    missing_cols = sorted(set(PANEL_COLUMNS) - set(panel.columns))
    if missing_cols:
        raise ValueError(f"loaded panel is missing columns: {missing_cols}")

    all_rows: list[dict[str, object]] = []
    pending_rows: list[dict[str, object]] = []
    completed_groups = 0
    total_groups = int(windows[["window_id", "formation_start", "formation_end"]].drop_duplicates().shape[0])
    workers = max(1, int(args.workers))
    checkpoint_every = max(1, int(args.checkpoint_every_windows))

    for (_window_id, formation_start, formation_end), group in windows.groupby(
        ["window_id", "formation_start", "formation_end"],
        sort=True,
    ):
        tasks: list[tuple[dict[str, object], pd.Series, pd.Series]] = []
        frame = panel[
            panel["trade_date"].astype(str).between(str(formation_start), str(formation_end))
            & panel["ticker"].isin(sorted(set(group["ticker_a"]).union(set(group["ticker_b"]))))
        ]
        prices = frame.pivot_table(
            index="timestamp_utc",
            columns="ticker",
            values="model_price_close",
            aggfunc="last",
        ).sort_index()

        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            ticker_a = str(row_dict["ticker_a"]).upper()
            ticker_b = str(row_dict["ticker_b"]).upper()
            if prices.empty or ticker_a not in prices.columns or ticker_b not in prices.columns:
                pending_rows.append({"dataset": args.dataset, **row_dict, **_empty_adf_result(args, "empty aligned formation frame")})
            else:
                tasks.append((row_dict, prices[ticker_a], prices[ticker_b]))

        if workers == 1:
            for row_dict, price_a, price_b in tasks:
                pending_rows.append(_run_one_adf(row_dict, price_a, price_b, args))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_run_one_adf, row_dict, price_a, price_b, args) for row_dict, price_a, price_b in tasks]
                for future in futures:
                    pending_rows.append(future.result())

        completed_groups += 1
        if completed_groups % checkpoint_every == 0:
            if csv_path is not None:
                _append_checkpoint(pending_rows, csv_path)
                print(f"{args.dataset}: checkpointed {completed_groups}/{total_groups} windows in this run")
                pending_rows.clear()
            else:
                all_rows.extend(pending_rows)
                pending_rows.clear()

    if csv_path is not None:
        _append_checkpoint(pending_rows, csv_path)
        pending_rows.clear()
        return pd.read_csv(csv_path)
    all_rows.extend(pending_rows)
    return pd.DataFrame(all_rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) / str(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "adf_all_pair_windows.csv"
    if csv_path.exists() and not args.resume:
        csv_path.unlink()
    results = run_adf_filters(args, csv_path=csv_path)
    print(f"wrote {csv_path}")

    summary = {
        "dataset": args.dataset,
        "rows": int(len(results)),
        "adf_pass": int(_bool_series(results.get("adf_pass", pd.Series(dtype=bool))).sum()),
        "adf_fail": int((~_bool_series(results.get("adf_pass", pd.Series(dtype=bool)))).sum()),
        "significance": float(args.significance),
        "regression": args.regression,
        "autolag": args.autolag,
        "maxlag": args.maxlag,
    }
    summary_path = out_dir / "adf_summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)
    print(f"wrote {summary_path}")

    if args.write_passing:
        passing_path = out_dir / "adf_passing_pair_windows.csv"
        results[_bool_series(results["adf_pass"])].to_csv(passing_path, index=False)
        print(f"wrote {passing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
