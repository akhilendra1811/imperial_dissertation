#!/usr/bin/env python3
"""Estimate OU models from synthetic data or pair-window CSVs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from levy_ou.experiments.outputs import write_csv_json
from levy_ou.experiments.pipeline import run_synthetic_estimation, run_window_estimation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "windows"], default="windows")
    parser.add_argument("--models", default="gaussian,nig,cgmy,symmetric_bg")
    parser.add_argument("--output-dir", default="outputs/estimates")
    parser.add_argument("--pair-windows-csv")
    parser.add_argument("--data-path", required=False, default="data/processed_lobster_all_tickers/lobster_minute_prices_model_ready.csv.gz")
    parser.add_argument("--price-col", default="model_price_close")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--observations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fft-grid-size", type=int, default=512)
    parser.add_argument("--truncation-l", type=float, default=6.0)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--max-optimizer-starts", type=int, default=1)
    parser.add_argument("--max-candidate-starts-to-score", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fit_kwargs = {
        "fft_grid_size": args.fft_grid_size,
        "truncation_l": args.truncation_l,
        "maxiter": args.maxiter,
        "max_optimizer_starts": args.max_optimizer_starts,
        "max_candidate_starts_to_score": args.max_candidate_starts_to_score,
    }
    if args.mode == "synthetic":
        estimates = run_synthetic_estimation(args.models, observations=args.observations, seed=args.seed, **fit_kwargs)
    else:
        if not args.pair_windows_csv:
            raise SystemExit("--pair-windows-csv is required when --mode windows")
        estimates = run_window_estimation(
            pair_windows_csv=args.pair_windows_csv,
            models=args.models,
            data_path=args.data_path,
            price_col=args.price_col,
            min_observations=args.min_observations,
            max_windows=args.max_windows,
            seed=args.seed,
            n_jobs=args.n_jobs,
            **fit_kwargs,
        )
    csv_path, json_path = write_csv_json(estimates, Path(args.output_dir), "model_estimates", summary=vars(args))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
