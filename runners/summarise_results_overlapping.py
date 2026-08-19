#!/usr/bin/env python3
"""Summarise final backtest results on overlapping rolling-window vintages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from levy_ou.results_summary import (
    DEFAULT_FAMA_FRENCH_DAILY_FACTORS_PATH,
    SummaryConfig,
    canonical_model,
    canonical_return_basis,
    run_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sector", required=True, help="Sector label, for example energy or communication.")
    parser.add_argument("--year", required=True, help="Dataset year, for example 2008 or 2024.")
    parser.add_argument("--model", default="all", help="Model alias: all, gaussian, symmetric_bg, nig, cgmy.")
    parser.add_argument(
        "--selection-scope",
        choices=["all", "gaussian_top10", "adf_capped10"],
        default="all",
        help="Backtest branch to read.",
    )
    parser.add_argument("--outputs-root", default="outputs", help="Root containing cleaned outputs.")
    parser.add_argument(
        "--return-basis",
        default=None,
        choices=["all", "bid_ask", "midquote_fixed_bps", "midquote"],
        help="Return basis to summarise. Defaults to bid_ask.",
    )
    parser.add_argument(
        "--return-col",
        default=None,
        help="Deprecated alias for --return-basis; existing return-column names are accepted.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Accepted for compatibility. The runner writes the fixed result-summary filenames.",
    )
    parser.add_argument("--pair-slots", type=int, default=10, help="Committed pair slots per vintage portfolio.")
    parser.add_argument(
        "--pair-slot-mode",
        choices=["fixed_slots"],
        default="fixed_slots",
        help="Accepted for compatibility; result summaries use fixed committed pair slots.",
    )
    parser.add_argument("--trading-window-days", type=int, default=10, help="Number of live vintages for full_only.")
    parser.add_argument("--annualisation-days", type=int, default=252, help="Annualisation day count.")
    parser.add_argument(
        "--fama-french-factors-path",
        type=Path,
        default=DEFAULT_FAMA_FRENCH_DAILY_FACTORS_PATH,
        help="Kenneth French U.S. Fama/French 3 Factors daily CSV. Uses RF/100 by date.",
    )
    parser.add_argument("--hac-lags", type=int, default=10, help="Newey-West HAC lag count.")
    parser.add_argument(
        "--boundary-policy",
        choices=["full_only", "cash_padded", "active_average"],
        default="full_only",
        help="How to handle start/end dates without a full vintage committee.",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Optional explicit processed LOBSTER quote panel.")
    parser.add_argument("--reconciliation-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    basis = canonical_return_basis(args.return_basis or args.return_col or "bid_ask")
    config = SummaryConfig(
        sector=args.sector,
        year=str(args.year),
        model=canonical_model(args.model),
        selection_scope=args.selection_scope,
        outputs_root=Path(args.outputs_root),
        return_basis=basis,
        pair_slots=int(args.pair_slots),
        trading_window_days=int(args.trading_window_days),
        annualisation_days=int(args.annualisation_days),
        hac_lags=int(args.hac_lags),
        boundary_policy=args.boundary_policy,
        data_path=args.data_path,
        fama_french_factors_path=args.fama_french_factors_path,
        reconciliation_tolerance=float(args.reconciliation_tolerance),
        cli_args={key: str(value) for key, value in vars(args).items()},
    )
    output_dir = run_summary(config, mode="overlapping")
    print(f"wrote overlapping result summaries under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
