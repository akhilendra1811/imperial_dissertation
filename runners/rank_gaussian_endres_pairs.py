#!/usr/bin/env python3
"""Build Gaussian Endres-ranked pair-window selections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from levy_ou.ranking import add_gaussian_endres_rank, top_n_by_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-estimates", required=True, help="CSV containing Gaussian OU estimates for candidate pair-windows.")
    parser.add_argument("--output-dir", default="data/selections")
    parser.add_argument("--dataset", required=True, help="Dataset label such as energy_2008.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--ranked-name", default=None)
    parser.add_argument("--top-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimates = pd.read_csv(args.gaussian_estimates)
    ranked = add_gaussian_endres_rank(estimates, top_n=args.top_n)
    top = top_n_by_window(ranked, "gaussian_endres_selection_rank", args.top_n)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranked_path = out / (args.ranked_name or f"{args.dataset}_gaussian_all_pair_windows_ranked.csv")
    top_path = out / (args.top_name or f"{args.dataset}_gaussian_top{args.top_n}_pair_windows.csv")
    summary_path = out / f"{args.dataset}_gaussian_all_pair_windows_ranked_summary.json"

    ranked.to_csv(ranked_path, index=False)
    top.to_csv(top_path, index=False)
    summary = {
        "inputs": {"gaussian_estimates": str(Path(args.gaussian_estimates).resolve())},
        "outputs": {"ranked_all_pairs": str(ranked_path.resolve()), "gaussian_top": str(top_path.resolve())},
        "windows": int(ranked["window_id"].nunique()) if "window_id" in ranked.columns else 0,
        "all_pair_windows": int(len(ranked)),
        "rankable_pair_windows": int(ranked["is_rankable_estimate"].sum()) if "is_rankable_estimate" in ranked.columns else 0,
        "gaussian_top_rows": int(len(top)),
        "ranking_rule": "within each window: rank theta descending and sigma_eq descending, sum ranks, then sort by rank sum, theta, sigma_eq, and tickers",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {ranked_path}")
    print(f"wrote {top_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
