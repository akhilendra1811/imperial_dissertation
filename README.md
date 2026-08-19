# Imperial Dissertation OU Pairs Code

This repository contains the core code logic for the dissertation pair-trading experiments. It is not intended to be a polished Python package; the runners add `src/` to `sys.path` so they can be run directly from the repository root.

## Workflow

The main empirical workflow is:

1. Preprocess LOBSTER data into model-ready minute quote panels.
2. Estimate Gaussian OU parameters for all candidate pair-windows.
3. Rank pairs using the Gaussian Endres rule.
4. Optionally run ADF filters and keep the best ADF-passing pairs.
5. Run the final backtests.
6. Build overlapping-window result summaries.

## Runners

The main runner is:

```bash
python runners/run_all_backtests.py \
  --sector energy \
  --year 2008 \
  --selection-scope gaussian_top10 \
  --pair-windows-csv data/selections/energy_2008_gaussian_top10_pair_windows.csv \
  --data-path data/processed_lobster_energy_2008/lobster_minute_prices_model_ready.csv.gz
```

`run_all_backtests.py` estimates the Levy models and then backtests them. It also runs the deterministic Gaussian baselines. Its default model set is:

```text
gaussian_fixed_sigma_eq
formation_mean_std
zeng_lee_gaussian_conventional
symmetric_bg
nig
cgmy
```

The default final settings are:

```text
Gaussian fixed sigma_eq multiple: 1.5
Formation mean/std multiple:      1.5
Zeng-Lee convention:              conventional, exit at mean
Cost cases:                       c0, midquote_5bps, bidask_median_c, bidask_worst_c
Gamma multipliers:                0, 0.5, 1, 2, 4
Exit rule for simulated Levy OU:  mean
```

If BG/NIG/CGMY estimates already exist, pass them with `--estimates-csv`:

```bash
python runners/run_all_backtests.py \
  --sector energy \
  --year 2008 \
  --selection-scope gaussian_top10 \
  --pair-windows-csv data/selections/energy_2008_gaussian_top10_pair_windows.csv \
  --data-path data/processed_lobster_energy_2008/lobster_minute_prices_model_ready.csv.gz \
  --estimates-csv outputs/energy_2008/estimates/model_estimates.csv
```

With `--estimates-csv`, the runner skips BG/NIG/CGMY estimation and backtests the supplied estimate rows. The Gaussian fixed 1.5, formation 1.5, and Zeng-Lee conventional baselines are still computed directly from the pair-window formation/trading data.

Other useful runners:

```bash
python runners/rank_gaussian_endres_pairs.py --gaussian-estimates PATH --dataset energy_2008
python runners/run_stationarity_filters.py --dataset energy_2008 --data-path PATH --pair-windows-csv PATH --write-passing
python runners/estimate_models.py --mode windows --pair-windows-csv PATH --data-path PATH --models gaussian,nig,cgmy,symmetric_bg
python runners/summarise_results_overlapping.py --sector energy --year 2008 --selection-scope gaussian_top10 --return-basis bid_ask
```

`estimate_models.py` is optional if you use `run_all_backtests.py`. It is useful when you want to estimate models once, inspect `model_estimates.csv`, and then pass that file back into `run_all_backtests.py` with `--estimates-csv`. For final runs, use `run_all_backtests.py` without `--estimates-csv`.

## Main Logic

Pair construction:

- `levy_ou.spreads.build_spread`
- `levy_ou.spreads.build_spread_with_anchor`
- `levy_ou.windows.PairWindow`

Gaussian ranking:

- `levy_ou.ranking.add_gaussian_endres_rank`
- `levy_ou.ranking.top_n_by_window`

Gaussian OU estimation:

- `levy_ou.estimators.gaussian_ou.fit_brownian_ou_from_spread`
- `levy_ou.estimators.gaussian_ou.estimate_brownian_ou`

Levy OU estimation:

- `levy_ou.estimators.symmetric_bg_ou.estimate_symmetric_bg_ou_wu_innovation_moments`
- `levy_ou.estimators.nig_ou.estimate_nig_ou_fixed_mean_fft_multistart`
- `levy_ou.estimators.cgmy_ou.estimate_cgmy_ou_fft_mle`

Simulation:

- `levy_ou.experiments.models.simulate_paths_from_fit`
- `levy_ou.simulation.symmetric_bg_ou.SymmetricBGOU`
- `levy_ou.simulation.nig_simulator.NIGOUFGMC`
- `levy_ou.simulation.cgmy_simulator.CGMYOUFGMC`

Backtesting:

- `levy_ou.backtesting.trade_replay.trade_real_window`
- `levy_ou.backtesting.threshold_optimisation.optimize_cost_gamma_grid`
- `levy_ou.backtesting.zeng_lee_gaussian.solve_zeng_lee_gaussian_ou_from_fit`
- `levy_ou.backtesting.basic_baseline.fit_basic_baseline`

Results:

- `levy_ou.results_summary.run_summary`
- `levy_ou.results_summary.SummaryConfig`

## Gaussian Mean-Centering Convention

All models trade the same selected pairs. Pair selection is based on the Gaussian OU Endres rank, not model-specific rankings.

The formation Gaussian OU mean is used as the fixed process mean. The fitted Levy process is centred by subtracting this Gaussian mean:

```text
Y_t = X_t - u_form
```

where `u_form` is the Gaussian OU long-run mean estimated on the formation window. The Levy model is fitted to the centred residual process `Y_t`, and simulation/backtesting adds `u_form` back so thresholds and trades are expressed in the original spread scale.

## Dependencies

Install the basic scientific stack:

```bash
pip install -r requirements.txt
```

The repository expects local LOBSTER-derived data under `data/`, but data files are ignored by git.
