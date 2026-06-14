# PatchTST and ICEEMDAN for Markowitz Portfolio Optimization

**Comparison of Expected-Return Estimators for Markowitz Portfolio Optimization: Historical Mean, AutoARIMA, PatchTST, and PatchTST over ICEEMDAN Decomposition**

**Author:** Iurii Krotov
**Date:** June 2026

🇷🇺 **Russian version:** [README.md](README.md)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all models (interactive)
python run_all.py
```

Results are saved to `results/`.

---

## Topic and Objective

Empirical test of the hypothesis: does replacing historical means with PatchTST transformer forecasts improve the expected-return estimate μ in Markowitz portfolio optimization — and does a prior signal decomposition of the series (ICEEMDAN) unlock the transformer's potential?

## Problem Statement

The classical Markowitz problem — maximization of the tangency-portfolio Sharpe ratio:

```
max  (w'μ - r_f) / sqrt(w'Σw)
s.t. Σ w_i = 1,  min_w ≤ w_i ≤ max_w  (long-only)
```

where **w** — asset weights, **μ** — expected-return vector (the project compares ways to estimate it), **Σ** — covariance matrix (Ledoit–Wolf, **identical across all methods**), **r_f** — risk-free rate. The compared strategies differ **only in how μ is estimated** — everything else (windows, covariance, constraints, optimizer) is identical, making the comparison fair.

## Compared Approaches

| Strategy | μ estimate | Description |
|---|---|---|
| **Baseline 1** | mean(r) × 252 | Classic Markowitz (historical mean) |
| **Baseline 2** | AutoARIMA(21).mean × 252 | StatsForecast AutoARIMA |
| **PatchTST** | forecast(21).mean × 252 | Self-supervised transformer on the raw series |
| **PatchTST + ICEEMDAN** | forecast(21).mean × 252 | The same transformer on decomposition components |
| **Equal Weight (1/N)** | — | Naive benchmark: equal weights |
| **Buy & Hold (SPY)** | — | Naive benchmark: S&P 500 index |

**PatchTST + ICEEMDAN** is the project's main contribution. Each asset's training window is decomposed causally (no look-ahead) via CEEMDAN/ICEEMDAN; the variable number of IMFs is deterministically grouped into 3 channels by mean period — **noise** (< 5 days), **cycle** (5–63 days), **trend** (> 63 days + residue). The channels feed a single multi-channel PatchTST (channel-independence, shared weights); the final forecast is the sum of channel forecasts.

## Evaluation Methodology

- **Walk-forward backtest:** training window 1260 days (5 years), test 21 days (1 month), step 21 days. Models are retrained at every step.
- **Transaction costs:** turnover is computed against drifted weights, 5 bps per unit of turnover by default. Metrics are reported both gross and net.
- **Honest hyperparameter split:** *validation* (test months ≤ 2014-12-31) — tuning; *holdout* (2015–2026) — final out-of-tuning evaluation; *full* — entire period.
- **Significance test:** Sharpe-ratio differences are tested via paired bootstrap (10,000 iterations).

## Data

- **Assets:** 20 stocks from 10 S&P 500 sectors (set in `config/config.yaml`).
- **Data period:** 2000-01-01 — 2026-01-15; the backtest covers ~2005–2026 (251 rebalancings).
- **Source:** Yahoo Finance (Adjusted Close, accounts for dividends and splits).
- **Files:** `data/raw/prices.csv`, `data/raw/log_returns.csv`, `data/raw/benchmark_log_returns.csv` (SPY).

> ⚠️ A universe of 20 currently-traded firms with full history since 2000 carries **survivorship bias** — absolute returns of all strategies are inflated. This is accounted for in interpretation (see [RESULTS_EN.md](RESULTS_EN.md)).

---

## Results (full period, net, 5 bps costs)

| Metric | Baseline 1 | AutoARIMA | PatchTST | **PatchTST+ICEEMDAN** | 1/N | SPY |
|---|---|---|---|---|---|---|
| Sharpe Ratio | 0.72 | 0.68 | 0.52 | **0.87** | 0.66 | 0.48 |
| Calmar Ratio | 0.44 | 0.40 | 0.21 | **0.49** | 0.35 | 0.21 |
| Max Drawdown | −33.3% | −35.9% | −53.8% | −34.9% | −37.4% | −50.2% |
| Annual Return | 14.7% | 14.5% | 11.5% | **17.0%** | 13.1% | 10.8% |
| Turnover/mo | 18% | 50% | 113% | 112% | 4% | 0% |

**By period (Sharpe, net):**

| Period | Baseline 1 | AutoARIMA | PatchTST | **ICEEMDAN** | 1/N | SPY |
|---|---|---|---|---|---|---|
| validation 2005–2015 | 0.68 | 0.73 | 0.31 | **0.86** | 0.56 | 0.29 |
| **holdout 2015–2026** | 0.75 | 0.63 | 0.75 | **0.88** | 0.75 | 0.66 |
| full 2005–2026 | 0.72 | 0.68 | 0.52 | **0.87** | 0.66 | 0.48 |

![Cumulative Returns Comparison](results/cumulative_returns_20260612_091600.png)

### Key Findings

1. **Decomposition "rescues" the transformer.** The same network yields Sharpe 0.52 on the raw series and **0.87** on ICEEMDAN components. The difference is statistically significant (bootstrap, p = 0.005) — the project's main result.
2. **ICEEMDAN is best on a risk-adjusted basis across all three periods** (0.86 / 0.88 / 0.87) — the advantage is robust and holds out-of-tuning.
3. **Honest boundary of the result.** Under the rigorous Ledoit–Wolf test (HAC + studentized bootstrap, Holm correction) ICEEMDAN significantly beats the naive benchmarks (1/N, SPY) and the raw transformer, but over historical-mean Markowitz its superiority is only **marginally significant (10%, Holm-p ≈ 0.088)**, not reaching 5%. This matches the literature finding that simple baselines are very hard to beat reliably (DeMiguel et al., 2009).
4. **Forecast accuracy ≠ portfolio quality.** ICEEMDAN has the worst RMSE and hit-rate yet the best portfolio: the optimizer cares about the relative structure of μ across assets, not point accuracy.
5. **Limitation — turnover.** ICEEMDAN rebalances ~112%/month. Its edge over the baseline survives up to ~20–25 bps of costs and disappears beyond 30 bps.

Detailed breakdown, significance test, and cost-sensitivity analysis: **[RESULTS_EN.md](RESULTS_EN.md)**.

---

## Configuration

All parameters live in `config/config.yaml`. Main blocks:

- `data` — tickers, period, benchmark ticker;
- `backtest` — train/test windows, `data_end` (period trimming for tuning);
- `models.patchtst` — `fast`/`full` mode, `padding_patch`, `n_workers` (per-ticker parallelism);
- `models.iceemdan` — CEEMDAN params (`trials`, `epsilon`, `seed`), IMF grouping, denoising flag, cache;
- `optimization` — risk-free rate, covariance method, weight constraints;
- `evaluation` — transaction costs, validation/holdout boundary.

### How to Change the Stock Universe

1. Edit the `data.tickers` list in `config/config.yaml` (Yahoo Finance tickers). Adjust `start_date`/`end_date` and `benchmark_ticker` if needed.
2. Re-download the data:
   ```bash
   python src/data/downloader.py        # overwrites data/raw/*.csv for the new list
   ```
   or run `python run_all.py` and answer `y` to "Download data again?".
3. The old decomposition cache (`data/cache/iceemdan/`) need not be cleared — the cache key includes window values, so new data is recomputed automatically.

> Constraint: every ticker must have quotes over the entire period — rows with gaps are dropped wholesale (panel alignment on common trading days).

---

## Installation and Running

```bash
python3 -m pip install -r requirements.txt
```

### Full run (interactive)

```bash
python3 run_all.py
```

The script asks whether to download data and which models to run (`Enter` = all four + benchmarks). All output is mirrored to `results/run_log_<timestamp>.txt`.

### Background execution (long runs)

A full PatchTST/ICEEMDAN run takes hours. To survive a terminal close and system sleep (on macOS):

```bash
# macOS: caffeinate prevents sleep; keep the lid open
nohup caffeinate -is bash -c 'printf "n\n\n" | python3 run_all.py' > run.log 2>&1 &

# Live progress
tail -f results/run_log_*.txt
```

`printf "n\n\n"` answers "don't download" and "run all models" to the interactive prompts.

### Selecting individual strategies

When prompted for models, enter comma-separated numbers: `1` — Baseline 1, `2` — AutoARIMA, `3` — PatchTST, `4` — PatchTST + ICEEMDAN. The 1/N and SPY benchmarks are always computed.

### Individual backtests

```bash
python3 src/backtesting/backtest.py                   # Baseline 1
python3 src/backtesting/backtest_statsforecast.py     # AutoARIMA
python3 src/backtesting/backtest_patchtst.py          # PatchTST
python3 src/backtesting/backtest_patchtst_iceemdan.py # PatchTST + ICEEMDAN
```

### Running on GPU (Google Colab)

For machines without a local GPU there is a notebook [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb): upload the project archive to Google Drive, pick a GPU runtime, and run the cells. The notebook includes a speed benchmark and decomposition precompute. Note: on this project's small model, Apple Silicon (MPS) can be faster than a cloud L4 — see the benchmark cell.

---

## Project Structure

See [PROJECT_STRUCTURE_EN.md](PROJECT_STRUCTURE_EN.md). In brief:

```
├── run_all.py                          # Orchestrator: all strategies + benchmarks + costs
├── config/config.yaml                  # All configuration
├── data/raw/                           # prices, log_returns, benchmark
├── scripts/precompute_decompositions.py # Parallel ICEEMDAN precompute (CPU)
├── notebooks/colab_run.ipynb           # GPU run on Colab
├── src/
│   ├── data/                           # Loading and preprocessing
│   ├── decomposition/iceemdan.py       # CEEMDAN/ICEEMDAN + IMF grouping + cache
│   ├── models/patchtst.py              # PatchTST (single- and multi-channel) + reference/
│   ├── optimization/                   # Markowitz (max Sharpe) + covariance
│   ├── backtesting/                    # 4 strategies + benchmarks.py
│   └── utils/                          # Forecast and portfolio metrics (turnover, costs, split)
└── results/                            # Experiment results
```

## License

MIT License. See [LICENSE](LICENSE).

### Acknowledgements

- [PatchTST](https://github.com/yuqinie98/PatchTST) — Transformer architecture for time series (Apache 2.0)
- [PyEMD / EMD-signal](https://github.com/laszukdawid/PyEMD) — CEEMDAN/ICEEMDAN implementation
- [StatsForecast](https://github.com/Nixtla/statsforecast) — AutoARIMA
