# Project Structure

🇷🇺 **Russian version:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

Results are saved to `results/`.

---

## File Tree

```
VKR_Patch/
├── config/
│   └── config.yaml                  # All experiment configuration
│
├── data/
│   ├── raw/                         # Yahoo Finance data
│   │   ├── prices.csv               # Stock prices (Adj Close)
│   │   ├── log_returns.csv          # Asset log returns
│   │   └── benchmark_log_returns.csv # Benchmark log returns (SPY)
│   └── cache/iceemdan/              # Decomposition cache (gitignored)
│
├── src/
│   ├── data/
│   │   ├── downloader.py            # Data download + download_benchmark()
│   │   └── preprocessor.py          # Preprocessing (log-returns)
│   │
│   ├── decomposition/
│   │   └── iceemdan.py              # CEEMDAN/ICEEMDAN + IMF grouping + cache
│   │
│   ├── models/
│   │   ├── patchtst.py              # PatchTST: single-channel + multi-channel (for ICEEMDAN)
│   │   └── patchtst_reference/      # Reference implementation (yuqinie98/PatchTST)
│   │
│   ├── optimization/
│   │   ├── markowitz.py             # Markowitz optimizer (max Sharpe, SLSQP)
│   │   └── covariance.py            # Covariance (sample / Ledoit-Wolf)
│   │
│   ├── utils/
│   │   ├── forecast_metrics.py      # Forecast metrics (RMSE, MAE, hit-rate)
│   │   └── portfolio_metrics.py     # Portfolio metrics, turnover, costs, date split
│   │
│   └── backtesting/
│       ├── backtest.py              # Baseline 1: historical mean
│       ├── backtest_statsforecast.py # Baseline 2: AutoARIMA
│       ├── backtest_patchtst.py     # PatchTST (raw series)
│       ├── backtest_patchtst_iceemdan.py # PatchTST + ICEEMDAN
│       └── benchmarks.py            # Equal Weight (1/N) and Buy & Hold (SPY)
│
├── scripts/
│   └── precompute_decompositions.py # Parallel ICEEMDAN precompute (CPU)
│
├── notebooks/
│   ├── colab_run.ipynb              # GPU run on Google Colab
│   └── portfolio_comparison.py      # Standalone script (legacy self-contained copy)
│
├── results/                         # Backtest results + run_log_*.txt
│
├── README.md / README_EN.md         # Project description (RU / EN)
├── RESULTS.md / RESULTS_EN.md       # Research results (RU / EN)
├── PROJECT_STRUCTURE.md / _EN.md    # This file (RU / EN)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT
└── run_all.py                       # Orchestrator: strategies + benchmarks + costs
```

---

## Compared Strategies

All use **identical** windows, covariance (Ledoit-Wolf), constraints, and optimizer; only the μ estimate differs.

| Strategy | μ estimate | File |
|---|---|---|
| Baseline 1 | mean(r) × 252 | `backtest.py` |
| Baseline 2 | AutoARIMA(21).mean × 252 | `backtest_statsforecast.py` |
| PatchTST | forecast(21).mean × 252 (raw series) | `backtest_patchtst.py` |
| PatchTST + ICEEMDAN | forecast(21).mean × 252 (over components) | `backtest_patchtst_iceemdan.py` |
| Equal Weight (1/N) | — | `benchmarks.py` |
| Buy & Hold (SPY) | — | `benchmarks.py` |

---

## PatchTST + ICEEMDAN Pipeline

```
Asset training window (1260 days of log returns)
        │
        ▼  causal decomposition (training only)
CEEMDAN/ICEEMDAN → K variable IMFs + residue
        │
        ▼  deterministic grouping by mean period
3 channels: noise (<5d) | cycle (5–63d) | trend (>63d + residue)
        │
        ▼  multi-channel PatchTST (channel-independence, shared weights)
3-channel forecast for 21 days  →  sum of channels
        │
        ▼
μ = mean(forecast sum) × 252  →  Markowitz optimization
```

Decompositions are cached on disk (`data/cache/iceemdan/`). The cache key is a hash of window values and parameters, so the cache is invalidated automatically when data/params change, and look-ahead leakage is impossible.

---

## Markowitz Optimization

```
max (w'μ - rf) / √(w'Σw)
s.t. Σw = 1,  min_w ≤ w ≤ max_w  (long-only, fully invested)
```

- **μ** — expected returns (differ by method);
- **Σ** — Ledoit-Wolf covariance (same for all);
- solver — `scipy.optimize.minimize` (SLSQP).

---

## Metrics

**Portfolio** (`src/utils/portfolio_metrics.py`), on monthly simple returns annualized ×12 / √12:

| Metric | Formula |
|---|---|
| Annual Return (CAGR) | ∏(1+r)^(12/N) − 1 |
| Annual Volatility | std(r) × √12 |
| Sharpe Ratio | (mean(r) − rf_mo) / std(r) × √12 |
| Max Drawdown | min over the cumulative curve |
| Calmar Ratio | CAGR / |MaxDD| |
| Turnover | Σ|w_t − w_drift| against drifted weights |

Metrics are reported gross and net: `net = (exp(r) − 1) − cost_rate × turnover`.

**Forecast** (`src/utils/forecast_metrics.py`): RMSE, MAE, Hit Rate on monthly sums.

---

## Configuration (`config/config.yaml`)

| Block | Key parameters |
|---|---|
| `data` | `tickers`, `start_date`, `end_date`, `benchmark_ticker` |
| `backtest` | `train_window`, `test_window`, `data_end` (trim for tuning) |
| `models.patchtst` | `mode` (fast/full), `padding_patch`, `n_workers`, mode hyperparameters |
| `models.iceemdan` | `trials`, `epsilon`, `seed`, `grouping`, `forecast_noise`, `cache` |
| `optimization` | `risk_free_rate`, `covariance`, `constraints` (weights, long-only) |
| `evaluation` | `transaction_costs.cost_rate`, `split.validation_end` |

### How to Change the Stock Universe

1. Edit `data.tickers` in `config/config.yaml`.
2. Re-download data: `python src/data/downloader.py` (or answer `y` in `run_all.py`).
3. No need to clear the decomposition cache — it is keyed to data values.

---

## Running

```bash
# All strategies + benchmarks (Enter = all)
python run_all.py

# Background run (macOS: won't sleep, survives terminal close)
nohup caffeinate -is bash -c 'printf "n\n\n" | python3 run_all.py' > run.log 2>&1 &
tail -f results/run_log_*.txt

# Individual strategies
python src/backtesting/backtest.py                    # Baseline 1
python src/backtesting/backtest_statsforecast.py      # AutoARIMA
python src/backtesting/backtest_patchtst.py           # PatchTST
python src/backtesting/backtest_patchtst_iceemdan.py  # PatchTST + ICEEMDAN

# Precompute decompositions (speeds up the ICEEMDAN run)
python scripts/precompute_decompositions.py --all
```

### Output Files (in `results/`, timestamped)

- `comparison_full|validation|holdout_<ts>.csv` — summary tables by period;
- `metrics_<ts>.json` — metrics (gross/net by period) + run parameters;
- `<strategy>_returns[_net]_<ts>.csv`, `<strategy>_weights_<ts>.csv`, `<strategy>_forecasts_<ts>.csv`;
- `cumulative_returns_<ts>.png` — chart; `weight_analysis_*_<ts>.txt` — weight analysis; `run_log_<ts>.txt` — full log.

---

## Dependencies (`requirements.txt`)

```
Data:          yfinance, pandas, numpy
ML/DL:         torch
Time Series:   statsforecast (AutoARIMA), EMD-signal (CEEMDAN/ICEEMDAN)
Optimization:  scipy
Visualization: matplotlib, seaborn
Utils:         pyyaml, tqdm, scikit-learn
```
