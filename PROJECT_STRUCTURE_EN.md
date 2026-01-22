# VKR_Patch Project Structure

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all models (recommended)
python run_all.py

# 3. Or fast PatchTST mode (for debugging)
python run_all.py --fast
```

Results are saved to `results/`.

---

## File Tree

```
VKR_Patch/
├── config/
│   └── config.yaml                 # Experiment configuration
│
├── data/
│   └── raw/                        # Raw data from Yahoo Finance
│       ├── prices.csv              # Stock prices
│       └── log_returns.csv         # Log returns
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py           # Yahoo Finance data download
│   │   └── preprocessor.py         # Preprocessing (log-returns)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patchtst.py             # PatchTST Self-Supervised model
│   │   └── patchtst_reference/     # PatchTST reference implementation
│   │       ├── PatchTST_backbone.py
│   │       ├── PatchTST_layers.py
│   │       └── patchTST_selfsupervised.py
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── markowitz.py            # Markowitz optimizer (max Sharpe)
│   │   └── covariance.py           # Covariance estimation (sample / Ledoit-Wolf)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── forecast_metrics.py     # Forecast metrics (MAE, RMSE, DA)
│   │
│   └── backtesting/
│       ├── __init__.py
│       ├── backtest.py             # Baseline 1: historical mean
│       ├── backtest_statsforecast.py  # Baseline 2: StatsForecast AutoARIMA
│       └── backtest_patchtst.py    # PatchTST Self-Supervised
│
├── notebooks/
│   └── portfolio_comparison.py     # Standalone script (all three methods)
│
├── results/                        # Backtest results
│
├── .gitignore
├── LICENSE                         # MIT License
├── README.md                       # Project description (Russian)
├── README_EN.md                    # Project description (English)
├── RESULTS.md                      # Research results (Russian)
├── RESULTS_EN.md                   # Research results (English)
├── PROJECT_STRUCTURE.md            # This file (Russian)
├── PROJECT_STRUCTURE_EN.md         # This file (English)
├── requirements.txt                # Python dependencies
└── run_all.py                      # Run all models + save to results/
```

---

## Three Approaches to μ Estimation

All three methods use **identical backtest parameters** (from `config/config.yaml`):
- **TRAIN_WINDOW = 1260 days** (5 years)
- **TEST_WINDOW = 21 days** (1 month)
- **RF = 0.04** (risk-free rate)

| Approach | μ Estimation | Input | File |
|----------|--------------|-------|------|
| **Baseline 1** | mean(r) × 252 | 1260 days | `backtest.py` |
| **Baseline 2** | AutoARIMA(21).mean × 252 | 1260 days | `backtest_statsforecast.py` |
| **PatchTST** | forecast(21).mean × 252 | 1260 days | `backtest_patchtst.py` |

---

## PatchTST Self-Supervised

**Source:** https://github.com/yuqinie98/PatchTST

### Architecture

```
Input: 252 days (1 year)
    ↓
Patching: 30 patches (patch=16, stride=8)
    ↓
Embedding: Linear(16 → 128)
    ↓
Positional Encoding
    ↓
Transformer Encoder (3 layers, 16 heads)
    ↓
Prediction Head → 21 days
    ↓
μ = mean(forecast) × 252
```

### Self-Supervised Pre-training

```
1. Randomly mask 15% of patches
2. Model learns to reconstruct masked patches
3. Loss = MSE(predicted_patches, real_patches)
```

### Model Parameters (full mode)

```yaml
patchtst:
  input_length: 252         # 1 year
  pred_length: 21           # 1 month
  patch_length: 16
  stride: 8
  d_model: 128
  n_heads: 16
  n_layers: 3
  d_ff: 512
  dropout: 0.1
  use_revin: true
  mask_ratio: 0.15          # For self-supervised pretraining
  pretrain_epochs: 20
  finetune_epochs: 10
  pretrain_lr: 0.005
  batch_size: 64
```

---

## Markowitz Optimization

```
max (w'μ - rf) / √(w'Σw)
s.t. Σw = 1, w ≥ 0
```

- **μ** — expected return vector (differs by method)
- **Σ** — covariance matrix (same for all methods)
- **Constraints:** long-only, fully invested

---

## Comparison Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Sharpe Ratio** | Return per unit of risk | (R - Rf) / σ |
| **Annual Return** | Annualized return | mean(r) × 12 |
| **Annual Volatility** | Annualized volatility | std(r) × √12 |
| **Max Drawdown** | Maximum drawdown | max(peak - trough) |
| **Total Return** | Total return | exp(Σr) - 1 |

---

## Experiment Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                           DATA                                   │
│  Yahoo Finance → Adj Close → Log Returns                         │
│  20 S&P 500 stocks from 10 sectors, 2010-2025                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ROLLING WINDOW (same for all)                    │
│                                                                  │
│  Train: 1260 days (5 years) → Test: 21 days (1 month)           │
│  Step: 21 days                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FORECAST μ (expected returns)                 │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Baseline 1  │  │  Baseline 2  │  │   PatchTST   │          │
│  │  Historical  │  │    ARIMA     │  │Self-Supervised│          │
│  │    Mean      │  │              │  │              │          │
│  │ (1260 days)  │  │ (1260 days)  │  │ (1260 days)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MARKOWITZ OPTIMIZATION                        │
│                                                                  │
│  max (w'μ - rf) / √(w'Σw)                                       │
│  s.t. Σw = 1, w ≥ 0                                             │
│                                                                  │
│  Σ — covariance matrix (same for all approaches)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    COMPARE RESULTS                               │
│                                                                  │
│  Sharpe, MaxDD, Annual Return, Annual Volatility, Total Return  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Running

### All models at once (recommended):

```bash
# Run all three models + save results to results/
python run_all.py

# Fast PatchTST mode (for debugging)
python run_all.py --fast
```

Results are saved to `results/`:
- `comparison_YYYYMMDD_HHMMSS.csv` — metrics summary table
- `metrics_YYYYMMDD_HHMMSS.json` — metrics in JSON
- `*_returns_YYYYMMDD_HHMMSS.csv` — returns for each model

### Individual backtests:

```bash
# Baseline 1: Historical mean
python src/backtesting/backtest.py

# Baseline 2: StatsForecast AutoARIMA
python src/backtesting/backtest_statsforecast.py

# PatchTST Self-Supervised
python src/backtesting/backtest_patchtst.py
```

---

## Dependencies (requirements.txt)

```
# Data
yfinance, pandas, numpy

# ML/DL
torch, pytorch-lightning

# Time Series
statsforecast (AutoARIMA)

# Optimization
scipy, cvxpy

# Visualization
matplotlib, seaborn, plotly

# Utils
pyyaml, tqdm, scikit-learn

# Testing
pytest
```
