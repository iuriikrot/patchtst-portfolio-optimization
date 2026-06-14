# Research Results

**Author:** Iurii Krotov
**Date:** June 2026

🇷🇺 **Russian version:** [RESULTS.md](RESULTS.md)

---

## 1. Experiment Parameters

| Parameter | Value |
|---|---|
| Data period | 2000-01-01 — 2026-01-15 |
| Backtest period | ~2005 — 2026 (251 rebalancings) |
| Number of stocks | 20 (from 10 S&P 500 sectors) |
| Training window | 1260 days (5 years) |
| Forecast horizon / rebalancing | 21 days (1 month) |
| Risk-free rate | 4% annually |
| Covariance matrix | Ledoit–Wolf (identical across all methods) |
| Weight constraints | min 1%, max 20%, long-only, fully invested |
| Transaction costs | 5 bps per unit of turnover |
| Validation / holdout boundary | 2014-12-31 |

### PatchTST Parameters (full mode)

| Parameter | Value |  | Parameter | Value |
|---|---|---|---|---|
| input_length | 252 | | d_ff | 512 |
| pred_length | 21 | | dropout | 0.1 |
| patch_length | 16 | | mask_ratio | 0.15 |
| stride | 8 | | pretrain_epochs | 20 |
| d_model | 128 | | finetune_epochs | 10 |
| n_heads | 16 | | pretrain_lr | 0.005 |
| n_layers | 3 | | padding_patch | true |

### ICEEMDAN Parameters

| Parameter | Value |
|---|---|
| Implementation | PyEMD / EMD-signal (CEEMDAN with Colominas 2014 improvements ≈ ICEEMDAN) |
| trials (noise ensemble) | 50 |
| epsilon | 0.2 |
| IMF grouping | noise < 5 days, cycle 5–63 days, trend > 63 days + residue |
| Channels | 3 (noise / cycle / trend) → multi-channel PatchTST |
| Causality | only the training window is decomposed; the test set is never used |

---

## 2. Portfolio Metrics Comparison (full period, net, 5 bps)

| Metric | Baseline 1 | AutoARIMA | PatchTST | **ICEEMDAN** | 1/N | SPY |
|---|---|---|---|---|---|---|
| Annual Return | 14.7% | 14.5% | 11.5% | **17.0%** | 13.1% | 10.8% |
| Annual Volatility | 15.4% | 16.2% | 15.9% | 16.9% | 15.2% | 16.8% |
| Sharpe Ratio | 0.72 | 0.68 | 0.52 | **0.87** | 0.66 | 0.48 |
| Calmar Ratio | 0.44 | 0.40 | 0.21 | **0.49** | 0.35 | 0.21 |
| Max Drawdown | −33.3% | −35.9% | −53.8% | −34.9% | −37.4% | −50.2% |
| Total Return | 1663% | 1602% | 876% | **2584%** | 1212% | 747% |
| Turnover/mo | 18% | 50% | 113% | 112% | 4% | 0% |

Gross (no costs) Sharpe: ICEEMDAN = 0.92, Baseline 1 = 0.73.

---

## 3. Results by Period (Sharpe, net)

The core methodological idea is to separate the hyperparameter-tuning period (*validation*) from honest evaluation (*holdout*) to rule out overfitting to the test period.

| Period | Baseline 1 | AutoARIMA | PatchTST | **ICEEMDAN** | 1/N | SPY |
|---|---|---|---|---|---|---|
| validation 2005–2015 | 0.68 | 0.73 | 0.31 | **0.86** | 0.56 | 0.29 |
| **holdout 2015–2026** | 0.75 | 0.63 | 0.75 | **0.88** | 0.75 | 0.66 |
| full 2005–2026 | 0.72 | 0.68 | 0.52 | **0.87** | 0.66 | 0.48 |

**ICEEMDAN is the best by Sharpe across all three periods.** This is fundamentally different from a situation where a good result is achieved only on the period the hyperparameters were tuned on (see Section 7).

Notes:
- On **validation** (includes the 2008 crisis) the spread between strategies is large: the raw PatchTST collapses (0.31), ICEEMDAN leads (0.86).
- **holdout** (2015–2026) is mostly a rising market with one sharp drawdown (COVID-2020). Everything does well there, so the gaps are smaller: the raw PatchTST recovers to 0.75 (still no better than 1/N), while ICEEMDAN keeps the lead (0.88).

---

## 4. Forecast Quality Metrics (full period)

| Metric | Baseline 1 | AutoARIMA | PatchTST | ICEEMDAN |
|---|---|---|---|---|
| RMSE | **0.074** | 0.075 | 0.095 | 0.108 |
| Hit Rate | **56.7%** | 54.0% | 51.4% | 51.3% |

**The accuracy paradox.** ICEEMDAN has the worst point forecast accuracy (RMSE 0.108, hit-rate 51% — coin-flip level) yet builds the best portfolio. This means that in the Markowitz problem what matters is not the accuracy of each stock's forecast in isolation, but the **relative ranking** of assets: which are over- or under-priced relative to one another. The decomposition apparently improves this relative structure rather than absolute accuracy.

---

## 5. Statistical Significance Test

Sharpe-ratio differences were tested via paired bootstrap (10,000 iterations; H₀: Sharpe(ICEEMDAN) ≤ Sharpe(X)).

**Full period (251 months):**

| ICEEMDAN vs | ΔSharpe | p-value | Significance |
|---|---|---|---|
| PatchTST (raw) | +0.35 | 0.005 | significant (1%) |
| SPY Buy & Hold | +0.39 | 0.004 | significant (1%) |
| Equal Weight (1/N) | +0.21 | 0.027 | significant (5%) |
| AutoARIMA | +0.19 | 0.094 | marginal (10%) |
| **Baseline 1 (hist. mean)** | +0.15 | 0.127 | **not significant** |

**Holdout (131 months):** on the holdout period alone, due to the smaller sample and the "easy" market, no difference reaches significance (p = 0.23–0.28 against the nearest competitors).

**Conclusion.** Two facts are statistically robust: (1) decomposition significantly improves the transformer (ICEEMDAN ≫ PatchTST), and (2) ICEEMDAN significantly beats the naive benchmarks. However, ICEEMDAN does **not** statistically beat plain historical-mean Markowitz — it is consistently ahead numerically, but the difference is within statistical noise.

---

## 6. Turnover and Cost Sensitivity

ICEEMDAN forms far more "mobile" portfolios: the mean weight change per rebalancing is 5.6% vs 0.8% for Baseline 1 (6.7× higher), turnover 112%/mo vs 18%. This is its main weakness.

**Sharpe (full period) at different cost levels:**

| Costs | ICEEMDAN | Baseline 1 |
|---|---|---|
| 0 bps | 0.92 | 0.73 |
| 5 bps (base case) | 0.87 | 0.72 |
| 10 bps | 0.83 | 0.71 |
| 20 bps | 0.74 | 0.70 |
| 30 bps | 0.65 | **0.69** ← baseline ahead |
| 50 bps | 0.47 | **0.66** |

ICEEMDAN's edge survives costs up to ~20–25 bps (realistic for liquid US large-caps) but disappears beyond 30 bps — on less liquid assets or once slippage is accounted for, the low-turnover baseline becomes preferable.

---

## 7. Key Findings

### 7.1. Decomposition "rescues" the transformer — the main result

The raw PatchTST on daily returns is one of the worst strategies (Sharpe 0.52 full period, 0.31 on the crisis validation): the high-capacity model finds no stable signal in the noisy series and overfits to noise. A prior ICEEMDAN decomposition changes the picture radically: **the same network, the same hyperparameters**, but the input is frequency-separated components, yields Sharpe 0.87 (significant, p = 0.005). The decomposition separates the predictable structure (trend, cycles) from the unpredictable noise, so the transformer stops spending capacity on memorizing the latter.

### 7.2. The advantage holds out-of-tuning

ICEEMDAN is the best by Sharpe on validation (0.86), holdout (0.88), and the full period (0.87). Consistency across non-overlapping periods is the main evidence that the advantage is not an artifact of hyperparameter tuning.

### 7.3. Honest boundary: we did not statistically beat plain Markowitz

ICEEMDAN significantly beats the naive benchmarks (1/N, SPY) and the raw transformer, but is **statistically indistinguishable** from classical historical-mean Markowitz (p = 0.127). This matches the literature (DeMiguel, Garlappi, Uppal, 2009: simple allocation rules are very hard to beat reliably). The numerical edge is real and robust, but claiming statistical superiority over the strongest baseline on this sample would be incorrect.

### 7.4. Forecast accuracy does not determine portfolio quality

The best portfolio method (ICEEMDAN) is the worst by RMSE and hit-rate. What matters for Markowitz optimization is the relative structure of expected returns, not point accuracy.

---

## 8. Limitations

1. **Survivorship bias.** A universe of 20 firms selected in 2026 with full history since 2000 excludes delistings and bankruptcies — absolute returns of all strategies are inflated. A point-in-time index composition would be more correct.
2. **Transaction costs.** A linear commission is modeled; market impact and slippage at 112%/mo turnover are not — ICEEMDAN's real-world result would be lower.
3. **Single seed.** The neural network is stochastic; results are for one seed (cross-platform MPS/CUDA check gave ±0.05 Sharpe spread). For publication, multiple seeds and confidence intervals are desirable.
4. **In-train decomposition compromise.** Supervised fine-tuning pairs are built from components computed over the whole training window, so there is mild in-window "peeking" (the test period is untouched — causality is preserved). A strictly causal alternative is an order of magnitude more expensive.
5. **EMD boundary effect.** The end of the window (which feeds the forecast) is decomposed least reliably — a known limitation of empirical-mode-decomposition methods.
6. **Constant risk-free rate** of 4% over 2005–2026, including the zero-rate era.

---

## 9. Directions for Further Research

1. Reducing ICEEMDAN turnover: turnover regularization in the optimizer, smoothing of μ forecasts, shrinkage toward the historical mean.
2. Point-in-time universe to remove survivorship bias.
3. Multiple seeds and formal confidence intervals for Sharpe.
4. Comparing decomposition schemes (number of channels, grouping thresholds, VMD vs ICEEMDAN).
5. Extension to other asset classes and markets.

---

## 10. Summary Table (full period, net)

```
============================================================
PORTFOLIO METRICS — net, 5 bps costs (251 periods, 2005–2026)
============================================================

Metric             Baseline1   AutoARIMA  PatchTST  ICEEMDAN     1/N     SPY
------------------------------------------------------------------------------
Annual Return         14.7%      14.5%      11.5%    17.0% ★    13.1%   10.8%
Sharpe Ratio           0.72       0.68       0.52     0.87 ★     0.66    0.48
Calmar Ratio           0.44       0.40       0.21     0.49 ★     0.35    0.21
Max Drawdown         -33.3%     -35.9%     -53.8%   -34.9%     -37.4%  -50.2%
Total Return          1663%      1602%       876%    2584% ★    1212%    747%
Turnover/mo             18%        50%       113%     112%        4%      0%

★ = best result. ICEEMDAN significantly (bootstrap) beats PatchTST,
1/N and SPY, but not Baseline 1 (p=0.127).
```
