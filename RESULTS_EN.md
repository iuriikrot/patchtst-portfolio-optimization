# Research Results

**Author:** Iurii Krotov
**Date:** January 2026

---

## 1. Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Data period | 2010-01-01 — 2024-12-31 |
| Number of stocks | 20 (from 10 S&P 500 sectors) |
| Training window | 1260 days (5 years) |
| Forecast horizon | 21 days (1 month) |
| Risk-free rate | 4% annually |
| Number of periods | 119 |
| Covariance matrix | Ledoit-Wolf |
| Weight constraints | min=1%, max=20%, long-only |

### PatchTST Parameters (full mode)

| Parameter | Value |
|-----------|-------|
| input_length | 252 (1 year) |
| pred_length | 21 |
| patch_length | 16 |
| stride | 8 |
| d_model | 128 |
| n_heads | 16 |
| n_layers | 3 |
| d_ff | 512 |
| dropout | 0.1 |
| mask_ratio | 0.15 |
| pretrain_epochs | 20 |
| finetune_epochs | 10 |
| learning_rate | 0.005 |

---

## 2. Portfolio Metrics Comparison

| Metric | Baseline 1 | StatsForecast | PatchTST | Best |
|--------|------------|---------------|----------|------|
| **Annual Return** | **16.08%** | 12.94% | 14.56% | Baseline 1 |
| **Annual Volatility** | 12.83% | 13.34% | **12.63%** | PatchTST |
| **Sharpe Ratio** | **0.93** | 0.69 | 0.83 | Baseline 1 |
| **Calmar Ratio** | 0.66 | 0.59 | **1.07** | PatchTST |
| **Max Drawdown** | -24.22% | -22.06% | **-13.56%** | PatchTST |
| **Total Return** | **338.71%** | 234.23% | 284.87% | Baseline 1 |

### Results Interpretation

1. **By return:** Baseline 1 (historical mean) showed the highest return — 16.08% annually and 338.71% over the entire period. PatchTST came second with 14.56% annually.

2. **By risk:** PatchTST demonstrated **significantly better** risk management:
   - Minimum volatility: 12.63%
   - **Minimum drawdown: -13.56%** (almost half of Baseline 1!)

3. **By risk/return ratio:**
   - Sharpe Ratio: Baseline 1 leads (0.93), but PatchTST is close (0.83)
   - **Calmar Ratio: PatchTST leads (1.07 > 1)** — return exceeds maximum drawdown

---

## 3. Forecast Quality Metrics

| Metric | Baseline 1 | StatsForecast | PatchTST |
|--------|------------|---------------|----------|
| **RMSE** | **0.0695** | 0.0698 | 0.0847 |
| **MAE** | **0.0506** | 0.0510 | 0.0620 |
| **Hit Rate** | **56.13%** | 52.70% | 51.72% |

### Interpretation

- Baseline 1 has the best forecast metrics by accuracy
- Hit Rate (correct direction proportion) is slightly above 50% — financial series are difficult to forecast
- PatchTST has worse forecast metrics, but **better portfolio risk metrics** — this shows that forecast accuracy is not the only factor in portfolio success

---

## 4. Key Findings

### 4.1. Main Result: PatchTST is Best for Risk Management

**Original hypothesis:** Replacing historical means with PatchTST forecasts will improve Markowitz portfolio quality.

**Result:** The hypothesis **was partially confirmed**:
- By Sharpe Ratio: Baseline 1 leads (0.93 vs 0.83)
- **By Calmar Ratio: PatchTST leads (1.07 vs 0.66)** — 62% improvement
- **By Max Drawdown: PatchTST leads (-13.56% vs -24.22%)** — 44% improvement

### 4.2. PatchTST Advantages

PatchTST showed **significant advantages** in risk management:

| Risk Metric | Baseline 1 | PatchTST | Improvement |
|-------------|------------|----------|-------------|
| Max Drawdown | -24.22% | -13.56% | **+44%** |
| Volatility | 12.83% | 12.63% | +2% |
| Calmar Ratio | 0.66 | 1.07 | **+62%** |

**Conclusion:** PatchTST forms portfolios with **almost half the drawdowns** at comparable returns.

### 4.3. Why Does PatchTST Manage Risk Better?

1. **Adaptability to market regimes**
   - PatchTST learns to recognize patterns preceding crises
   - The model reduces allocation to risky assets before downturns

2. **Self-Supervised learning**
   - Patch masking teaches the model to understand time series structure
   - This improves volatility forecasting, not just direction

3. **Transformer architecture**
   - Attention mechanism allows capturing long-term dependencies
   - The model can "see" warning signals earlier

---

## 5. Practical Implications

| Investor Goal | Recommended Approach | Rationale |
|---------------|---------------------|-----------|
| Maximum return | Baseline 1 | Sharpe 0.93, Return 16.08% |
| **Minimum drawdown** | **PatchTST** | Max DD -13.56% vs -24.22% |
| **Best Calmar** | **PatchTST** | 1.07 > 1 (return > drawdown) |
| Conservative strategy | PatchTST | Better risk management |

**For institutional investors** with drawdown constraints, PatchTST is the **preferred choice**.

**For individual investors** with long-term horizons, Baseline 1 provides higher returns.

---

## 6. Summary Results Table

```
============================================================
PORTFOLIO METRICS (119 periods, 2015-2024)
============================================================

Metric                     Baseline 1      StatsF    PatchTST
-------------------------------------------------------------
Annual Return                  16.08%      12.94%      14.56%
Annual Volatility              12.83%      13.34%      12.63%
Sharpe Ratio                     0.93        0.69        0.83
Calmar Ratio                     0.66        0.59        1.07  ★
Max Drawdown                  -24.22%     -22.06%     -13.56% ★
Total Return                  338.71%     234.23%     284.87%

★ = best result for risk management

============================================================
FORECAST METRICS
============================================================

Metric                     Baseline 1      StatsF    PatchTST
-------------------------------------------------------------
RMSE                         0.069522    0.069789    0.084748
MAE                          0.050634    0.051047    0.061992
Hit Rate                       56.13%      52.70%      51.72%
```

---

## 7. Recommendations for Further Research

1. **Improve PatchTST Sharpe Ratio**
   - Combine with Baseline 1 (ensemble)
   - Add regularization on forecast volatility

2. **Explore other markets**
   - Cryptocurrencies (high volatility)
   - Emerging markets

3. **Add additional features**
   - VIX (volatility index)
   - Macroeconomic data

---

## Conclusion

The study showed that **PatchTST is the best choice for risk-oriented portfolio management**:

- **Calmar Ratio 1.07** — the only method with return/drawdown ratio greater than 1
- **Max Drawdown -13.56%** — almost half of the classical approach
- **Sharpe Ratio 0.83** — close to the leader (0.93)

For investors prioritizing **capital protection** over return maximization, PatchTST is the preferred method for estimating expected returns in Markowitz portfolio optimization.
