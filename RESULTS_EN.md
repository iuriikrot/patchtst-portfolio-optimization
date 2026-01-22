# Research Results

**Author:** Iurii Krotov, HSE University
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
| learning_rate | 0.01 |

---

## 2. Portfolio Metrics Comparison

| Metric | Baseline 1 | StatsForecast | PatchTST | Best |
|--------|------------|---------------|----------|------|
| **Annual Return** | **16.08%** | 13.08% | 12.20% | Baseline 1 |
| **Annual Volatility** | 12.83% | 13.26% | **12.13%** | PatchTST |
| **Sharpe Ratio** | **0.93** | 0.70 | 0.69 | Baseline 1 |
| **Calmar Ratio** | 0.66 | 0.63 | **0.78** | PatchTST |
| **Max Drawdown** | -24.22% | -20.72% | **-15.72%** | PatchTST |
| **Total Return** | **338.71%** | 238.30% | 213.03% | Baseline 1 |

### Results Interpretation

1. **By return:** Baseline 1 (historical mean) showed the highest return — 16.08% annually and 338.71% over the entire period.

2. **By risk:** PatchTST demonstrated the lowest risk:
   - Minimum volatility: 12.13%
   - Minimum drawdown: -15.72% (vs -24.22% for Baseline 1)

3. **By risk/return ratio:**
   - Sharpe Ratio: Baseline 1 leads (0.93)
   - Calmar Ratio: PatchTST leads (0.78) — best return to maximum drawdown ratio

---

## 3. Forecast Quality Metrics

| Metric | Baseline 1 | StatsForecast | PatchTST |
|--------|------------|---------------|----------|
| **RMSE** | **0.0695** | 0.0698 | 0.0768 |
| **MAE** | **0.0506** | 0.0510 | 0.0559 |
| **Hit Rate** | **56.13%** | 52.77% | 51.22% |

### Interpretation

- All three methods show similar forecast accuracy (RMSE ≈ 0.07)
- Hit Rate (correct direction proportion) is slightly above 50% — financial series are difficult to forecast
- Baseline 1 has the best forecast metrics despite its simplicity

---

## 4. Key Findings

### 4.1. Hypothesis Not Confirmed

**Original hypothesis:** Replacing historical means with PatchTST forecasts will improve Markowitz portfolio quality.

**Result:** The hypothesis **was not confirmed** in this experiment. The classical approach (historical mean) showed better results by Sharpe Ratio and total return.

### 4.2. PatchTST Advantages

Despite lower returns, PatchTST showed important advantages in risk management:

| Risk Metric | Baseline 1 | PatchTST | Improvement |
|-------------|------------|----------|-------------|
| Max Drawdown | -24.22% | -15.72% | +35% |
| Volatility | 12.83% | 12.13% | +5% |
| Calmar Ratio | 0.66 | 0.78 | +18% |

**Conclusion:** PatchTST forms more conservative portfolios with smaller drawdowns.

### 4.3. Possible Reasons for Results

1. **Difficulty in forecasting financial series**
   - Financial time series contain high noise levels
   - Historical mean may be a robust estimate under high uncertainty

2. **"Regression to mean" effect**
   - Historical mean smooths outliers
   - ML models may overfit to noise

3. **Model parameters**
   - Longer training may be required
   - Hyperparameters may not be optimal for financial data

4. **2015-2024 period characteristics**
   - Includes COVID-19 crisis (2020)
   - High volatility complicates forecasting

---

## 5. Recommendations for Further Research

1. **Increase PatchTST training period**
   - Increase pretrain_epochs and finetune_epochs
   - Use more data for pretraining

2. **Add additional features**
   - Macroeconomic indicators
   - Technical indicators
   - Sentiment data

3. **Explore other architectures**
   - Compare with Informer, Autoformer, FEDformer
   - Try supervised PatchTST mode

4. **Use ensemble methods**
   - Combination of Historical Mean + PatchTST
   - Weighted average of forecasts

5. **Test on other data**
   - Other markets (European, Asian)
   - Cryptocurrencies (more volatile)
   - Commodity markets

---

## 6. Practical Implications

For practical application, the research results show:

| Investor Goal | Recommended Approach |
|---------------|---------------------|
| Maximum return | Baseline 1 (historical mean) |
| Minimum drawdown | PatchTST |
| Risk/return balance | Baseline 1 or combination |

**For conservative investors** PatchTST may be preferable due to smaller drawdowns (-15.72% vs -24.22%).

**For aggressive investors** historical mean provides higher returns.

---

## 7. Summary Results Table

```
============================================================
PORTFOLIO METRICS (119 periods, 2015-2024)
============================================================

Metric                     Baseline 1      StatsF    PatchTST
-------------------------------------------------------------
Annual Return                  16.08%      13.08%      12.20%
Annual Volatility              12.83%      13.26%      12.13%
Sharpe Ratio                     0.93        0.70        0.69
Calmar Ratio                     0.66        0.63        0.78
Max Drawdown                  -24.22%     -20.72%     -15.72%
Total Return                  338.71%     238.30%     213.03%

============================================================
FORECAST METRICS
============================================================

Metric                     Baseline 1      StatsF    PatchTST
-------------------------------------------------------------
RMSE                         0.069522    0.069783    0.076804
MAE                          0.050634    0.051043    0.055938
Hit Rate                       56.13%      52.77%      51.22%
```

---

## Conclusion

The study showed that in the context of Markowitz portfolio optimization, the classical approach using historical mean for expected return estimation outperforms more complex forecasting methods (AutoARIMA and PatchTST) by Sharpe Ratio.

However, PatchTST demonstrates a significant advantage in risk management, forming portfolios with smaller drawdowns and volatility. This makes the approach attractive for conservative investors focused on capital preservation.

The results are consistent with the well-known observation in financial literature about the difficulty of return forecasting and the effectiveness of simple expected return estimation methods.
