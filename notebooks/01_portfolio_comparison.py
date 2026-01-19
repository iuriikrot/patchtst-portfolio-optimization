# %% [markdown]
# # Сравнение подходов к оценке ожидаемой доходности в портфельной оптимизации Марковица
#
# **Три подхода:**
# 1. Baseline 1: Историческое среднее
# 2. Baseline 2: StatsForecast AutoARIMA прогноз
# 3. PatchTST Self-Supervised прогноз
#
# **Источник PatchTST:** https://github.com/yuqinie98/PatchTST

# %% [markdown]
# ## 0. Конфигурация
# Все параметры эксперимента в одном месте

# %%
#==============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# Измените параметры здесь перед запуском
#==============================================================================

CONFIG = {
    # Данные
    'data': {
        'tickers': [
            "AAPL",  # Technology
            "MSFT",  # Technology
            "JNJ",   # Healthcare
            "UNH",   # Healthcare
            "JPM",   # Financials
            "WFC",   # Financials (Wells Fargo)
            "XOM",   # Energy
            "CVX",   # Energy
            "PG",    # Consumer Staples
            "KO",    # Consumer Staples
        ],
        'start_date': "2000-01-01",
        'end_date': "2025-01-01",
    },

    # Параметры бэктеста
    'backtest': {
        'train_window': 1260,  # 5 лет (252 * 5)
        'test_window': 21,     # 1 месяц
    },

    # Оптимизация Марковица
    'optimization': {
        'risk_free_rate': 0.02,  # 2% годовых
        'constraints': {
            'long_only': True,      # Только длинные позиции
            'fully_invested': True, # Полное инвестирование
            'max_weight': 0.4,      # Максимальный вес одного актива
            'min_weight': 0.0,      # Минимальный вес
            'gross_exposure': 1.5,  # sum(|w|) <= L (используется только при long_only=false)
        },
        'covariance': 'sample',    # sample | ledoit_wolf
    },

    # Модели
    'models': {
        # ARIMA(p, d, q) с автоматическим подбором (StatsForecast AutoARIMA):
        #   p — порядок авторегрессии (AR), зависимость от прошлых значений
        #   d — порядок дифференцирования (I), для приведения к стационарности
        #   q — порядок скользящего среднего (MA), зависимость от прошлых ошибок
        # max_d: верхняя граница для d
        # stepwise=True: умный поиск вместо полного перебора (~10x быстрее)
        'arima': {
            'max_p': 3,  # AR: r_t зависит от r_{t-1}, r_{t-2}, r_{t-3}
            'max_d': 0,  # Максимальный порядок дифференцирования
            'max_q': 3,  # MA: r_t зависит от ε_{t-1}, ε_{t-2}, ε_{t-3}
            'stepwise': True,  # Умный поиск (не полный перебор)
        },

        # PatchTST Self-Supervised
        'patchtst': {
            # Режим: 'fast' для отладки, 'full' для финальных результатов
            'mode': 'full',

            # Режим отладки (~10 минут, проверка что код работает)
            'fast': {
                'input_length': 252,     # 1 год (даёт 988 примеров для fine-tuning)
                'pred_length': 21,
                'patch_length': 21,      # Больше → меньше патчей
                'stride': 21,            # Без перекрытия → 12 патчей
                'd_model': 64,           # Уменьшено для скорости
                'n_heads': 4,            # Уменьшено для скорости
                'n_layers': 2,           # Уменьшено для скорости
                'd_ff': 256,             # Уменьшено для скорости
                'dropout': 0.2,
                'use_revin': True,
                'mask_ratio': 0.4,
                'pretrain_epochs': 3,    # Минимум для проверки
                'finetune_epochs': 3,    # Fine-tuning prediction head
                'pretrain_lr': 0.0001,
                'batch_size': 64,
            },

            # Полный режим (официальные параметры Self-Supervised PatchTST)
            # Источник: github.com/yuqinie98/PatchTST/PatchTST_self_supervised/patchtst_pretrain.py
            # input_length=252 (1 год) даёт: 988 примеров, 30 патчей
            'full': {
                'input_length': 252,     # 1 год (официальный подход: input << train_window)
                'pred_length': 21,
                'patch_length': 16,      # Официальное
                'stride': 8,             # Официальное → 30 патчей
                'd_model': 128,          # Официальное
                'n_heads': 16,           # Официальное
                'n_layers': 3,           # Официальное
                'd_ff': 512,             # Официальное
                'dropout': 0.2,          # Официальное
                'use_revin': True,
                'mask_ratio': 0.4,       # Официальное
                'pretrain_epochs': 10,   # Официальное (self-supervised)
                'finetune_epochs': 5,    # Fine-tuning prediction head
                'pretrain_lr': 0.0001,   # Официальное (1e-4)
                'batch_size': 64,        # Официальное
            },
        },
    },
}

# Извлекаем параметры для удобства
TICKERS = CONFIG['data']['tickers']
START_DATE = CONFIG['data']['start_date']
END_DATE = CONFIG['data']['end_date']

TRAIN_WINDOW = CONFIG['backtest']['train_window']
TEST_WINDOW = CONFIG['backtest']['test_window']
RF = CONFIG['optimization']['risk_free_rate']
CONSTRAINTS = CONFIG['optimization'].get('constraints', {})
MIN_WEIGHT = CONSTRAINTS.get('min_weight', 0.0)
MAX_WEIGHT = CONSTRAINTS.get('max_weight', 1.0)
LONG_ONLY = CONSTRAINTS.get('long_only', True)
FULLY_INVESTED = CONSTRAINTS.get('fully_invested', True)
COV_METHOD = CONFIG['optimization'].get('covariance', 'sample')
GROSS_EXPOSURE = CONSTRAINTS.get('gross_exposure')

# ARIMA
ARIMA_MAX_P = CONFIG['models']['arima']['max_p']
ARIMA_MAX_D = CONFIG['models']['arima']['max_d']
ARIMA_MAX_Q = CONFIG['models']['arima']['max_q']
ARIMA_STEPWISE = CONFIG['models']['arima']['stepwise']

# PatchTST - выбираем режим
PATCHTST_MODE = CONFIG['models']['patchtst']['mode']
patchtst_config = CONFIG['models']['patchtst'][PATCHTST_MODE]

INPUT_LEN = patchtst_config['input_length']
PRED_LEN = patchtst_config['pred_length']
PATCH_LEN = patchtst_config['patch_length']
STRIDE = patchtst_config['stride']
D_MODEL = patchtst_config['d_model']
N_HEADS = patchtst_config['n_heads']
N_LAYERS = patchtst_config['n_layers']
D_FF = patchtst_config['d_ff']
DROPOUT = patchtst_config['dropout']
USE_REVIN = patchtst_config['use_revin']
MASK_RATIO = patchtst_config['mask_ratio']
PRETRAIN_EPOCHS = patchtst_config['pretrain_epochs']
FINETUNE_EPOCHS = patchtst_config['finetune_epochs']
PRETRAIN_LR = patchtst_config['pretrain_lr']
BATCH_SIZE = patchtst_config['batch_size']

print("="*60)
print("КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА")
print("="*60)
print(f"Данные: {len(TICKERS)} акций, {START_DATE} — {END_DATE}")
print(f"Бэктест: train={TRAIN_WINDOW} дней, test={TEST_WINDOW} дней")
print(f"PatchTST режим: {PATCHTST_MODE.upper()}")
print(f"  - патчей: {(INPUT_LEN - PATCH_LEN) // STRIDE + 1}")
print(f"  - d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")
print(f"  - pretrain_epochs={PRETRAIN_EPOCHS}, finetune_epochs={FINETUNE_EPOCHS}")
print("="*60)

# %% [markdown]
# ## 1. Установка и импорты

# %%
!pip install yfinance statsforecast torch -q

# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('seaborn-v0_8-whitegrid')

# Проверка GPU (MPS -> CUDA -> CPU)
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"PyTorch device: {device}")

# %% [markdown]
# ## 2. Загрузка данных

# %%
print("Скачивание данных с Yahoo Finance...")
data = yf.download(
    tickers=TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False
)

prices = data["Adj Close"]
prices = prices.dropna()

log_returns = np.log(prices / prices.shift(1))
log_returns = log_returns.dropna()

print(f"Загружено {len(prices)} торговых дней")
print(f"Период: {prices.index[0].date()} — {prices.index[-1].date()}")
print(f"Рассчитано {len(log_returns)} дневных доходностей")

# %%
# Визуализация
normalized_prices = prices / prices.iloc[0] * 100

plt.figure(figsize=(14, 7))
for ticker in TICKERS:
    plt.plot(normalized_prices[ticker], label=ticker, alpha=0.8)

plt.title("Динамика цен акций (нормализовано к 100)")
plt.xlabel("Дата")
plt.ylabel("Цена (нормализованная)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Вспомогательные функции

# %%
def maximize_sharpe(
    mu,
    cov,
    rf=0.02,
    min_weight=0.0,
    max_weight=1.0,
    long_only=True,
    fully_invested=True
):
    """Максимизация коэффициента Шарпа."""
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        if port_vol < 1e-10:
            return 1e10  # Избегаем деления на ноль
        return -(port_return - rf) / port_vol

    if long_only and min_weight < 0:
        min_weight = 0.0

    constraints = []
    if fully_invested:
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(min_weight, max_weight) for _ in range(n)]

    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def calculate_metrics(returns, rf=0.02):
    """Расчёт метрик портфеля (returns — месячные лог-доходности)."""
    simple_returns = np.exp(returns) - 1
    monthly_rf = (1 + rf) ** (1 / 12) - 1
    excess = simple_returns - monthly_rf

    if len(simple_returns) > 0:
        annual_return = (1 + simple_returns).prod() ** (12 / len(simple_returns)) - 1
    else:
        annual_return = 0
    annual_vol = simple_returns.std() * np.sqrt(12)
    sharpe = (excess.mean() / simple_returns.std() * np.sqrt(12)) if simple_returns.std() > 0 else 0

    cumulative = (1 + simple_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    total_return = (1 + simple_returns).prod() - 1

    # Calmar Ratio = Annual Return / |Max Drawdown|
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown,
        'Total Return': total_return,
        'Num Periods': len(returns)
    }


def compute_monthly_log_return(test_data, weights, fully_invested=True):
    """Доходность за месяц при ребалансировке раз в месяц (buy-and-hold)."""
    asset_gross = np.exp(test_data.sum(axis=0).values)
    portfolio_gross = np.dot(weights, asset_gross)
    if not fully_invested:
        portfolio_gross += (1 - weights.sum())
    return np.log(portfolio_gross)


def compute_covariance(returns, method="sample", annualize=252):
    """Оценка ковариации."""
    if method == "sample":
        cov = returns.cov().values
    elif method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
    else:
        raise ValueError(f"Неизвестный метод ковариации: {method}")
    return cov * annualize


def calculate_forecast_metrics(actual, predicted):
    """
    Рассчитать метрики качества прогноза.

    Args:
        actual: array — фактические месячные доходности по тикерам
        predicted: array — прогнозные месячные доходности по тикерам

    Returns:
        dict с метриками: RMSE, MAE, Hit Rate
    """
    # Убираем NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'hit_rate': np.nan}

    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # MAE
    mae = np.mean(np.abs(actual - predicted))

    # Hit Rate (совпадение знаков)
    nonzero_mask = (actual != 0) & (predicted != 0)
    if nonzero_mask.sum() > 0:
        hits = np.sign(actual[nonzero_mask]) == np.sign(predicted[nonzero_mask])
        hit_rate = hits.mean()
    else:
        hit_rate = np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'hit_rate': hit_rate
    }


def aggregate_forecast_metrics(forecasts_df):
    """
    Агрегировать метрики по всем периодам бэктеста.

    Args:
        forecasts_df: DataFrame с колонками:
            - date: дата ребалансировки
            - ticker: тикер
            - actual: фактическая месячная доходность
            - predicted: прогнозная месячная доходность

    Returns:
        dict с агрегированными метриками
    """
    actual = forecasts_df['actual'].values
    predicted = forecasts_df['predicted'].values

    return calculate_forecast_metrics(actual, predicted)

# %% [markdown]
# ## 4. Baseline 1: Историческое среднее

# %%
def run_backtest_baseline1(returns, train_window, test_window, rf, collect_forecasts=True, collect_weights=False):
    """Бэктест: μ = историческое среднее."""
    n = len(returns)
    portfolio_returns = []
    dates = []
    forecast_records = [] if collect_forecasts else None
    weights_list = [] if collect_weights else None

    total_steps = (n - train_window - test_window) // test_window + 1
    i = 0
    step = 0

    while i + train_window + test_window <= n:
        train_data = returns.iloc[i:i + train_window]
        test_data = returns.iloc[i + train_window:i + train_window + test_window]

        daily_mean = train_data.mean()
        mu = daily_mean.values * 252
        cov = compute_covariance(train_data, method=COV_METHOD, annualize=252)

        # Собираем прогнозы если нужно
        if collect_forecasts:
            # Для Baseline 1: прогноз = историческое среднее на каждый день
            raw_forecasts = pd.DataFrame(
                np.tile(daily_mean.values, (test_window, 1)),
                columns=returns.columns
            )
            # Actual = сумма дневных доходностей за месяц
            actual_monthly = test_data.sum(axis=0)
            # Predicted = сумма прогнозов за месяц
            predicted_monthly = raw_forecasts.sum(axis=0)
            # Собираем записи для каждого тикера
            for ticker in returns.columns:
                forecast_records.append({
                    'date': test_data.index[0],
                    'ticker': ticker,
                    'actual': actual_monthly[ticker],
                    'predicted': predicted_monthly[ticker],
                    'model': 'Historical'
                })

        weights = maximize_sharpe(
            mu,
            cov,
            rf=rf,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            long_only=LONG_ONLY,
            fully_invested=FULLY_INVESTED,
            gross_exposure=GROSS_EXPOSURE
        )

        month_return = compute_monthly_log_return(
            test_data,
            weights,
            fully_invested=FULLY_INVESTED
        )

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        if weights_list is not None:
            weights_list.append(weights)

        step += 1
        i += test_window

        if step % 50 == 0:
            print(f"  Шаг {step}/{total_steps} ({step*100//total_steps}%)")

    print(f"  Завершено: {step} периодов")
    result = pd.Series(portfolio_returns, index=dates)

    # Возврат результатов
    returns_tuple = [result]
    if collect_forecasts:
        forecasts_df = pd.DataFrame(forecast_records)
        returns_tuple.append(forecasts_df)
    if collect_weights:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        returns_tuple.append(weights_df)

    return tuple(returns_tuple) if len(returns_tuple) > 1 else result

# %%
print("="*50)
print("Baseline 1: Историческое среднее")
print("="*50)
baseline1_returns, baseline1_forecasts, baseline1_weights = run_backtest_baseline1(
    log_returns, TRAIN_WINDOW, TEST_WINDOW, RF, collect_forecasts=True, collect_weights=True
)
baseline1_metrics = calculate_metrics(baseline1_returns, rf=RF)
baseline1_forecast_metrics = aggregate_forecast_metrics(baseline1_forecasts)

print("\nРезультаты (Портфель):")
for name, value in baseline1_metrics.items():
    if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
        print(f"  {name}: {value:.2%}")
    elif 'Ratio' in name:
        print(f"  {name}: {value:.2f}")
    else:
        print(f"  {name}: {value}")

print("\nМетрики прогнозов:")
print(f"  RMSE: {baseline1_forecast_metrics['rmse']:.6f}")
print(f"  MAE: {baseline1_forecast_metrics['mae']:.6f}")
print(f"  Hit Rate: {baseline1_forecast_metrics['hit_rate']:.2%}")

# %% [markdown]
# ## 5. Baseline 2: StatsForecast AutoARIMA

# %%
def build_long_frame(returns):
    """Преобразование в long-формат для StatsForecast."""
    df_long = returns.stack().reset_index()
    df_long.columns = ['ds', 'unique_id', 'y']
    return df_long


def forecast_returns_statsforecast(train_returns, horizon, max_p, max_d, max_q, stepwise=True, return_raw=False):
    """Прогноз доходностей с помощью StatsForecast AutoARIMA."""
    df_long = build_long_frame(train_returns)
    fallback = train_returns.mean()

    model = AutoARIMA(
        max_p=max_p,
        max_q=max_q,
        d=None,
        max_d=max_d,
        seasonal=False,
        stepwise=stepwise
    )
    sf = StatsForecast(models=[model], freq='B', n_jobs=1)

    try:
        forecast_df = sf.forecast(h=horizon, df=df_long)
    except Exception:
        if return_raw:
            # Возвращаем константу fallback для всех дней
            raw = pd.DataFrame(
                np.tile(fallback.values, (horizon, 1)),
                columns=train_returns.columns
            )
            return fallback.values * 252, raw
        return fallback.values * 252

    col_name = 'AutoARIMA'
    if col_name not in forecast_df.columns:
        if return_raw:
            raw = pd.DataFrame(
                np.tile(fallback.values, (horizon, 1)),
                columns=train_returns.columns
            )
            return fallback.values * 252, raw
        return fallback.values * 252

    # Годовая доходность
    preds = forecast_df.groupby('unique_id')[col_name].mean()
    preds = preds.reindex(train_returns.columns).fillna(fallback)
    mu = preds.values * 252

    if return_raw:
        # Raw прогнозы: pivot для каждого дня
        raw = forecast_df.pivot(index='ds', columns='unique_id', values=col_name)
        raw = raw.reindex(columns=train_returns.columns).fillna(fallback)
        return mu, raw

    return mu


def run_backtest_statsforecast(returns, train_window, test_window, rf, max_p, max_d, max_q, stepwise=True, collect_forecasts=True, collect_weights=False):
    """Бэктест: μ = прогноз StatsForecast AutoARIMA."""
    n = len(returns)
    portfolio_returns = []
    dates = []
    forecast_records = [] if collect_forecasts else None
    weights_list = [] if collect_weights else None

    total_steps = (n - train_window - test_window) // test_window + 1
    i = 0
    step = 0

    while i + train_window + test_window <= n:
        train_data = returns.iloc[i:i + train_window]
        test_data = returns.iloc[i + train_window:i + train_window + test_window]

        step += 1

        # μ из ARIMA прогнозов
        if collect_forecasts:
            mu, raw_forecasts = forecast_returns_statsforecast(
                train_data, test_window, max_p, max_d, max_q, stepwise, return_raw=True
            )
            # Actual = сумма дневных доходностей за месяц
            actual_monthly = test_data.sum(axis=0)
            # Predicted = сумма прогнозов за месяц
            predicted_monthly = raw_forecasts.sum(axis=0)
            # Собираем записи для каждого тикера
            for ticker in returns.columns:
                forecast_records.append({
                    'date': test_data.index[0],
                    'ticker': ticker,
                    'actual': actual_monthly[ticker],
                    'predicted': predicted_monthly[ticker],
                    'model': 'StatsForecast'
                })
        else:
            mu = forecast_returns_statsforecast(train_data, test_window, max_p, max_d, max_q, stepwise)

        cov = compute_covariance(train_data, method=COV_METHOD, annualize=252)
        weights = maximize_sharpe(
            mu,
            cov,
            rf=rf,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            long_only=LONG_ONLY,
            fully_invested=FULLY_INVESTED,
            gross_exposure=GROSS_EXPOSURE
        )

        month_return = compute_monthly_log_return(
            test_data,
            weights,
            fully_invested=FULLY_INVESTED
        )

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        if weights_list is not None:
            weights_list.append(weights)

        if step % 10 == 0 or step == 1:
            pct = step * 100 // total_steps
            print(f"  Шаг {step:3d}/{total_steps} ({pct:2d}%) | Дата: {test_data.index[0].date()}")

        i += test_window

    print(f"  Завершено: {step} периодов")
    result = pd.Series(portfolio_returns, index=dates)

    # Возврат результатов
    returns_tuple = [result]
    if collect_forecasts:
        forecasts_df = pd.DataFrame(forecast_records)
        returns_tuple.append(forecasts_df)
    if collect_weights:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        returns_tuple.append(weights_df)

    return tuple(returns_tuple) if len(returns_tuple) > 1 else result

# %%
print("="*50)
print("Baseline 2: StatsForecast AutoARIMA")
print("="*50)
print(f"Параметры: max_p={ARIMA_MAX_P}, max_d={ARIMA_MAX_D}, max_q={ARIMA_MAX_Q}, stepwise={ARIMA_STEPWISE}")
print("(stepwise=True ускоряет подбор ~10x)\n")

baseline2_returns, baseline2_forecasts, baseline2_weights = run_backtest_statsforecast(
    log_returns, TRAIN_WINDOW, TEST_WINDOW, RF, ARIMA_MAX_P, ARIMA_MAX_D, ARIMA_MAX_Q, ARIMA_STEPWISE,
    collect_forecasts=True, collect_weights=True
)
baseline2_metrics = calculate_metrics(baseline2_returns, rf=RF)
baseline2_forecast_metrics = aggregate_forecast_metrics(baseline2_forecasts)

print("\nРезультаты (Портфель):")
for name, value in baseline2_metrics.items():
    if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
        print(f"  {name}: {value:.2%}")
    elif 'Ratio' in name:
        print(f"  {name}: {value:.2f}")
    else:
        print(f"  {name}: {value}")

print("\nМетрики прогнозов:")
print(f"  RMSE: {baseline2_forecast_metrics['rmse']:.6f}")
print(f"  MAE: {baseline2_forecast_metrics['mae']:.6f}")
print(f"  Hit Rate: {baseline2_forecast_metrics['hit_rate']:.2%}")

# %% [markdown]
# ## 6. PatchTST Self-Supervised

# %%
# ============================================================
# PatchTST Official Architecture
# (Based on https://github.com/yuqinie98/PatchTST)
# Key features: Residual Attention, BatchNorm, Official heads
# ============================================================

class Transpose(nn.Module):
    """Transpose for BatchNorm."""
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available')


def positional_encoding(pe, learn_pe, q_len, d_model):
    """Official positional encoding with uniform initialization."""
    if pe == 'zeros' or pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'sincos':
        W_pos = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        W_pos[:, 0::2] = torch.sin(position * div_term)
        W_pos[:, 1::2] = torch.cos(position * div_term)
        W_pos = W_pos - W_pos.mean()
        W_pos = W_pos / (W_pos.std() * 10)
    else:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            x = x * self.std + self.mean
        return x


class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention with Residual Attention support.
    Residual Attention: scores are passed between layers for better
    modeling of long sequences.
    """
    def __init__(self, d_model, n_heads, d_k=None, d_v=None,
                 attn_dropout=0., proj_dropout=0., res_attention=False):
        super().__init__()
        d_k = d_k or d_model // n_heads
        d_v = d_v or d_model // n_heads
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.res_attention = res_attention

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_O = nn.Linear(d_v * n_heads, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.scale = d_k ** -0.5

    def forward(self, Q, K, V, prev=None):
        bs, q_len, _ = Q.shape
        _, k_len, _ = K.shape

        Q = self.W_Q(Q).view(bs, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(bs, k_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(bs, k_len, self.n_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Residual Attention: add previous scores
        if prev is not None:
            scores = scores + prev

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(bs, q_len, -1)
        output = self.W_O(context)
        output = self.proj_dropout(output)

        if self.res_attention:
            return output, attn, scores
        return output, attn


class TSTEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with official architecture:
    - BatchNorm instead of LayerNorm (default)
    - Residual Attention support
    - Pre/Post norm option
    """
    def __init__(self, d_model, n_heads, d_ff=256,
                 norm='BatchNorm', attn_dropout=0., dropout=0.,
                 activation='gelu', res_attention=False, pre_norm=False):
        super().__init__()
        assert d_model % n_heads == 0
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.res_attention = res_attention
        self.pre_norm = pre_norm

        self.self_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention
        )

        self.dropout_attn = nn.Dropout(dropout)

        # Normalization after attention
        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout_ffn = nn.Dropout(dropout)

        # Normalization after FFN
        if 'batch' in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None):
        if self.pre_norm:
            src2 = self.norm_attn(src)
        else:
            src2 = src

        if self.res_attention:
            src2, attn, scores = self.self_attn(src2, src2, src2, prev)
        else:
            src2, attn = self.self_attn(src2, src2, src2)
            scores = None

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src2 = self.ff(self.norm_ffn(src))
        else:
            src2 = self.ff(src)

        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        return src


class TSTEncoder(nn.Module):
    """Stack of TSTEncoderLayers."""
    def __init__(self, d_model, n_heads, d_ff=256,
                 norm='BatchNorm', attn_dropout=0., dropout=0.,
                 activation='gelu', res_attention=False, pre_norm=False,
                 n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads, d_ff, norm, attn_dropout, dropout,
                           activation, res_attention, pre_norm)
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src):
        output = src
        scores = None
        if self.res_attention:
            for layer in self.layers:
                output, scores = layer(output, prev=scores)
        else:
            for layer in self.layers:
                output = layer(output)
        return output


class PretrainHead(nn.Module):
    """Head for self-supervised pretraining (official: dropout BEFORE linear)."""
    def __init__(self, d_model, patch_len, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        return self.linear(self.dropout(x))


class PredictionHead(nn.Module):
    """Head for prediction (official: single linear layer)."""
    def __init__(self, d_model, num_patches, pred_len, dropout=0.):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * num_patches, pred_len)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.linear(x)


class PatchTST_SelfSupervised(nn.Module):
    """
    PatchTST with Official Architecture.
    Key features:
    - Residual Attention
    - BatchNorm
    - Official head structure
    - Self-Supervised pretraining
    """
    def __init__(self, input_len, pred_len, patch_len, stride,
                 d_model, n_heads, n_layers, d_ff, dropout,
                 mask_ratio, use_revin,
                 # Official parameters
                 norm='BatchNorm', res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, attn_dropout=0.):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.use_revin = use_revin
        self.d_model = d_model
        self.num_patches = (input_len - patch_len) // stride + 1

        # RevIN
        if use_revin:
            self.revin = RevIN(1, affine=True)

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding (official)
        self.W_pos = positional_encoding(pe, learn_pe, self.num_patches, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder (official architecture)
        self.encoder = TSTEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            activation='gelu', res_attention=res_attention,
            pre_norm=pre_norm, n_layers=n_layers
        )

        # Heads (official architecture)
        self.pretrain_head = PretrainHead(d_model, patch_len, dropout)
        self.prediction_head = PredictionHead(d_model, self.num_patches, pred_len, dropout)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def create_patches(self, x):
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            patches.append(x[:, start:start + self.patch_len])
        return torch.stack(patches, dim=1)

    def random_masking(self, patches):
        bs, num_patches, _ = patches.shape
        num_mask = int(num_patches * self.mask_ratio)
        noise = torch.rand(bs, num_patches, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.zeros(bs, num_patches, device=patches.device)
        mask[:, :num_mask] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()
        return mask

    def forward_encoder(self, x, mask=None):
        bs = x.shape[0]

        # RevIN normalization
        if self.use_revin:
            x = x.unsqueeze(-1)
            x = self.revin(x, 'norm')
            x = x.squeeze(-1)

        # Create patches
        patches = self.create_patches(x)

        # Embedding
        x = self.patch_embedding(patches)

        # Apply mask
        if mask is not None:
            mask_tokens = self.mask_token.expand(bs, self.num_patches, -1)
            x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        # Positional encoding
        x = self.dropout(x + self.W_pos)

        # Transformer encoder
        x = self.encoder(x)

        return x, patches

    def forward_pretrain(self, x):
        with torch.no_grad():
            patches_for_mask = self.create_patches(x)
            mask = self.random_masking(patches_for_mask)

        encoded, original_patches = self.forward_encoder(x, mask)
        pred_patches = self.pretrain_head(encoded)
        loss = F.mse_loss(pred_patches[mask], original_patches[mask])
        return loss, pred_patches, mask

    def forward_predict(self, x):
        encoded, _ = self.forward_encoder(x, mask=None)
        prediction = self.prediction_head(encoded)

        # RevIN denormalization
        if self.use_revin and hasattr(self.revin, 'std'):
            prediction = prediction * self.revin.std.squeeze(-1) + self.revin.mean.squeeze(-1)

        return prediction

    def forward(self, x, mode='predict'):
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        return self.forward_predict(x)


def pretrain_patchtst(model, data, epochs, lr, batch_size, verbose=False):
    """Self-Supervised pre-training."""
    model.train()
    input_len = model.input_len

    step = 5
    X_train = []
    for i in range(0, len(data) - input_len + 1, step):
        X_train.append(data[i:i + input_len])
    if len(X_train) == 0:
        X_train = [data[-input_len:]]

    X_train = np.array(X_train)
    X_tensor = torch.FloatTensor(X_train).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss, _, _ = model(batch, mode='pretrain')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model


def forecast_patchtst(model, last_input):
    """Прогноз."""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(last_input).unsqueeze(0).to(device)
        pred = model(x, mode='predict')
    return pred.cpu().numpy().flatten()


def create_sequences(data, input_len, pred_len):
    """Создание пар (вход, выход) для fine-tuning."""
    X, y = [], []
    for i in range(len(data) - input_len - pred_len + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + pred_len])
    return np.array(X), np.array(y)


def finetune_patchtst(model, X_train, y_train, epochs, lr, batch_size, verbose=False):
    """Fine-tuning prediction head."""
    model.train()

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch, mode='predict')
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Finetune Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model


def forecast_returns_patchtst(train_returns, config, return_raw=False):
    """Прогноз доходностей с помощью PatchTST."""
    tickers = train_returns.columns
    fallback = train_returns.mean()
    horizon = config['pred_length']

    # Собираем raw прогнозы для всех тикеров
    raw_forecasts = pd.DataFrame(index=range(horizon), columns=tickers, dtype=float)

    for ticker in tickers:
        series = train_returns[ticker].values

        if len(series) < config['input_length']:
            # Мало данных — берём историческое среднее (константа на все дни)
            raw_forecasts[ticker] = fallback[ticker]
            continue

        model = PatchTST_SelfSupervised(
            input_len=config['input_length'],
            pred_len=config['pred_length'],
            patch_len=config['patch_length'],
            stride=config['stride'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            mask_ratio=config['mask_ratio'],
            use_revin=config['use_revin']
        ).to(device)

        model = pretrain_patchtst(
            model, series,
            epochs=config['pretrain_epochs'],
            lr=config['pretrain_lr'],
            batch_size=config['batch_size'],
            verbose=False
        )

        # Fine-tuning: обучаем prediction head (официальный подход)
        X_train, y_train = create_sequences(series, config['input_length'], config['pred_length'])
        if len(X_train) > 0:
            model = finetune_patchtst(
                model, X_train, y_train,
                epochs=config['finetune_epochs'],
                lr=config['pretrain_lr'] * 0.1,  # Меньший lr для fine-tuning
                batch_size=config['batch_size'],
                verbose=False
            )

        last_input = series[-config['input_length']:]
        forecast = forecast_patchtst(model, last_input)

        # Сохраняем raw прогноз (все horizon дней)
        if len(forecast) == horizon:
            raw_forecasts[ticker] = forecast
        else:
            # Fallback если прогноз другой длины
            raw_forecasts[ticker] = fallback[ticker]

    # mu = среднее по дням × 252
    mu = raw_forecasts.mean(axis=0).values * 252

    if return_raw:
        return mu, raw_forecasts
    return mu


def run_backtest_patchtst(returns, train_window, test_window, rf, config, collect_forecasts=True, collect_weights=False):
    """Бэктест: μ = прогноз PatchTST."""
    n = len(returns)
    portfolio_returns = []
    dates = []
    forecast_records = [] if collect_forecasts else None
    weights_list = [] if collect_weights else None

    total_steps = (n - train_window - test_window) // test_window + 1
    i = 0
    step = 0

    while i + train_window + test_window <= n:
        train_data = returns.iloc[i:i + train_window]
        test_data = returns.iloc[i + train_window:i + train_window + test_window]

        step += 1

        # μ из PatchTST прогнозов
        if collect_forecasts:
            mu, raw_forecasts = forecast_returns_patchtst(train_data, config, return_raw=True)
            # Actual = сумма дневных доходностей за месяц
            actual_monthly = test_data.sum(axis=0)
            # Predicted = сумма прогнозов за месяц
            predicted_monthly = raw_forecasts.sum(axis=0)
            # Собираем записи для каждого тикера
            for ticker in returns.columns:
                forecast_records.append({
                    'date': test_data.index[0],
                    'ticker': ticker,
                    'actual': actual_monthly[ticker],
                    'predicted': predicted_monthly[ticker],
                    'model': 'PatchTST'
                })
        else:
            mu = forecast_returns_patchtst(train_data, config)

        cov = compute_covariance(train_data, method=COV_METHOD, annualize=252)
        weights = maximize_sharpe(
            mu,
            cov,
            rf=rf,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            long_only=LONG_ONLY,
            fully_invested=FULLY_INVESTED,
            gross_exposure=GROSS_EXPOSURE
        )

        month_return = compute_monthly_log_return(
            test_data,
            weights,
            fully_invested=FULLY_INVESTED
        )

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        if weights_list is not None:
            weights_list.append(weights)

        if step % 5 == 0 or step == 1:
            pct = step * 100 // total_steps
            print(f"  Шаг {step:3d}/{total_steps} ({pct:2d}%) | Дата: {test_data.index[0].date()}")

        i += test_window

    print(f"  Завершено: {step} периодов")
    result = pd.Series(portfolio_returns, index=dates)

    # Возврат результатов
    returns_tuple = [result]
    if collect_forecasts:
        forecasts_df = pd.DataFrame(forecast_records)
        returns_tuple.append(forecasts_df)
    if collect_weights:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        returns_tuple.append(weights_df)

    return tuple(returns_tuple) if len(returns_tuple) > 1 else result

# %%
print("="*60)
print(f"PatchTST Self-Supervised (режим: {PATCHTST_MODE.upper()})")
print("="*60)
print(f"Устройство: {device}")
print(f"Патчей: {(INPUT_LEN - PATCH_LEN) // STRIDE + 1}")
print(f"d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")
print(f"pretrain_epochs={PRETRAIN_EPOCHS}, finetune_epochs={FINETUNE_EPOCHS}, lr={PRETRAIN_LR}")
print()

patchtst_returns, patchtst_forecasts, patchtst_weights = run_backtest_patchtst(
    log_returns, TRAIN_WINDOW, TEST_WINDOW, RF, patchtst_config,
    collect_forecasts=True, collect_weights=True
)
patchtst_metrics = calculate_metrics(patchtst_returns, rf=RF)
patchtst_forecast_metrics = aggregate_forecast_metrics(patchtst_forecasts)

print("\nРезультаты (Портфель):")
for name, value in patchtst_metrics.items():
    if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
        print(f"  {name}: {value:.2%}")
    elif 'Ratio' in name:
        print(f"  {name}: {value:.2f}")
    else:
        print(f"  {name}: {value}")

print("\nМетрики прогнозов:")
print(f"  RMSE: {patchtst_forecast_metrics['rmse']:.6f}")
print(f"  MAE: {patchtst_forecast_metrics['mae']:.6f}")
print(f"  Hit Rate: {patchtst_forecast_metrics['hit_rate']:.2%}")

# %% [markdown]
# ## 7. Сравнение результатов

# %%
comparison_df = pd.DataFrame({
    'Метрика': ['Годовая доходность', 'Годовая волатильность', 'Коэффициент Шарпа',
                'Коэффициент Кальмара', 'Максимальная просадка', 'Общая доходность'],
    'Baseline 1 (Ист. среднее)': [
        f"{baseline1_metrics['Annual Return']:.2%}",
        f"{baseline1_metrics['Annual Volatility']:.2%}",
        f"{baseline1_metrics['Sharpe Ratio']:.2f}",
        f"{baseline1_metrics['Calmar Ratio']:.2f}",
        f"{baseline1_metrics['Max Drawdown']:.2%}",
        f"{baseline1_metrics['Total Return']:.2%}"
    ],
    'Baseline 2 (StatsForecast)': [
        f"{baseline2_metrics['Annual Return']:.2%}",
        f"{baseline2_metrics['Annual Volatility']:.2%}",
        f"{baseline2_metrics['Sharpe Ratio']:.2f}",
        f"{baseline2_metrics['Calmar Ratio']:.2f}",
        f"{baseline2_metrics['Max Drawdown']:.2%}",
        f"{baseline2_metrics['Total Return']:.2%}"
    ],
    'PatchTST': [
        f"{patchtst_metrics['Annual Return']:.2%}",
        f"{patchtst_metrics['Annual Volatility']:.2%}",
        f"{patchtst_metrics['Sharpe Ratio']:.2f}",
        f"{patchtst_metrics['Calmar Ratio']:.2f}",
        f"{patchtst_metrics['Max Drawdown']:.2%}",
        f"{patchtst_metrics['Total Return']:.2%}"
    ]
})

print("="*60)
print("СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
print("="*60)
comparison_df

# %%
# График
cumulative1 = (1 + (np.exp(baseline1_returns) - 1)).cumprod()
cumulative2 = (1 + (np.exp(baseline2_returns) - 1)).cumprod()
cumulative3 = (1 + (np.exp(patchtst_returns) - 1)).cumprod()

plt.figure(figsize=(14, 7))
plt.plot(cumulative1.index, cumulative1.values, label='Baseline 1: Историческое среднее', linewidth=2)
plt.plot(cumulative2.index, cumulative2.values, label='Baseline 2: StatsForecast', linewidth=2)
plt.plot(cumulative3.index, cumulative3.values, label='PatchTST', linewidth=2, color='green')
plt.title('Сравнение кумулятивных доходностей')
plt.xlabel('Дата')
plt.ylabel('Рост капитала ($1 → $X)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Baseline 1: $1 → ${cumulative1.iloc[-1]:.2f}")
print(f"Baseline 2: $1 → ${cumulative2.iloc[-1]:.2f}")
print(f"PatchTST:   $1 → ${cumulative3.iloc[-1]:.2f}")

# %% [markdown]
# ## 8. Анализ ребалансировки портфелей
#
# В этом разделе анализируем, как модели меняют веса активов во времени.

# %%
import seaborn as sns

# Функция для расчёта turnover (оборота портфеля)
def calculate_turnover(weights_df):
    """Turnover = среднее изменение весов между периодами."""
    diff = weights_df.diff().abs()
    turnover = diff.sum(axis=1) / 2  # Делим на 2, т.к. продажа и покупка считаются дважды
    return turnover

# Рассчитываем turnover для всех моделей
baseline1_turnover = calculate_turnover(baseline1_weights)
baseline2_turnover = calculate_turnover(baseline2_weights)
patchtst_turnover = calculate_turnover(patchtst_weights)

print("="*60)
print("АНАЛИЗ РЕБАЛАНСИРОВКИ ПОРТФЕЛЕЙ")
print("="*60)
print(f"\nBaseline 1 (Historical Mean):")
print(f"  Средний turnover: {baseline1_turnover.mean():.2%}")
print(f"  Максимум: {baseline1_turnover.max():.2%}")

print(f"\nBaseline 2 (StatsForecast):")
print(f"  Средний turnover: {baseline2_turnover.mean():.2%}")
print(f"  Максимум: {baseline2_turnover.max():.2%}")

print(f"\nPatchTST:")
print(f"  Средний turnover: {patchtst_turnover.mean():.2%}")
print(f"  Максимум: {patchtst_turnover.max():.2%}")

# %%
# Heatmaps весов для каждой модели
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Baseline 1
sns.heatmap(baseline1_weights.T, cmap='RdYlGn', center=0.1,
            vmin=0, vmax=0.25, cbar_kws={'label': 'Вес в портфеле'},
            ax=axes[0], xticklabels=20)
axes[0].set_title('Baseline 1: Веса портфеля во времени', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Тикер')
axes[0].set_xlabel('')

# Baseline 2
sns.heatmap(baseline2_weights.T, cmap='RdYlGn', center=0.1,
            vmin=0, vmax=0.25, cbar_kws={'label': 'Вес в портфеле'},
            ax=axes[1], xticklabels=20)
axes[1].set_title('Baseline 2 (StatsForecast): Веса портфеля во времени', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Тикер')
axes[1].set_xlabel('')

# PatchTST
sns.heatmap(patchtst_weights.T, cmap='RdYlGn', center=0.1,
            vmin=0, vmax=0.25, cbar_kws={'label': 'Вес в портфеле'},
            ax=axes[2], xticklabels=20)
axes[2].set_title('PatchTST: Веса портфеля во времени', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Тикер')
axes[2].set_xlabel('Период ребалансировки')

plt.tight_layout()
plt.show()

# %%
# Траектории весов для PatchTST (наиболее активная модель)
fig, ax = plt.subplots(figsize=(16, 8))

for ticker in patchtst_weights.columns:
    ax.plot(patchtst_weights.index, patchtst_weights[ticker],
            label=ticker, linewidth=1.5, alpha=0.8)

ax.set_title('PatchTST: Траектории весов активов во времени', fontsize=14, fontweight='bold')
ax.set_xlabel('Дата')
ax.set_ylabel('Вес в портфеле')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.30)
plt.tight_layout()
plt.show()

# %%
# Сравнение turnover по моделям
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(baseline1_turnover.index, baseline1_turnover.values,
        label='Baseline 1', linewidth=2, alpha=0.7)
ax.plot(baseline2_turnover.index, baseline2_turnover.values,
        label='Baseline 2 (StatsForecast)', linewidth=2, alpha=0.7)
ax.plot(patchtst_turnover.index, patchtst_turnover.values,
        label='PatchTST', linewidth=2, alpha=0.7, color='green')

ax.set_title('Сравнение оборота портфеля (Turnover)', fontsize=14, fontweight='bold')
ax.set_xlabel('Дата')
ax.set_ylabel('Turnover (доля изменённых весов)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Волатильность весов по активам
weights_volatility = pd.DataFrame({
    'Baseline 1': baseline1_weights.std(),
    'Baseline 2': baseline2_weights.std(),
    'PatchTST': patchtst_weights.std()
})

fig, ax = plt.subplots(figsize=(12, 6))
weights_volatility.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Волатильность весов по активам', fontsize=14, fontweight='bold')
ax.set_xlabel('Тикер')
ax.set_ylabel('Стандартное отклонение веса')
ax.legend(title='Модель')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nВолатильность весов по активам:")
print(weights_volatility.round(4))

# %%
# Сохранение результатов
baseline1_returns.to_csv('baseline1_returns.csv')
baseline2_returns.to_csv('baseline2_returns.csv')
patchtst_returns.to_csv('patchtst_returns.csv')
comparison_df.to_csv('comparison_results.csv', index=False)

print("Результаты сохранены")
