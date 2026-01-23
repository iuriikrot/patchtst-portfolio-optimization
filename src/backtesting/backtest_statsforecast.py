"""
Бэктест Baseline 2 (StatsForecast AutoARIMA) для оценки μ.

Требует библиотеку statsforecast:
  pip install statsforecast
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import yaml
# Подавляем шумные warnings от библиотек, но не наши собственные (UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='statsforecast')

sys.path.append(str(Path(__file__).parent.parent))

from optimization.markowitz import maximize_sharpe
from optimization.covariance import compute_covariance

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
except ImportError as exc:
    raise ImportError(
        "statsforecast не установлен. Установите: pip install statsforecast"
    ) from exc

# Загружаем конфигурацию
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Параметры из config
TRAIN_WINDOW = config['backtest']['train_window']  # 1260 дней (5 лет)
TEST_WINDOW = config['backtest']['test_window']    # 21 день (1 месяц)
RF = config['optimization']['risk_free_rate']      # 0.02
CV_METHOD = config['optimization'].get('covariance', 'sample')

CONSTRAINTS = config.get('optimization', {}).get('constraints', {})
MIN_WEIGHT = CONSTRAINTS.get('min_weight', 0.0)
MAX_WEIGHT = CONSTRAINTS.get('max_weight', 1.0)
LONG_ONLY = CONSTRAINTS.get('long_only', True)
FULLY_INVESTED = CONSTRAINTS.get('fully_invested', True)
GROSS_EXPOSURE = CONSTRAINTS.get('gross_exposure')

# ARIMA параметры из config
MAX_P = config['models']['arima']['max_p']  # 3
MAX_D = config['models']['arima']['max_d']  # 0 (лог-доходности стационарны)
MAX_Q = config['models']['arima']['max_q']  # 3
STEPWISE = config['models']['arima'].get('stepwise', True)  # Умный поиск


def build_long_frame(returns):
    """Преобразование в long-формат для StatsForecast."""
    df_long = returns.stack().reset_index()
    df_long.columns = ['ds', 'unique_id', 'y']
    return df_long


def forecast_returns_statsforecast(train_returns, horizon=21, return_raw=False):
    """
    Прогноз доходностей для всех акций с помощью StatsForecast AutoARIMA.

    Args:
        train_returns: DataFrame с доходностями (train период)
        horizon: горизонт прогноза в днях
        return_raw: если True, вернуть также raw прогнозы (horizon × N_tickers)

    Returns:
        mu: вектор ожидаемых годовых доходностей
        raw_forecasts: DataFrame (horizon × N_tickers) — если return_raw=True
    """
    df_long = build_long_frame(train_returns)
    fallback = train_returns.mean()
    tickers = train_returns.columns

    model = AutoARIMA(
        max_p=MAX_P,
        max_q=MAX_Q,
        d=None,
        max_d=MAX_D,
        seasonal=False,
        stepwise=STEPWISE
    )

    sf = StatsForecast(
        models=[model],
        freq='B',
        n_jobs=1
    )

    try:
        forecast_df = sf.forecast(h=horizon, df=df_long)
    except Exception:
        # Fallback: константный прогноз
        mu = fallback.values * 252
        if return_raw:
            raw = pd.DataFrame(
                np.tile(fallback.values, (horizon, 1)),
                columns=tickers
            )
            return mu, raw
        return mu

    col_name = 'AutoARIMA'
    if col_name not in forecast_df.columns:
        mu = fallback.values * 252
        if return_raw:
            raw = pd.DataFrame(
                np.tile(fallback.values, (horizon, 1)),
                columns=tickers
            )
            return mu, raw
        return mu

    # Собираем raw прогнозы в wide формат (horizon × N_tickers)
    # forecast_df имеет unique_id как индекс и колонки ds, AutoARIMA
    forecast_df = forecast_df.reset_index()

    # Pivot: строки = порядковый номер прогноза (0..horizon-1), колонки = тикеры
    raw_forecasts = pd.DataFrame(index=range(horizon), columns=tickers, dtype=float)

    for ticker in tickers:
        ticker_data = forecast_df[forecast_df['unique_id'] == ticker][col_name].values
        if len(ticker_data) == horizon:
            raw_forecasts[ticker] = ticker_data
        else:
            # Fallback для этого тикера
            raw_forecasts[ticker] = fallback[ticker]

    # mu = среднее по дням × 252
    preds = raw_forecasts.mean(axis=0)
    mu = preds.values * 252

    if return_raw:
        return mu, raw_forecasts
    return mu


def compute_monthly_log_return(test_data, weights, fully_invested=True):
    """Доходность за месяц при ребалансировке раз в месяц (buy-and-hold)."""
    asset_gross = np.exp(test_data.sum(axis=0).values)
    portfolio_gross = np.dot(weights, asset_gross)
    if not fully_invested:
        portfolio_gross += (1 - weights.sum())
    return np.log(portfolio_gross)


def run_backtest(returns, save_weights_path=None, collect_forecasts=False):
    """
    Бэктест со скользящим окном.

    Args:
        returns: DataFrame с лог-доходностями
        save_weights_path: путь для сохранения весов
        collect_forecasts: собирать прогнозы для расчёта forecast metrics

    Returns:
        portfolio_returns: Series с доходностями портфеля
        forecasts_df: DataFrame с прогнозами (если collect_forecasts=True)
    """
    n = len(returns)
    portfolio_returns = []
    dates = []
    weights_list = [] if save_weights_path else None
    forecast_records = [] if collect_forecasts else None

    print(f"Всего дней: {n}")
    print(f"Train окно: {TRAIN_WINDOW} дней")
    print(f"Test окно: {TEST_WINDOW} дней")
    print(f"Акций: {len(returns.columns)}")
    print(f"StatsForecast AutoARIMA: max_p={MAX_P}, max_d={MAX_D}, max_q={MAX_Q}, stepwise={STEPWISE}")
    print("\nЗапуск бэктеста...\n")

    i = 0
    step = 0

    while i + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_data = returns.iloc[i:i + TRAIN_WINDOW]
        test_data = returns.iloc[i + TRAIN_WINDOW:i + TRAIN_WINDOW + TEST_WINDOW]

        # μ из AutoARIMA прогнозов
        if collect_forecasts:
            mu, raw_forecasts = forecast_returns_statsforecast(
                train_data, horizon=TEST_WINDOW, return_raw=True
            )
            # Собираем прогнозы: actual vs predicted (месячные)
            actual_monthly = test_data.sum(axis=0)  # sum лог-доходностей за 21 день
            predicted_monthly = raw_forecasts.sum(axis=0)  # sum прогнозов за 21 день
            for ticker in returns.columns:
                forecast_records.append({
                    'date': test_data.index[0],
                    'ticker': ticker,
                    'actual': actual_monthly[ticker],
                    'predicted': predicted_monthly[ticker],
                    'model': 'ARIMA'
                })
        else:
            mu = forecast_returns_statsforecast(train_data, horizon=TEST_WINDOW)

        # Σ — ковариация (годовая)
        cov = compute_covariance(train_data, method=CV_METHOD, annualize=252)

        # Оптимизация
        weights = maximize_sharpe(
            mu,
            cov,
            rf=RF,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            long_only=LONG_ONLY,
            fully_invested=FULLY_INVESTED,
            gross_exposure=GROSS_EXPOSURE
        )

        # Доходность портфеля на test (ребалансировка раз в месяц)
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
        i += TEST_WINDOW

        if step % 10 == 0 or step == 1:
            # Топ-3 актива с наибольшими весами
            top_idx = weights.argsort()[-3:][::-1]
            top_weights = [(returns.columns[i], weights[i]) for i in top_idx]
            top_str = ", ".join([f"{ticker}:{w:.1%}" for ticker, w in top_weights])
            print(f"Шаг {step}: {test_data.index[0].date()}")
            print(f"  μ range: [{mu.min():.4f}, {mu.max():.4f}]")
            print(f"  top-3: {top_str}, std={weights.std():.3f}")

    print(f"\nЗавершено. Всего периодов: {len(portfolio_returns)}")

    if weights_list is not None:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        weights_df.to_csv(save_weights_path)

    result = pd.Series(portfolio_returns, index=dates)

    if collect_forecasts:
        forecasts_df = pd.DataFrame(forecast_records)
        return result, forecasts_df

    return result


def calculate_metrics(returns, rf=0.02):
    """Расчёт метрик."""
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


if __name__ == "__main__":
    # Загружаем данные
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "log_returns.csv"
    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
    results_path = Path(__file__).parent.parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    print("="*60)
    print("БЭКТЕСТ: StatsForecast AutoARIMA")
    print("="*60)
    print(f"Данные: {returns.index[0].date()} — {returns.index[-1].date()}")
    print()

    # Бэктест
    portfolio_returns = run_backtest(
        returns,
        save_weights_path=results_path / "statsforecast_weights.csv"
    )

    # Метрики
    metrics = calculate_metrics(portfolio_returns, rf=RF)

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ: StatsForecast AutoARIMA")
    print("="*60)
    for name, value in metrics.items():
        if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
            print(f"{name}: {value:.2%}")
        elif 'Ratio' in name:
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")
