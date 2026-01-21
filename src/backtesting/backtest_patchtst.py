"""
Бэктест с PatchTST Self-Supervised для оценки μ.

Подход:
1. Pre-training на 5-летнем окне (1260 дней) с маскированием патчей
2. Прогноз на 21 день вперёд
3. μ = mean(forecast) × 252 (годовая доходность)
4. Оптимизация портфеля по Марковицу

Согласовано с Baseline 1 и Baseline 2 по входным данным (1260 дней).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import yaml
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from optimization.markowitz import maximize_sharpe
from optimization.covariance import compute_covariance
from models.patchtst import (
    PatchTST_SelfSupervised,
    pretrain_patchtst,
    finetune_patchtst,
    forecast_patchtst,
    create_sequences
)

import torch

# Загружаем конфигурацию
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Параметры бэктеста из config
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

# Режим PatchTST из config: 'fast' или 'full'
MODE = config['models']['patchtst'].get('mode', 'fast')

# Загружаем параметры для выбранного режима
mode_config = config['models']['patchtst'][MODE]

INPUT_LEN = mode_config['input_length']
PRED_LEN = mode_config['pred_length']
PATCH_LEN = mode_config['patch_length']
STRIDE = mode_config['stride']
D_MODEL = mode_config['d_model']
N_HEADS = mode_config['n_heads']
N_LAYERS = mode_config['n_layers']
D_FF = mode_config['d_ff']
DROPOUT = mode_config['dropout']
USE_REVIN = mode_config['use_revin']
MASK_RATIO = mode_config['mask_ratio']
PRETRAIN_EPOCHS = mode_config['pretrain_epochs']
FINETUNE_EPOCHS = mode_config.get('finetune_epochs', 5)  # Fine-tuning epochs из config
PRETRAIN_LR = mode_config['pretrain_lr']
BATCH_SIZE = mode_config['batch_size']


def select_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def forecast_returns_patchtst(train_returns, horizon=21, verbose=False, return_raw=False):
    """
    Прогноз доходностей для всех акций с помощью PatchTST Self-Supervised.

    Для каждой акции:
    1. Pre-train модель на 5-летних данных (маскирование патчей)
    2. Прогноз на 21 день
    3. μ = mean(forecast) × 252

    Args:
        train_returns: DataFrame с доходностями (train период, 1260 дней)
        horizon: горизонт прогноза в днях
        verbose: выводить прогресс обучения
        return_raw: если True, вернуть также raw прогнозы (horizon × N_tickers)

    Returns:
        mu: вектор ожидаемых годовых доходностей
        raw_forecasts: DataFrame (horizon × N_tickers) — если return_raw=True
    """
    tickers = train_returns.columns
    fallback = train_returns.mean()
    device = select_device()

    # Собираем raw прогнозы для всех тикеров
    raw_forecasts = pd.DataFrame(index=range(horizon), columns=tickers, dtype=float)

    for ticker in tickers:
        series = train_returns[ticker].values

        if len(series) < INPUT_LEN:
            # Мало данных — берём историческое среднее (константа на все дни)
            raw_forecasts[ticker] = fallback[ticker]
            continue

        # Создаём модель (параметры из config)
        model = PatchTST_SelfSupervised(
            input_len=INPUT_LEN,
            pred_len=PRED_LEN,
            patch_len=PATCH_LEN,
            stride=STRIDE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            mask_ratio=MASK_RATIO,
            dropout=DROPOUT,
            use_revin=USE_REVIN
        ).to(device)

        # Pre-training с маскированием патчей
        model = pretrain_patchtst(
            model, series,
            epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            batch_size=BATCH_SIZE,
            verbose=verbose
        )

        # Fine-tuning: обучаем prediction head на supervised данных
        X_train, y_train = create_sequences(series, INPUT_LEN, PRED_LEN)
        if len(X_train) > 0:
            model = finetune_patchtst(
                model, X_train, y_train,
                epochs=FINETUNE_EPOCHS,
                lr=PRETRAIN_LR * 0.1,
                batch_size=BATCH_SIZE,
                verbose=verbose
            )

        # Прогноз на последних INPUT_LEN данных
        last_input = series[-INPUT_LEN:]
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


def run_backtest(returns, save_weights_path=None, collect_forecasts=False):
    """
    Бэктест со скользящим окном.

    Использует те же параметры что и Baseline 1 и 2:
    - TRAIN_WINDOW = 1260 дней (5 лет)
    - TEST_WINDOW = 21 день (1 месяц)

    Args:
        returns: DataFrame с лог-доходностями
        save_weights_path: путь для сохранения весов (опционально)
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

    device = select_device()
    print(f"Устройство: {device}")
    print(f"Всего дней: {n}")
    print(f"Train окно: {TRAIN_WINDOW} дней")
    print(f"Test окно: {TEST_WINDOW} дней")
    print(f"Акций: {len(returns.columns)}")
    print(f"PatchTST параметры:")
    print(f"  - input_len: {INPUT_LEN}, pred_len: {PRED_LEN}")
    print(f"  - patch_len: {PATCH_LEN}, stride: {STRIDE}")
    print(f"  - d_model: {D_MODEL}, n_heads: {N_HEADS}, n_layers: {N_LAYERS}, d_ff: {D_FF}")
    print(f"  - dropout: {DROPOUT}, use_revin: {USE_REVIN}")
    print(f"  - mask_ratio: {MASK_RATIO}")
    print(f"  - pretrain_epochs: {PRETRAIN_EPOCHS}, finetune_epochs: {FINETUNE_EPOCHS}")
    print(f"  - pretrain_lr: {PRETRAIN_LR}, batch_size: {BATCH_SIZE}")
    print("\nЗапуск бэктеста...\n")

    total_steps = (n - TRAIN_WINDOW - TEST_WINDOW) // TEST_WINDOW + 1
    i = 0
    step = 0

    while i + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_data = returns.iloc[i:i + TRAIN_WINDOW]
        test_data = returns.iloc[i + TRAIN_WINDOW:i + TRAIN_WINDOW + TEST_WINDOW]

        step += 1

        # μ из PatchTST прогнозов
        if collect_forecasts:
            mu, raw_forecasts = forecast_returns_patchtst(
                train_data, horizon=TEST_WINDOW, verbose=False, return_raw=True
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
                    'model': 'PatchTST'
                })
        else:
            mu = forecast_returns_patchtst(train_data, horizon=TEST_WINDOW, verbose=False)

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
        asset_gross = np.exp(test_data.sum(axis=0).values)
        portfolio_gross = np.dot(weights, asset_gross)
        if not FULLY_INVESTED:
            portfolio_gross += (1 - weights.sum())
        month_return = np.log(portfolio_gross)

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        if weights_list is not None:
            weights_list.append(weights)

        if step % 5 == 0 or step == 1:
            pct = step * 100 // total_steps
            # Топ-3 актива с наибольшими весами
            top_idx = weights.argsort()[-3:][::-1]
            top_weights = [(returns.columns[i], weights[i]) for i in top_idx]
            top_str = ", ".join([f"{ticker}:{w:.1%}" for ticker, w in top_weights])
            print(f"Шаг {step}/{total_steps} ({pct}%): {test_data.index[0].date()}")
            print(f"  μ range: [{mu.min():.4f}, {mu.max():.4f}]")
            print(f"  top-3: {top_str}, std={weights.std():.3f}")

        i += TEST_WINDOW

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

    print("="*60)
    print("БЭКТЕСТ: PatchTST Self-Supervised")
    print("="*60)
    print(f"Данные: {returns.index[0].date()} — {returns.index[-1].date()}")
    print()

    results_path = Path(__file__).parent.parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    # Бэктест
    portfolio_returns = run_backtest(
        returns,
        save_weights_path=results_path / "patchtst_weights.csv"
    )

    # Метрики
    metrics = calculate_metrics(portfolio_returns, rf=RF)

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ: PatchTST Self-Supervised")
    print("="*60)
    for name, value in metrics.items():
        if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
            print(f"{name}: {value:.2%}")
        elif 'Ratio' in name:
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")

    # Сохраняем результаты
    portfolio_returns.to_csv(results_path / "patchtst_returns.csv")
    print(f"\nРезультаты сохранены в {results_path / 'patchtst_returns.csv'}")
