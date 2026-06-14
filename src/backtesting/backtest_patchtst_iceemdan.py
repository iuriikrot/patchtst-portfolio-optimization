"""
Бэктест PatchTST + ICEEMDAN: прогноз μ по компонентам декомпозиции.

Подход (отличия от backtest_patchtst.py):
1. Train-окно (1260 дней) каждого тикера каузально раскладывается CEEMDAN
   (только train — тестовые данные в декомпозицию не попадают)
2. IMF группируются в 3 фиксированных канала: noise / cycle / trend
3. Каналы прогнозируются одной многоканальной PatchTST (общие веса,
   channel-independence), прогноз ряда = сумма прогнозов каналов
4. μ = mean(прогноз) × 252, дальше Марковиц — как у остальных стратегий

Архитектура сети и схема pretrain+finetune наследуются из models.patchtst.{mode}.
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
warnings.filterwarnings('ignore', module='torch')

sys.path.append(str(Path(__file__).parent.parent))

from optimization.markowitz import maximize_sharpe
from optimization.covariance import compute_covariance
from utils.portfolio_metrics import calculate_metrics
from decomposition.iceemdan import decompose_and_group
from models.patchtst import (
    PatchTST_MultiChannel,
    pretrain_patchtst_mc,
    finetune_patchtst_mc,
    forecast_patchtst_mc,
    create_sequences_mc
)
from backtesting.backtest_patchtst import set_seed, select_device, task_seed

import torch

# Загружаем конфигурацию
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

RANDOM_SEED = config.get('random_seed', 42)

# Параметры бэктеста из config
TRAIN_WINDOW = config['backtest']['train_window']
TEST_WINDOW = config['backtest']['test_window']
RF = config['optimization']['risk_free_rate']
CV_METHOD = config['optimization'].get('covariance', 'sample')
CONSTRAINTS = config.get('optimization', {}).get('constraints', {})
MIN_WEIGHT = CONSTRAINTS.get('min_weight', 0.0)
MAX_WEIGHT = CONSTRAINTS.get('max_weight', 1.0)
LONG_ONLY = CONSTRAINTS.get('long_only', True)
FULLY_INVESTED = CONSTRAINTS.get('fully_invested', True)
GROSS_EXPOSURE = CONSTRAINTS.get('gross_exposure')

# Архитектура PatchTST — те же параметры, что у основной PatchTST-стратегии
MODE = config['models']['patchtst'].get('mode', 'fast')
PADDING_PATCH = config['models']['patchtst'].get('padding_patch', True)
N_WORKERS = config['models']['patchtst'].get('n_workers', 1)
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
FINETUNE_EPOCHS = mode_config.get('finetune_epochs', 5)
PRETRAIN_LR = mode_config['pretrain_lr']
BATCH_SIZE = mode_config['batch_size']

# Параметры декомпозиции
ICEEMDAN_CFG = config['models'].get('iceemdan', {})
FORECAST_NOISE = ICEEMDAN_CFG.get('forecast_noise', True)
_cache_cfg = ICEEMDAN_CFG.get('cache', {})
if _cache_cfg.get('enabled', True):
    CACHE_DIR = Path(__file__).parent.parent.parent / _cache_cfg.get('dir', 'data/cache/iceemdan')
else:
    CACHE_DIR = None

N_CHANNELS = 3  # noise / cycle / trend


# ============================================================
# Параллелизм по тикерам (декомпозиция + обучение в воркерах)
# ============================================================

def _worker_init():
    torch.set_num_threads(1)


def _train_and_forecast_components(components, device, verbose=False):
    """Полный цикл одного тикера: модель по 3 каналам, прогноз = сумма каналов."""
    # Каналы, тождественно равные нулю в train (пустая группа IMF),
    # не прогнозируются моделью
    zero_channels = [c for c in range(N_CHANNELS) if np.allclose(components[c], 0)]

    model = PatchTST_MultiChannel(
        n_channels=N_CHANNELS,
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
        use_revin=USE_REVIN,
        padding_patch=PADDING_PATCH
    ).to(device)

    # Pre-training с маскированием патчей (на компонентах)
    model = pretrain_patchtst_mc(
        model, components,
        epochs=PRETRAIN_EPOCHS,
        lr=PRETRAIN_LR,
        batch_size=BATCH_SIZE,
        verbose=verbose
    )

    # Fine-tuning end-to-end на supervised-парах компонент
    X_train, y_train = create_sequences_mc(components, INPUT_LEN, PRED_LEN)
    if len(X_train) > 0:
        model = finetune_patchtst_mc(
            model, X_train, y_train,
            epochs=FINETUNE_EPOCHS,
            lr=PRETRAIN_LR * 0.1,
            batch_size=BATCH_SIZE,
            verbose=verbose
        )

    # Прогноз компонент по последним INPUT_LEN точкам
    last_input = components[:, -INPUT_LEN:]
    comp_forecast = forecast_patchtst_mc(model, last_input)  # (3, pred_len)

    for c in zero_channels:
        comp_forecast[c] = 0.0
    if not FORECAST_NOISE:
        # Денойзинг: высокочастотная компонента в прогнозе обнуляется
        comp_forecast[0] = 0.0

    return comp_forecast.sum(axis=0)


def _forecast_ticker_worker(args):
    """Задача воркера: каузальная декомпозиция + обучение + прогноз одного тикера."""
    series, seed, device = args
    try:
        components = decompose_and_group(series, ICEEMDAN_CFG, cache_dir=CACHE_DIR)
    except ImportError:
        # Отсутствие зависимости — не per-ticker сбой, валим прогон целиком
        raise
    except Exception as e:
        return {'forecast': None, 'error': str(e)}
    set_seed(seed)
    return {'forecast': _train_and_forecast_components(components, device), 'error': None}


_POOL = None


def _get_pool():
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        _POOL = mp.get_context('spawn').Pool(N_WORKERS, initializer=_worker_init)
    return _POOL


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None


def forecast_returns_patchtst_iceemdan(train_returns, horizon=21, verbose=False, return_raw=False):
    """
    Прогноз доходностей всех акций: ICEEMDAN-декомпозиция train-окна +
    многоканальный PatchTST по компонентам.

    Args:
        train_returns: DataFrame с доходностями (train период, 1260 дней)
        horizon: горизонт прогноза в днях
        verbose: выводить прогресс обучения
        return_raw: если True, вернуть также raw прогнозы (horizon × N_tickers)

    Returns:
        mu: вектор ожидаемых годовых доходностей
        raw_forecasts: DataFrame (horizon × N_tickers) — если return_raw=True
    """
    if horizon != PRED_LEN:
        warnings.warn(
            f"horizon ({horizon}) != PRED_LEN ({PRED_LEN}) из config. "
            f"Модель обучена на pred_len={PRED_LEN}, но запрошен прогноз на {horizon} дней."
        )

    tickers = train_returns.columns
    fallback = train_returns.mean()
    window_end = train_returns.index[-1].date()
    # Ключ окна для посидовых сидов: дата конца train-окна
    window_key = int(train_returns.index[-1].strftime('%Y%m%d'))

    raw_forecasts = pd.DataFrame(index=range(horizon), columns=tickers, dtype=float)
    fallback_count = 0

    # Тикеры с достаточной историей — в задачи, остальные — fallback
    tasks = []
    for idx, ticker in enumerate(tickers):
        series = train_returns[ticker].values
        if len(series) < INPUT_LEN:
            raw_forecasts[ticker] = fallback[ticker]
            fallback_count += 1
            continue
        tasks.append((ticker, series, task_seed(RANDOM_SEED, window_key, idx)))

    if N_WORKERS > 1:
        pool = _get_pool()
        outcomes = pool.map(
            _forecast_ticker_worker,
            [(series, seed, 'cpu') for _, series, seed in tasks]
        )
    else:
        device = select_device()
        outcomes = [
            _forecast_ticker_worker((series, seed, device))
            for _, series, seed in tasks
        ]

    for (ticker, _, _), outcome in zip(tasks, outcomes):
        if outcome['error'] is not None:
            warnings.warn(
                f"Декомпозиция для {ticker} (окно до {window_end}) не удалась "
                f"({outcome['error']}). Используется fallback (историческое среднее)."
            )
            raw_forecasts[ticker] = fallback[ticker]
            fallback_count += 1
            continue

        forecast = outcome['forecast']
        if len(forecast) == horizon:
            raw_forecasts[ticker] = forecast
        else:
            warnings.warn(
                f"Прогноз для {ticker}: len={len(forecast)}, ожидалось horizon={horizon}. "
                f"Используется fallback (историческое среднее)."
            )
            raw_forecasts[ticker] = fallback[ticker]
            fallback_count += 1

    if fallback_count == len(tickers):
        # 100% fallback'ов = стратегия молча выродилась бы в Baseline 1
        raise RuntimeError(
            f"Все {len(tickers)} тикеров ушли в fallback (окно до {window_end}) — "
            f"декомпозиция/прогноз не работают, результат был бы дубликатом Baseline 1."
        )

    # mu = среднее по дням × 252
    mu = raw_forecasts.mean(axis=0).values * 252

    if return_raw:
        return mu, raw_forecasts
    return mu


def run_backtest(returns, save_weights_path=None, collect_forecasts=False):
    """
    Бэктест со скользящим окном (та же сетка, что у остальных стратегий).

    Args:
        returns: DataFrame с лог-доходностями
        save_weights_path: путь для сохранения весов (опционально)
        collect_forecasts: собирать прогнозы для расчёта forecast metrics

    Returns:
        portfolio_returns: Series с доходностями портфеля
        forecasts_df: DataFrame с прогнозами (если collect_forecasts=True)
    """
    set_seed(RANDOM_SEED)

    n = len(returns)
    portfolio_returns = []
    dates = []
    weights_list = [] if save_weights_path else None
    forecast_records = [] if collect_forecasts else None

    if N_WORKERS > 1:
        print(f"Устройство: cpu × {N_WORKERS} воркеров (параллельно по тикерам)")
    else:
        print(f"Устройство: {select_device()}")
    print(f"Всего дней: {n}")
    print(f"Train окно: {TRAIN_WINDOW} дней")
    print(f"Test окно: {TEST_WINDOW} дней")
    print(f"Акций: {len(returns.columns)}")
    print(f"PatchTST+ICEEMDAN параметры:")
    print(f"  - каналы: noise/cycle/trend, forecast_noise: {FORECAST_NOISE}")
    print(f"  - ICEEMDAN: trials={ICEEMDAN_CFG.get('trials')}, epsilon={ICEEMDAN_CFG.get('epsilon')}, "
          f"группировка: <{ICEEMDAN_CFG.get('grouping', {}).get('noise_max_period')} / "
          f"<={ICEEMDAN_CFG.get('grouping', {}).get('cycle_max_period')} дней")
    print(f"  - кэш декомпозиций: {CACHE_DIR}")
    print(f"  - input_len: {INPUT_LEN}, pred_len: {PRED_LEN}")
    print(f"  - patch_len: {PATCH_LEN}, stride: {STRIDE}, padding_patch: {PADDING_PATCH}")
    print(f"  - d_model: {D_MODEL}, n_heads: {N_HEADS}, n_layers: {N_LAYERS}, d_ff: {D_FF}")
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

        # μ из прогнозов PatchTST+ICEEMDAN
        if collect_forecasts:
            mu, raw_forecasts = forecast_returns_patchtst_iceemdan(
                train_data, horizon=TEST_WINDOW, verbose=False, return_raw=True
            )
            actual_monthly = test_data.sum(axis=0)
            predicted_monthly = raw_forecasts.sum(axis=0)
            for ticker in returns.columns:
                forecast_records.append({
                    'date': test_data.index[0],
                    'ticker': ticker,
                    'actual': actual_monthly[ticker],
                    'predicted': predicted_monthly[ticker],
                    'model': 'PatchTST-ICEEMDAN'
                })
        else:
            mu = forecast_returns_patchtst_iceemdan(train_data, horizon=TEST_WINDOW, verbose=False)

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
            top_idx = weights.argsort()[-3:][::-1]
            top_weights = [(returns.columns[i], weights[i]) for i in top_idx]
            top_str = ", ".join([f"{ticker}:{w:.1%}" for ticker, w in top_weights])
            print(f"Шаг {step}/{total_steps} ({pct}%): {test_data.index[0].date()}")
            print(f"  μ range: [{mu.min():.4f}, {mu.max():.4f}]")
            print(f"  top-3: {top_str}, std={weights.std():.3f}")

        i += TEST_WINDOW

    print(f"\nЗавершено. Всего периодов: {len(portfolio_returns)}")
    close_pool()

    if weights_list is not None:
        weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
        weights_df.to_csv(save_weights_path)

    result = pd.Series(portfolio_returns, index=dates)

    if collect_forecasts:
        forecasts_df = pd.DataFrame(forecast_records)
        return result, forecasts_df
    return result


if __name__ == "__main__":
    # Загружаем данные
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "log_returns.csv"
    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

    data_end = config['backtest'].get('data_end')
    if data_end:
        returns = returns.loc[:data_end]

    print("=" * 60)
    print("БЭКТЕСТ: PatchTST + ICEEMDAN")
    print("=" * 60)
    print(f"Данные: {returns.index[0].date()} — {returns.index[-1].date()}")
    print()

    results_path = Path(__file__).parent.parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    portfolio_returns = run_backtest(
        returns,
        save_weights_path=results_path / "patchtst_iceemdan_weights.csv"
    )

    metrics = calculate_metrics(portfolio_returns, rf=RF)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ: PatchTST + ICEEMDAN")
    print("=" * 60)
    for name, value in metrics.items():
        if 'Return' in name or 'Volatility' in name or 'Drawdown' in name:
            print(f"{name}: {value:.2%}")
        elif 'Ratio' in name:
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")

    portfolio_returns.to_csv(results_path / "patchtst_iceemdan_returns.csv")
    print(f"\nРезультаты сохранены в {results_path / 'patchtst_iceemdan_returns.csv'}")
