"""
Наивные бенчмарки для сравнения со стратегиями Марковица:
- Equal Weight (1/N): равные веса, та же ежемесячная ребалансировка
- Buy & Hold индексного бенчмарка (SPY): купил и держишь

Используют тот же walk-forward цикл (train 1260 / test 21 / шаг 21),
чтобы периоды совпадали с основными стратегиями один в один.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))

# Загружаем конфигурацию
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

TRAIN_WINDOW = config['backtest']['train_window']
TEST_WINDOW = config['backtest']['test_window']


def run_equal_weight(returns, train_window=None, test_window=None):
    """
    Бэктест Equal Weight (1/N) с ежемесячной ребалансировкой.

    Args:
        returns: DataFrame дневных лог-доходностей
        train_window, test_window: размеры окон (по умолчанию из config)

    Returns:
        portfolio_returns: Series месячных лог-доходностей
        weights_df: DataFrame весов (для расчёта оборота от дрейфа)
    """
    train_window = train_window or TRAIN_WINDOW
    test_window = test_window or TEST_WINDOW

    n = len(returns)
    n_assets = len(returns.columns)
    w = np.full(n_assets, 1.0 / n_assets)

    portfolio_returns = []
    dates = []
    weights_list = []

    i = 0
    while i + train_window + test_window <= n:
        test_data = returns.iloc[i + train_window:i + train_window + test_window]

        asset_gross = np.exp(test_data.sum(axis=0).values)
        month_return = np.log(np.dot(w, asset_gross))

        portfolio_returns.append(month_return)
        dates.append(test_data.index[0])
        weights_list.append(w.copy())

        i += test_window

    weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)
    return pd.Series(portfolio_returns, index=dates), weights_df


def run_buy_and_hold(benchmark_log_returns, universe_returns,
                     train_window=None, test_window=None):
    """
    Buy & Hold бенчмарка (например SPY) на тех же месячных периодах.

    Args:
        benchmark_log_returns: Series дневных лог-доходностей бенчмарка
        universe_returns: DataFrame доходностей акций (задаёт сетку окон)
        train_window, test_window: размеры окон (по умолчанию из config)

    Returns:
        Series месячных лог-доходностей бенчмарка
    """
    train_window = train_window or TRAIN_WINDOW
    test_window = test_window or TEST_WINDOW

    n = len(universe_returns)
    if n <= train_window:
        return pd.Series(dtype=float)

    # Покрытие: бенчмарк должен закрывать все test-окна, иначе месяцы
    # с пустым срезом молча превратились бы в фиктивные 0%
    first_test = universe_returns.index[train_window]
    last_test = universe_returns.index[-1]
    if (benchmark_log_returns.index.min() > first_test
            or benchmark_log_returns.index.max() < last_test):
        raise ValueError(
            f"Бенчмарк покрывает {benchmark_log_returns.index.min().date()}.."
            f"{benchmark_log_returns.index.max().date()}, а test-окна требуют "
            f"{first_test.date()}..{last_test.date()}. Пересоздайте кэш: "
            f"from data.downloader import download_benchmark; download_benchmark()"
        )

    portfolio_returns = []
    dates = []

    i = 0
    while i + train_window + test_window <= n:
        test_index = universe_returns.index[i + train_window:i + train_window + test_window]

        # Дни бенчмарка за тот же календарный месяц (границы — дни universe)
        month = benchmark_log_returns.loc[
            (benchmark_log_returns.index >= test_index[0])
            & (benchmark_log_returns.index <= test_index[-1])
        ]
        if month.empty:
            raise ValueError(
                f"Нет данных бенчмарка за окно {test_index[0].date()}..{test_index[-1].date()}"
            )
        portfolio_returns.append(month.sum())
        dates.append(test_index[0])

        i += test_window

    return pd.Series(portfolio_returns, index=dates)
