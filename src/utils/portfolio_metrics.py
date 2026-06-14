"""
Портфельные метрики, оборот и транзакционные издержки.

Каноничная реализация calculate_metrics (формулы идентичны прежним копиям
в run_all.py и бэктестах) + расчёт оборота по дрейфованным весам,
вычет издержек и разбиение результатов на validation/holdout периоды.
"""

import numpy as np
import pandas as pd


def calculate_metrics(returns, rf=0.02):
    """Расчёт метрик портфеля (returns — месячные лог-доходности)."""
    if len(returns) == 0:
        return {
            'Annual Return': np.nan,
            'Annual Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Calmar Ratio': np.nan,
            'Max Drawdown': np.nan,
            'Total Return': np.nan,
            'Num Periods': 0
        }

    simple_returns = np.exp(returns) - 1
    monthly_rf = (1 + rf) ** (1 / 12) - 1
    excess = simple_returns - monthly_rf

    annual_return = (1 + simple_returns).prod() ** (12 / len(simple_returns)) - 1
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


def compute_turnover_series(weights_df, asset_log_returns):
    """
    Оборот портфеля на каждую дату ребалансировки.

    Веса предыдущего месяца дрейфуют вместе с ценами (buy-and-hold внутри
    месяца), поэтому оборот считается против дрейфованных весов:
        g_i = exp(сумма дневных лог-доходностей актива i за месяц)
        w_drift = (w_prev * g) / (w_prev . g)
        turnover_t = sum_i |w_t_i - w_drift_i|

    Args:
        weights_df: DataFrame весов, индекс — даты ребалансировки
        asset_log_returns: DataFrame дневных лог-доходностей тех же активов

    Returns:
        Series оборота (первая дата = 0: стартовая покупка одинакова
        для всех fully-invested стратегий и в сравнении не участвует)
    """
    asset_log_returns = asset_log_returns[weights_df.columns]
    turnover = pd.Series(0.0, index=weights_df.index)

    dates = weights_df.index
    for k in range(1, len(dates)):
        prev_date, cur_date = dates[k - 1], dates[k]
        # Месяц предыдущих весов: торговые дни [prev_date, cur_date)
        month_returns = asset_log_returns.loc[
            (asset_log_returns.index >= prev_date) & (asset_log_returns.index < cur_date)
        ]
        gross = np.exp(month_returns.sum(axis=0).values)

        w_prev = weights_df.iloc[k - 1].values
        portfolio_gross = float(np.dot(w_prev, gross))
        if portfolio_gross <= 0:
            w_drift = w_prev
        else:
            w_drift = w_prev * gross / portfolio_gross

        w_cur = weights_df.iloc[k].values
        turnover.iloc[k] = float(np.abs(w_cur - w_drift).sum())

    return turnover


def apply_transaction_costs(portfolio_log_returns, turnover, cost_rate):
    """
    Вычесть транзакционные издержки из месячных лог-доходностей.

    net_simple_t = (exp(r_t) - 1) - cost_rate * turnover_t

    Args:
        portfolio_log_returns: Series месячных лог-доходностей
        turnover: Series оборота (та же индексация по датам)
        cost_rate: издержки на единицу оборота (например 0.0005 = 5 б.п.)

    Returns:
        Series месячных лог-доходностей за вычетом издержек
    """
    turnover = turnover.reindex(portfolio_log_returns.index).fillna(0.0)
    net_simple = (np.exp(portfolio_log_returns) - 1) - cost_rate * turnover
    return pd.Series(np.log1p(net_simple), index=portfolio_log_returns.index)


def split_by_date(returns, validation_end):
    """
    Разбить серию доходностей на периоды оценки.

    Месяц относится к периоду по дате РЕБАЛАНСИРОВКИ (первый день
    test-окна): последний validation-месяц реализует доходность до
    ~test_window торговых дней после validation_end. Разбиение точное,
    без пересечений и двойного счёта.

    Args:
        returns: Series с DatetimeIndex (даты ребалансировок)
        validation_end: дата конца validation-периода (ISO-строка)

    Returns:
        dict: {'full': вся серия,
               'validation': даты <= validation_end (тюнинг гиперпараметров),
               'holdout': даты > validation_end (финальная оценка)}
    """
    validation_end = pd.Timestamp(validation_end)
    return {
        'full': returns,
        'validation': returns[returns.index <= validation_end],
        'holdout': returns[returns.index > validation_end],
    }
