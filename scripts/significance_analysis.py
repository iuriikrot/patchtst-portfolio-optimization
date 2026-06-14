"""
Статистический анализ значимости результатов бэктеста.

Считается на сохранённых рядах чистых (net) месячных доходностей —
без перезапуска моделей. Включает:

1. Тест Ледуа–Вольфа (2008) на разницу коэффициентов Шарпа:
   - аналитический вариант с HAC-робастной (Newey–West) стандартной
     ошибкой через метод дельта и функции влияния;
   - studentized stationary block bootstrap (Politis–Romano) —
     рекомендованный в работе вариант, устойчив к автокорреляции
     и ненормальности на конечной выборке.
2. Поправка Холма на множественные сравнения.
3. Block-bootstrap доверительные интервалы для Sharpe и Calmar
   каждой стратегии.

Использование:
    python scripts/significance_analysis.py [timestamp]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RF_ANNUAL = 0.04
RF_M = (1 + RF_ANNUAL) ** (1 / 12) - 1
MONTHS = 12
B_BOOT = 10000
SEED = 42

STRATEGIES = {
    'ICEEMDAN': 'patchtst_iceemdan',
    'Baseline 1': 'baseline1',
    'AutoARIMA': 'statsforecast',
    'PatchTST': 'patchtst',
    '1/N': 'equal_weight',
    'SPY': 'spy_buyhold',
}


def load_net_returns(prefix, ts):
    """Простые месячные net-доходности стратегии."""
    s = pd.read_csv(ROOT / 'results' / f'{prefix}_returns_net_{ts}.csv',
                    index_col=0, parse_dates=True).iloc[:, 0]
    return (np.exp(s) - 1).values


def sharpe(x):
    """Аннуализированный Sharpe по месячным простым доходностям (excess)."""
    e = x - RF_M
    sd = e.std(ddof=1)
    return e.mean() / sd * np.sqrt(MONTHS) if sd > 0 else 0.0


def calmar(x):
    """Calmar = CAGR / |MaxDD| по месячным простым доходностям."""
    cum = np.cumprod(1 + x)
    cagr = cum[-1] ** (MONTHS / len(x)) - 1
    dd = cum / np.maximum.accumulate(cum) - 1
    mdd = dd.min()
    return cagr / abs(mdd) if mdd < 0 else np.nan


def sharpe_influence(x):
    """Функция влияния Sharpe (монтхли, excess) для метода дельта."""
    e = x - RF_M
    mu = e.mean()
    var = e.var(ddof=0)
    sd = np.sqrt(var)
    sr = mu / sd
    # IF_t = (e-mu)/sd - 0.5*sr*((e-mu)^2/var - 1)
    return (e - mu) / sd - 0.5 * sr * (((e - mu) ** 2) / var - 1.0)


def hac_lrv(g, bandwidth=None):
    """Длиннопериодная дисперсия (Newey–West, ядро Бартлетта)."""
    T = len(g)
    g = g - g.mean()
    if bandwidth is None:
        bandwidth = int(np.floor(4 * (T / 100) ** (2 / 9)))
    gamma0 = (g @ g) / T
    lrv = gamma0
    for j in range(1, bandwidth + 1):
        w = 1 - j / (bandwidth + 1)
        gj = (g[j:] @ g[:-j]) / T
        lrv += 2 * w * gj
    return max(lrv, 1e-12)


def sharpe_diff_se(xa, xb):
    """HAC-SE разницы Sharpe (аннуализированной) через функции влияния."""
    T = len(xa)
    g = sharpe_influence(xa) - sharpe_influence(xb)   # разница IF (помесячно)
    lrv = hac_lrv(g)
    se_monthly = np.sqrt(lrv / T)
    return se_monthly * np.sqrt(MONTHS)               # аннуализация SE


def hac_test(xa, xb):
    """Аналитический HAC-тест: z и односторонний p (H1: SR_a > SR_b)."""
    from math import erf
    d = sharpe(xa) - sharpe(xb)
    se = sharpe_diff_se(xa, xb)
    z = d / se if se > 0 else 0.0
    p_one = 1 - 0.5 * (1 + erf(z / np.sqrt(2)))        # P(Z > z)
    return d, se, z, p_one


def stationary_bootstrap_indices(T, expected_block, rng):
    """Индексы одной реплики stationary bootstrap (Politis–Romano)."""
    p = 1.0 / expected_block
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(0, T)
    for t in range(1, T):
        if rng.random() < p:
            idx[t] = rng.integers(0, T)
        else:
            idx[t] = (idx[t - 1] + 1) % T
    return idx


def studentized_block_bootstrap(xa, xb, rng, expected_block=6, B=B_BOOT):
    """Studentized stationary block bootstrap для разницы Sharpe (Ледуа-Вольф)."""
    T = len(xa)
    d_obs = sharpe(xa) - sharpe(xb)
    se_obs = sharpe_diff_se(xa, xb)
    t_obs = d_obs / se_obs
    count = 0
    for _ in range(B):
        idx = stationary_bootstrap_indices(T, expected_block, rng)
        ba, bb = xa[idx], xb[idx]
        d_b = sharpe(ba) - sharpe(bb)
        se_b = sharpe_diff_se(ba, bb)
        if se_b <= 0:
            continue
        t_b = (d_b - d_obs) / se_b          # центрируем на наблюдённую разницу
        if t_b >= t_obs:                    # односторонний H1: SR_a > SR_b
            count += 1
    return d_obs, t_obs, count / B


def block_bootstrap_ci(x, fn, rng, expected_block=6, B=5000, alpha=0.05):
    """Перцентильный CI для метрики fn(x) через stationary block bootstrap."""
    T = len(x)
    vals = np.empty(B)
    for b in range(B):
        idx = stationary_bootstrap_indices(T, expected_block, rng)
        vals[b] = fn(x[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return fn(x), lo, hi


def holm(pvals):
    """Поправка Холма; возвращает скорректированные p в исходном порядке."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def main():
    ts = sys.argv[1] if len(sys.argv) > 1 else '20260612_091600'
    rng = np.random.default_rng(SEED)

    data = {name: load_net_returns(pref, ts) for name, pref in STRATEGIES.items()}
    T = len(data['ICEEMDAN'])
    print(f"Анализ значимости (timestamp {ts}, {T} месяцев, net-доходности)\n")

    # --- Доверительные интервалы Sharpe / Calmar (block bootstrap) ---
    print("=" * 74)
    print("Block-bootstrap 95% ДИ (stationary, ожид. блок 6 мес)")
    print("=" * 74)
    print(f"{'Стратегия':<14}{'Sharpe':>8}{'  95% ДИ Sharpe':>20}{'Calmar':>9}{'  95% ДИ Calmar':>20}")
    for name in STRATEGIES:
        x = data[name]
        s, slo, shi = block_bootstrap_ci(x, sharpe, rng)
        c, clo, chi = block_bootstrap_ci(x, calmar, rng)
        print(f"{name:<14}{s:>8.2f}  [{slo:>5.2f}, {shi:>5.2f}]   {c:>8.2f}  [{clo:>5.2f}, {chi:>5.2f}]")

    # --- Тест разницы Шарпов: ICEEMDAN против остальных ---
    print("\n" + "=" * 74)
    print("Разница Шарпов: ICEEMDAN vs X (H1: SR_ICEEMDAN > SR_X)")
    print("=" * 74)
    others = ['PatchTST', 'SPY', '1/N', 'AutoARIMA', 'Baseline 1']
    rows = []
    p_boot_list = []
    for name in others:
        d, se, z, p_hac = hac_test(data['ICEEMDAN'], data[name])
        _, t_obs, p_boot = studentized_block_bootstrap(data['ICEEMDAN'], data[name], rng)
        rows.append((name, d, se, z, p_hac, p_boot))
        p_boot_list.append(p_boot)

    p_holm = holm(np.array(p_boot_list))

    print(f"{'Сравнение':<22}{'ΔSR':>7}{'HAC SE':>8}{'z':>6}{'p(HAC)':>9}{'p(boot)':>9}{'p Холм':>9}{'  вывод':>14}")
    for (name, d, se, z, p_hac, p_boot), ph in zip(rows, p_holm):
        verdict = 'значимо 1%' if ph < 0.01 else 'значимо 5%' if ph < 0.05 else 'маргин. 10%' if ph < 0.10 else 'не значимо'
        print(f"{'ICEEMDAN vs '+name:<22}{d:>7.2f}{se:>8.3f}{z:>6.2f}{p_hac:>9.3f}{p_boot:>9.3f}{ph:>9.3f}{verdict:>14}")

    # --- Sanity-проверки ---
    print("\n" + "=" * 74)
    print("Sanity: HAC LRV при bandwidth=0 (≈ i.i.d.) vs авто — для ICEEMDAN vs Baseline 1")
    g = sharpe_influence(data['ICEEMDAN']) - sharpe_influence(data['Baseline 1'])
    print(f"  bandwidth=0 (i.i.d.): LRV={hac_lrv(g, 0):.4f}   авто Newey–West: LRV={hac_lrv(g):.4f}")
    print(f"  (если LRV растёт — есть положительная автокорреляция, наивный bootstrap занижал бы p)")


if __name__ == '__main__':
    main()
