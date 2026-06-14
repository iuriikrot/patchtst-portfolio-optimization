"""
Предрасчёт ICEEMDAN-декомпозиций для всех окон бэктеста (параллельно на CPU).

Декомпозиции детерминированы и кэшируются по хэшу значений окна+параметров,
поэтому предрасчёт полностью безопасен: бэктест просто возьмёт готовые файлы
из кэша. Это отделяет CPU-часть (CEEMDAN, хорошо параллелится) от GPU-части
(обучение моделей): на 8 ядрах полный период считается ~20-25 минут вместо
~2 часов внутри бэктеста.

Использование:
    python scripts/precompute_decompositions.py                # окна по config (с учётом data_end)
    python scripts/precompute_decompositions.py --all          # весь период, игнорируя data_end
    python scripts/precompute_decompositions.py --workers 4 --limit 5   # для проверки
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

_CFG = None
_CACHE_DIR = None


def _worker_init():
    """Каждый воркер один раз читает config."""
    global _CFG, _CACHE_DIR
    with open(ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    _CFG = config["models"].get("iceemdan", {})
    cache_cfg = _CFG.get("cache", {})
    _CACHE_DIR = ROOT / cache_cfg.get("dir", "data/cache/iceemdan")


def _decompose_one(series):
    from decomposition.iceemdan import decompose_and_group

    decompose_and_group(series, _CFG, cache_dir=_CACHE_DIR)
    return True


def main():
    parser = argparse.ArgumentParser(description="Предрасчёт ICEEMDAN-декомпозиций")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="число процессов (default: cpu_count - 2)")
    parser.add_argument("--all", action="store_true",
                        help="игнорировать backtest.data_end (весь период)")
    parser.add_argument("--limit", type=int, default=None,
                        help="только первые N окон (для проверки)")
    args = parser.parse_args()

    with open(ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    train_window = config["backtest"]["train_window"]
    test_window = config["backtest"]["test_window"]
    data_end = None if args.all else config["backtest"].get("data_end")

    returns = pd.read_csv(ROOT / "data" / "raw" / "log_returns.csv",
                          index_col=0, parse_dates=True)
    if data_end:
        returns = returns.loc[:data_end]

    # Та же сетка окон, что в бэктестах: train [i, i+train), шаг = test_window
    n = len(returns)
    tasks = []
    i = 0
    windows = 0
    while i + train_window + test_window <= n:
        for ticker in returns.columns:
            tasks.append(returns[ticker].values[i:i + train_window])
        windows += 1
        if args.limit and windows >= args.limit:
            break
        i += test_window

    print(f"Окон: {windows}, задач (окно × тикер): {len(tasks)}, воркеров: {args.workers}")
    print(f"Период: {returns.index[0].date()} — {returns.index[-1].date()}"
          f"{' (data_end игнорирован)' if args.all else ''}")

    import multiprocessing as mp
    t0 = time.time()
    done = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init) as pool:
        for _ in pool.imap_unordered(_decompose_one, tasks, chunksize=4):
            done += 1
            if done % 100 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(tasks)} ({done * 100 // len(tasks)}%), "
                      f"~{rate:.1f} задач/с, осталось ~{eta / 60:.0f} мин", flush=True)

    print(f"Готово за {(time.time() - t0) / 60:.1f} мин. Кэш: {ROOT / 'data/cache/iceemdan'}")


if __name__ == "__main__":
    main()
