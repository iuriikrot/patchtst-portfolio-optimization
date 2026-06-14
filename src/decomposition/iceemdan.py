"""
Каузальная ICEEMDAN-декомпозиция временного ряда для прогнозирования.

Используется PyEMD.CEEMDAN (пакет EMD-signal), включающий улучшения
из Colominas et al. 2014 (ICEEMDAN). Ключевые свойства реализации:

1. Каузальность: разложение вызывается ТОЛЬКО на train-окне бэктеста —
   тестовые данные в декомпозицию не попадают.
2. Детерминизм: шумовой ансамбль сидируется (noise_seed), parallel=False.
3. Фиксированное число выходных компонент: переменное число IMF
   детерминированно группируется в 3 канала (noise / cycle / trend)
   по среднему периоду; сумма каналов точно равна исходному ряду.
4. Кэш на диске: ключ — хэш значений окна и параметров, поэтому кэш
   автоматически инвалидируется при изменении данных и не может
   "протащить" информацию из будущего.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
# Жёсткий импорт на уровне модуля: отсутствие EMD-signal должно валить импорт
# стратегии (и ловиться try/except ImportError в run_all), а не тихо
# превращать каждый прогноз в fallback по историческому среднему
import PyEMD
from PyEMD import CEEMDAN

# Инкрементировать при изменении decompose_series/group_imfs/mean_period:
# входит в ключ кэша, чтобы старые декомпозиции не пережили смену алгоритма
_ALGO_VERSION = 1


def mean_period(x):
    """
    Средний период компоненты в торговых днях: 2*T / число_пересечений_нуля.
    Для компоненты без пересечений нуля (тренд/остаток) — бесконечность.
    """
    x = np.asarray(x, dtype=float)
    zero_crossings = int(np.sum(np.abs(np.diff(np.sign(x))) > 0))
    if zero_crossings == 0:
        return np.inf
    return 2.0 * len(x) / zero_crossings


def decompose_series(series, trials=50, epsilon=0.2, seed=42, max_imf=-1):
    """
    CEEMDAN-разложение ряда.

    Returns:
        components: (K+1, T) — K IMF + остаток (residue) последней строкой.
        Сумма строк равна исходному ряду с машинной точностью.
    """
    series = np.asarray(series, dtype=float)

    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=False)
    ceemdan.noise_seed(seed)

    imfs = ceemdan.ceemdan(series, max_imf=max_imf)
    residue = series - imfs.sum(axis=0)

    return np.vstack([imfs, residue[None, :]])


def group_imfs(components, noise_max_period=5.0, cycle_max_period=63.0):
    """
    Детерминированная группировка IMF в 3 фиксированных канала.

    Критерий — средний период IMF:
        period < noise_max_period             -> noise  (канал 0)
        noise_max_period <= period <= cycle  -> cycle  (канал 1)
        period > cycle_max_period            -> trend  (канал 2)
    Остаток (последняя строка components) всегда попадает в trend.

    Args:
        components: (K+1, T) — IMF + residue (из decompose_series)

    Returns:
        grouped: (3, T) — [noise, cycle, trend], sum(grouped) == sum(components)
    """
    components = np.asarray(components, dtype=float)
    T = components.shape[1]
    grouped = np.zeros((3, T))

    # Остаток — всегда тренд
    grouped[2] += components[-1]

    for imf in components[:-1]:
        period = mean_period(imf)
        if period < noise_max_period:
            grouped[0] += imf
        elif period <= cycle_max_period:
            grouped[1] += imf
        else:
            grouped[2] += imf

    return grouped


def _cache_key(series, params):
    """Ключ кэша: хэш значений окна + параметров декомпозиции."""
    payload = np.asarray(series, dtype=float).tobytes()
    payload += json.dumps(params, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()


def decompose_and_group(series, cfg, cache_dir=None):
    """
    Полный пайплайн: CEEMDAN -> группировка в 3 канала, с кэшем на диске.

    Args:
        series: 1D-массив лог-доходностей (строго train-окно!)
        cfg: dict с параметрами (models.iceemdan из config.yaml):
            trials, epsilon, seed, max_imf, grouping.{noise_max_period, cycle_max_period}
        cache_dir: директория кэша (None = без кэша)

    Returns:
        grouped: (3, T) — [noise, cycle, trend]

    Raises:
        исключение при сбое декомпозиции — fallback решает вызывающий код.
    """
    series = np.asarray(series, dtype=float)

    grouping = cfg.get('grouping', {})
    params = {
        'algo_version': _ALGO_VERSION,
        'pyemd_version': getattr(PyEMD, '__version__', 'unknown'),
        'trials': int(cfg.get('trials', 50)),
        'epsilon': float(cfg.get('epsilon', 0.2)),
        'seed': int(cfg.get('seed', 42)),
        'max_imf': int(cfg.get('max_imf', -1)),
        'noise_max_period': float(grouping.get('noise_max_period', 5)),
        'cycle_max_period': float(grouping.get('cycle_max_period', 63)),
    }

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{_cache_key(series, params)}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path) as cached:
                    return cached['grouped']
            except Exception:
                # Битый файл (например, прерванная запись) — удаляем и пересчитываем
                cache_path.unlink()

    components = decompose_series(
        series,
        trials=params['trials'],
        epsilon=params['epsilon'],
        seed=params['seed'],
        max_imf=params['max_imf'],
    )
    grouped = group_imfs(
        components,
        noise_max_period=params['noise_max_period'],
        cycle_max_period=params['cycle_max_period'],
    )

    # Точность реконструкции: сумма каналов == исходный ряд
    reconstruction_error = float(np.max(np.abs(grouped.sum(axis=0) - series)))
    if not np.allclose(grouped.sum(axis=0), series, atol=1e-8):
        raise ValueError(
            f"Декомпозиция не восстанавливает ряд: max|err|={reconstruction_error:.2e}"
        )

    if cache_path is not None:
        # Атомарная запись: temp-файл в той же директории + os.replace,
        # чтобы прерывание прогона не оставило битый .npz
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, suffix='.npz.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                np.savez_compressed(f, grouped=grouped, num_imfs=components.shape[0] - 1)
            os.replace(tmp_name, cache_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    return grouped
