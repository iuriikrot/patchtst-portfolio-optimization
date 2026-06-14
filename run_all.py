"""
Запуск всех моделей и сохранение результатов.

Использование:
    python run_all.py

Результаты сохраняются в results/.
Все параметры (включая режим PatchTST) берутся из config/config.yaml.

Стратегии: Baseline 1 (hist mean), Baseline 2 (AutoARIMA), PatchTST,
PatchTST + ICEEMDAN. Бенчмарки Equal Weight (1/N) и Buy & Hold (SPY)
считаются всегда. Метрики выводятся без издержек (gross) и с
транзакционными издержками (net), отдельно по периодам
full / validation / holdout (см. evaluation в config).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import json
from datetime import datetime
import warnings
# Подавляем только шумные warnings от библиотек, но не наши собственные (UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='numpy')

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.forecast_metrics import aggregate_forecast_metrics
from utils.portfolio_metrics import (
    calculate_metrics,
    compute_turnover_series,
    apply_transaction_costs,
    split_by_date,
)
from backtesting.benchmarks import run_equal_weight, run_buy_and_hold

# Импортируем бэктесты по отдельности — чтобы отсутствие одной зависимости
# не блокировало другие модели
run_baseline1_backtest = None
run_statsforecast = None
run_patchtst_backtest = None
run_patchtst_iceemdan_backtest = None

try:
    from backtesting.backtest import run_backtest as run_baseline1_backtest
except ImportError:
    pass

try:
    from backtesting.backtest_statsforecast import run_backtest as run_statsforecast
except ImportError:
    pass

try:
    from backtesting.backtest_patchtst import run_backtest as run_patchtst_backtest
except ImportError:
    pass

try:
    from backtesting.backtest_patchtst_iceemdan import run_backtest as run_patchtst_iceemdan_backtest
except ImportError:
    pass

# Загружаем конфигурацию
config_path = Path(__file__).parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Параметры
TRAIN_WINDOW = config['backtest']['train_window']
TEST_WINDOW = config['backtest']['test_window']
DATA_END = config['backtest'].get('data_end')
RF = config['optimization']['risk_free_rate']

EVAL_CFG = config.get('evaluation', {})
COST_RATE = EVAL_CFG.get('transaction_costs', {}).get('cost_rate', 0.0)
VALIDATION_END = EVAL_CFG.get('split', {}).get('validation_end', '2014-12-31')

PERIODS = ['full', 'validation', 'holdout']


# ============================================================
# Обёртки стратегий
# ============================================================

def run_baseline1(returns, save_weights_path=None, collect_forecasts=False):
    """Бэктест: μ = историческое среднее."""
    if run_baseline1_backtest is None:
        raise ImportError("Baseline 1 недоступен: не удалось импортировать backtest.py")
    return run_baseline1_backtest(
        returns,
        save_weights_path=save_weights_path,
        collect_forecasts=collect_forecasts
    )


def run_baseline2(returns, save_weights_path=None, collect_forecasts=False):
    """Бэктест: μ = прогноз StatsForecast AutoARIMA."""
    if run_statsforecast is None:
        raise ImportError("Baseline 2 недоступен: не удалось импортировать statsforecast")
    return run_statsforecast(
        returns,
        save_weights_path=save_weights_path,
        collect_forecasts=collect_forecasts
    )


def run_patchtst(returns, save_weights_path=None, collect_forecasts=False):
    """Бэктест: μ = прогноз PatchTST. Режим берётся из config."""
    if run_patchtst_backtest is None:
        raise ImportError("PatchTST недоступен: не удалось импортировать torch или patchtst")
    return run_patchtst_backtest(
        returns,
        save_weights_path=save_weights_path,
        collect_forecasts=collect_forecasts
    )


def run_patchtst_iceemdan(returns, save_weights_path=None, collect_forecasts=False):
    """Бэктест: μ = прогноз PatchTST по компонентам ICEEMDAN."""
    if run_patchtst_iceemdan_backtest is None:
        raise ImportError(
            "PatchTST+ICEEMDAN недоступен: не удалось импортировать torch или PyEMD (EMD-signal)"
        )
    return run_patchtst_iceemdan_backtest(
        returns,
        save_weights_path=save_weights_path,
        collect_forecasts=collect_forecasts
    )


# (key, подпись, префикс файлов, функция запуска)
STRATEGY_SPECS = [
    ('baseline1', 'Baseline 1 (Hist Mean)', 'baseline1', run_baseline1),
    ('baseline2', 'Baseline 2 (StatsForecast)', 'statsforecast', run_baseline2),
    ('patchtst', 'PatchTST', 'patchtst', run_patchtst),
    ('patchtst_iceemdan', 'PatchTST + ICEEMDAN', 'patchtst_iceemdan', run_patchtst_iceemdan),
]

SHORT_LABELS = {
    'baseline1': 'Baseline 1',
    'baseline2': 'StatsF',
    'patchtst': 'PatchTST',
    'patchtst_iceemdan': 'PT-ICEEMDAN',
    'equal_weight': '1/N',
    'spy_buyhold': 'SPY B&H',
}


def compute_period_metrics(entry):
    """Метрики gross/net по периодам full/validation/holdout для одной стратегии."""
    period_metrics = {}
    gross_split = split_by_date(entry['returns'], VALIDATION_END)
    net_split = split_by_date(entry['returns_net'], VALIDATION_END)
    turnover_split = split_by_date(entry['turnover'], VALIDATION_END)
    for period in PERIODS:
        period_metrics[period] = {
            'gross': calculate_metrics(gross_split[period], rf=RF),
            'net': calculate_metrics(net_split[period], rf=RF),
            'avg_turnover': float(turnover_split[period].mean()) if len(turnover_split[period]) else np.nan,
        }
    return period_metrics


def _cleanup_pools():
    """Закрыть пулы воркеров PatchTST-стратегий (страховка при ошибке:
    иначе 8 простаивающих процессов держат память, пока считается следующая)."""
    for mod_name in ('backtesting.backtest_patchtst', 'backtesting.backtest_patchtst_iceemdan'):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, 'close_pool'):
            try:
                mod.close_pool()
            except Exception:
                pass


class _Tee:
    """Дублирует поток вывода в лог-файл (line-buffered, чтобы во время
    долгих прогонов промежуточные результаты были видны через tail -f)."""

    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file = file_handle

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def load_benchmark_returns(data_dir):
    """Дневные лог-доходности бенчмарка (SPY): из кэша или скачать."""
    bench_path = data_dir / "benchmark_log_returns.csv"
    if not bench_path.exists():
        try:
            from data.downloader import download_benchmark
            download_benchmark()
        except Exception as e:
            warnings.warn(
                f"Бенчмарк недоступен ({e}). Buy & Hold пропущен. "
                f"Скачать вручную: python -c \"import sys; sys.path.insert(0,'src'); "
                f"from data.downloader import download_benchmark; download_benchmark()\""
            )
            return None
    bench = pd.read_csv(bench_path, index_col=0, parse_dates=True)
    return bench.iloc[:, 0]


# ============================================================
# MAIN
# ============================================================

def main():
    # Режим PatchTST берётся из config.yaml (fast/full)
    patchtst_mode = config['models']['patchtst'].get('mode', 'full')

    def prompt_yes_no(prompt, default=False):
        suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            ans = input(prompt + suffix).strip().lower()
            if ans == "":
                return default
            if ans in ("y", "yes", "да", "д"):
                return True
            if ans in ("n", "no", "нет", "н"):
                return False
            print("Введите 'y' или 'n'.")

    def prompt_models():
        print("Выберите модели для запуска:")
        print("  1 - Baseline 1 (Историческое среднее)")
        print("  2 - StatsForecast AutoARIMA")
        print("  3 - PatchTST")
        print("  4 - PatchTST + ICEEMDAN")
        while True:
            ans = input("Введите номера через запятую (Enter = все): ").strip()
            if ans == "":
                return {"baseline1", "baseline2", "patchtst", "patchtst_iceemdan"}
            parts = [p.strip() for p in ans.replace(" ", "").split(",") if p.strip()]
            mapping = {"1": "baseline1", "2": "baseline2", "3": "patchtst", "4": "patchtst_iceemdan"}
            selected = {mapping[p] for p in parts if p in mapping}
            if selected:
                return selected
            print("Не удалось распознать выбор. Пример: 1,3")

    # Создаём папку results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Весь вывод (прогресс бэктестов, метрики, warnings) дублируется в лог,
    # чтобы промежуточные результаты были видны во время прогона:
    #   tail -f results/run_log_{timestamp}.txt
    log_path = results_dir / f"run_log_{timestamp}.txt"
    log_file = open(log_path, 'a', encoding='utf-8')
    log_file.write(f"=== Прогон {timestamp}, запущен {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"Лог прогона: {log_path}")

    print("Параметры берутся из config/config.yaml")
    if prompt_yes_no("Скачать данные заново?", default=False):
        from data.downloader import download_and_prepare_data
        download_and_prepare_data()

    # Загружаем данные
    data_dir = Path(__file__).parent / "data" / "raw"
    data_path = data_dir / "log_returns.csv"
    if not data_path.exists():
        print("Ошибка: данные не найдены!")
        print(f"Ожидается файл: {data_path}")
        print("\nЗапустите сначала загрузку данных:")
        print("  python src/data/downloader.py")
        return

    returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
    if DATA_END:
        returns = returns.loc[:DATA_END]
        print(f"Данные обрезаны по backtest.data_end = {DATA_END}")

    selected_models = prompt_models()

    print("=" * 60)
    print("ЗАПУСК ВСЕХ МОДЕЛЕЙ")
    print("=" * 60)
    print(f"Данные: {returns.index[0].date()} — {returns.index[-1].date()}")
    print(f"Акций: {len(returns.columns)}")
    print(f"Train: {TRAIN_WINDOW} дней, Test: {TEST_WINDOW} дней")
    print(f"PatchTST режим: {patchtst_mode.upper()}")
    print(f"Издержки: {COST_RATE:.4%} на единицу оборота")
    print(f"Validation/holdout граница: {VALIDATION_END}")
    print()

    results = {}

    total_steps = len(selected_models)
    step_num = 0

    for key, label, prefix, runner in STRATEGY_SPECS:
        if key not in selected_models:
            continue
        step_num += 1
        print(f"[{step_num}/{total_steps}] {label}...")

        weights_path = results_dir / f"{prefix}_weights_{timestamp}.csv"
        try:
            strategy_returns, forecasts = runner(
                returns,
                save_weights_path=weights_path,
                collect_forecasts=True
            )
        except Exception as e:
            warnings.warn(f"{label} пропущена из-за ошибки: {e}")
            print(f"      ОШИБКА: {e}\n")
            continue
        finally:
            _cleanup_pools()
        forecast_metrics = aggregate_forecast_metrics(forecasts)

        weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        turnover = compute_turnover_series(weights_df, returns)
        returns_net = apply_transaction_costs(strategy_returns, turnover, COST_RATE)

        results[key] = {
            'label': label,
            'prefix': prefix,
            'returns': strategy_returns,
            'returns_net': returns_net,
            'turnover': turnover,
            'forecast_metrics': forecast_metrics,
            'forecasts': forecasts,
            'is_benchmark': False,
        }
        # Сохраняем сразу: падение следующей стратегии не должно терять
        # результаты уже досчитанных (PatchTST full — это часы)
        strategy_returns.to_csv(results_dir / f"{prefix}_returns_{timestamp}.csv")
        returns_net.to_csv(results_dir / f"{prefix}_returns_net_{timestamp}.csv")
        forecasts.to_csv(results_dir / f"{prefix}_forecasts_{timestamp}.csv", index=False)

        full_metrics = calculate_metrics(strategy_returns, rf=RF)
        net_metrics = calculate_metrics(returns_net, rf=RF)
        print(f"      Sharpe: {full_metrics['Sharpe Ratio']:.2f} (net: {net_metrics['Sharpe Ratio']:.2f}), "
              f"оборот: {turnover.mean():.1%}/мес")
        print(f"      RMSE: {forecast_metrics['rmse']:.6f}, MAE: {forecast_metrics['mae']:.6f}, "
              f"Hit Rate: {forecast_metrics['hit_rate']:.2%}")
        print()

    # ------------------------------------------------------------
    # Бенчмарки (считаются всегда — дёшево)
    # ------------------------------------------------------------
    print("Бенчмарки...")

    try:
        ew_returns, ew_weights = run_equal_weight(returns, TRAIN_WINDOW, TEST_WINDOW)
        ew_weights.to_csv(results_dir / f"equal_weight_weights_{timestamp}.csv")
        ew_turnover = compute_turnover_series(ew_weights, returns)
        results['equal_weight'] = {
            'label': 'Equal Weight (1/N)',
            'prefix': 'equal_weight',
            'returns': ew_returns,
            'returns_net': apply_transaction_costs(ew_returns, ew_turnover, COST_RATE),
            'turnover': ew_turnover,
            'forecast_metrics': None,
            'forecasts': None,
            'is_benchmark': True,
        }
        print(f"      1/N Sharpe: {calculate_metrics(ew_returns, rf=RF)['Sharpe Ratio']:.2f}, "
              f"оборот: {ew_turnover.mean():.1%}/мес")
    except Exception as e:
        warnings.warn(f"Бенчмарк 1/N пропущен из-за ошибки: {e}")

    try:
        benchmark_returns = load_benchmark_returns(data_dir)
        if benchmark_returns is not None:
            if DATA_END:
                benchmark_returns = benchmark_returns.loc[:DATA_END]
            spy_returns = run_buy_and_hold(benchmark_returns, returns, TRAIN_WINDOW, TEST_WINDOW)
            spy_turnover = pd.Series(0.0, index=spy_returns.index)  # Buy & Hold не торгует
            results['spy_buyhold'] = {
                'label': 'Buy & Hold (SPY)',
                'prefix': 'spy_buyhold',
                'returns': spy_returns,
                'returns_net': spy_returns.copy(),
                'turnover': spy_turnover,
                'forecast_metrics': None,
                'forecasts': None,
                'is_benchmark': True,
            }
            print(f"      SPY B&H Sharpe: {calculate_metrics(spy_returns, rf=RF)['Sharpe Ratio']:.2f}")
    except Exception as e:
        warnings.warn(f"Бенчмарк SPY пропущен из-за ошибки: {e}")
    print()

    if not results:
        print("Ни одна стратегия не досчиталась — нечего сохранять.")
        return

    # Доходности бенчмарков
    for key in ('equal_weight', 'spy_buyhold'):
        if key in results:
            entry = results[key]
            entry['returns'].to_csv(results_dir / f"{entry['prefix']}_returns_{timestamp}.csv")
            entry['returns_net'].to_csv(results_dir / f"{entry['prefix']}_returns_net_{timestamp}.csv")

    # ------------------------------------------------------------
    # Метрики по периодам и сводные таблицы
    # ------------------------------------------------------------
    for key, entry in results.items():
        entry['period_metrics'] = compute_period_metrics(entry)

    validation_ts = pd.Timestamp(VALIDATION_END)
    for period in PERIODS:
        comparison_data = {}
        for key, entry in results.items():
            pm = entry['period_metrics'][period]
            merged = dict(pm['gross'])
            merged.update({f"{k} (net)": v for k, v in pm['net'].items() if k != 'Num Periods'})
            merged['Avg Turnover'] = pm['avg_turnover']
            if entry['forecasts'] is not None:
                # Метрики прогнозов — по записям того же периода, что и метрики портфеля
                fdf = entry['forecasts']
                if period == 'validation':
                    fdf = fdf[fdf['date'] <= validation_ts]
                elif period == 'holdout':
                    fdf = fdf[fdf['date'] > validation_ts]
                fm = aggregate_forecast_metrics(fdf)
                merged.update({f"Forecast_{k}": v for k, v in fm.items()})
            comparison_data[entry['label']] = merged
        comparison = pd.DataFrame(comparison_data).T
        comparison.to_csv(results_dir / f"comparison_{period}_{timestamp}.csv")

    # JSON с метриками
    metrics_json = {
        'timestamp': timestamp,
        'config': {
            'train_window': TRAIN_WINDOW,
            'test_window': TEST_WINDOW,
            'data_end': DATA_END,
            'risk_free_rate': RF,
            'patchtst_mode': patchtst_mode,
            'padding_patch': config['models']['patchtst'].get('padding_patch', True),
            'cost_rate': COST_RATE,
            'validation_end': VALIDATION_END,
            'iceemdan': config['models'].get('iceemdan', {}),
        },
        'metrics': {},
        'forecast_metrics': {}
    }
    for key, entry in results.items():
        metrics_json['metrics'][key] = entry['period_metrics']
        if entry['forecast_metrics'] is not None:
            metrics_json['forecast_metrics'][key] = entry['forecast_metrics']
    with open(results_dir / f"metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics_json, f, indent=2, default=str)

    # ------------------------------------------------------------
    # Вывод результатов
    # ------------------------------------------------------------
    ordered_keys = [s[0] for s in STRATEGY_SPECS if s[0] in results]
    ordered_keys += [k for k in ('equal_weight', 'spy_buyhold') if k in results]
    labels = [(k, SHORT_LABELS[k]) for k in ordered_keys]

    for variant, title in [('gross', 'БЕЗ ИЗДЕРЖЕК (GROSS)'), ('net', f'С ИЗДЕРЖКАМИ {COST_RATE:.2%} (NET)')]:
        print("=" * 60)
        print(f"ПОРТФЕЛЬНЫЕ МЕТРИКИ, ПОЛНЫЙ ПЕРИОД: {title}")
        print("=" * 60)
        header = f"\n{'Метрика':<25}" + "".join([f"{label:>13}" for _, label in labels])
        print(header)
        print("-" * (25 + 13 * len(labels)))
        for metric in ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return']:
            row = f"{metric:<25}"
            for key, _ in labels:
                value = results[key]['period_metrics']['full'][variant][metric]
                row += f"{value:>13.2f}" if 'Ratio' in metric else f"{value:>13.2%}"
            print(row)
        row = f"{'Avg Turnover/мес':<25}"
        for key, _ in labels:
            row += f"{results[key]['period_metrics']['full']['avg_turnover']:>13.2%}"
        print(row)
        print()

    # Holdout-период (главная таблица для выводов)
    print("=" * 60)
    print(f"SHARPE ПО ПЕРИОДАМ (net): validation <= {VALIDATION_END} < holdout")
    print("=" * 60)
    header = f"\n{'Период':<25}" + "".join([f"{label:>13}" for _, label in labels])
    print(header)
    print("-" * (25 + 13 * len(labels)))
    for period in PERIODS:
        row = f"{period:<25}"
        for key, _ in labels:
            row += f"{results[key]['period_metrics'][period]['net']['Sharpe Ratio']:>13.2f}"
        print(row)

    # Вывод метрик прогнозов (только стратегии)
    forecast_labels = [(k, l) for k, l in labels if results[k]['forecast_metrics'] is not None]
    if forecast_labels:
        print()
        print("=" * 60)
        print("РЕЗУЛЬТАТЫ: МЕТРИКИ ПРОГНОЗОВ")
        print("=" * 60)
        header = f"\n{'Метрика':<25}" + "".join([f"{label:>13}" for _, label in forecast_labels])
        print(header)
        print("-" * (25 + 13 * len(forecast_labels)))
        for metric, fmt in [('rmse', '.6f'), ('mae', '.6f'), ('hit_rate', '.2%')]:
            row = f"{metric.upper():<25}" + "".join(
                [f"{results[key]['forecast_metrics'][metric]:>13{fmt}}" for key, _ in forecast_labels]
            )
            print(row)

    print()
    print(f"Результаты сохранены в: {results_dir}/")
    print(f"  - comparison_full|validation|holdout_{timestamp}.csv")
    print(f"  - metrics_{timestamp}.json")
    print(f"  - *_returns_{timestamp}.csv, *_returns_net_{timestamp}.csv")
    print(f"  - *_forecasts_{timestamp}.csv")
    print(f"  - *_weights_{timestamp}.csv")

    # Визуализация результатов
    try:
        import matplotlib.pyplot as plt

        print("\n" + "=" * 60)
        print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("=" * 60)

        # График кумулятивных доходностей (net — с издержками)
        fig, ax = plt.subplots(figsize=(14, 7))

        for key, label in labels:
            simple_returns = np.exp(results[key]['returns_net']) - 1
            cumulative = (1 + simple_returns).cumprod()
            linestyle = '--' if results[key]['is_benchmark'] else '-'
            ax.plot(cumulative.index, cumulative.values, label=label,
                    linewidth=2, linestyle=linestyle)

        validation_ts = pd.Timestamp(VALIDATION_END)
        date_min = min(e['returns'].index[0] for e in results.values())
        date_max = max(e['returns'].index[-1] for e in results.values())
        if date_min < validation_ts < date_max:
            ax.axvline(validation_ts, color='grey', linestyle=':', linewidth=1.5)
            ax.text(validation_ts, ax.get_ylim()[1] * 0.95, ' validation | holdout',
                    color='grey', fontsize=9, va='top')

        ax.set_title(f'Сравнение кумулятивных доходностей (издержки {COST_RATE:.2%}/оборот)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Рост капитала ($1 → $X)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Сохраняем график
        plot_path = results_dir / f"cumulative_returns_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранен: {plot_path}")

        plt.show()

        # Итоговые значения
        print("\nРост капитала с издержками ($1 → $X):")
        for key, label in labels:
            simple_returns = np.exp(results[key]['returns_net']) - 1
            cumulative = (1 + simple_returns).cumprod()
            print(f"  {label}: $1 → ${cumulative.iloc[-1]:.2f}")

    except ImportError:
        print("\nВизуализация недоступна (matplotlib не установлен)")
    except Exception as e:
        print(f"\nОшибка при создании визуализации: {e}")

    # Анализ весов: Baseline1 против каждой PatchTST-стратегии
    for pt_key in ('patchtst', 'patchtst_iceemdan'):
        if "baseline1" in results and pt_key in results:
            analyze_weight_differences(results_dir, timestamp, results, pt_key)


def analyze_weight_differences(results_dir, timestamp, results, pt_key='patchtst'):
    """
    Анализ различий в весах между Baseline1 и PatchTST-стратегией.
    Только фактические данные, без интерпретаций.
    """
    pt_label = results[pt_key]['label']
    pt_prefix = results[pt_key]['prefix']

    print("\n" + "=" * 60)
    print(f"АНАЛИЗ РАЗЛИЧИЙ ВЕСОВ: BASELINE1 vs {pt_label.upper()}")
    print("=" * 60)

    try:
        # Загружаем веса
        b1_weights_path = results_dir / f"baseline1_weights_{timestamp}.csv"
        pt_weights_path = results_dir / f"{pt_prefix}_weights_{timestamp}.csv"

        if not b1_weights_path.exists() or not pt_weights_path.exists():
            print("Файлы весов не найдены, пропускаем анализ")
            return

        b1_weights = pd.read_csv(b1_weights_path, index_col=0, parse_dates=True)
        pt_weights = pd.read_csv(pt_weights_path, index_col=0, parse_dates=True)

        # Загружаем доходности для анализа худших периодов
        b1_returns = results['baseline1']['returns']
        pt_returns = results[pt_key]['returns']

        # 1. Топ-5 худших периодов для Baseline1
        print("\n📉 TOP-5 ХУДШИХ ПЕРИОДОВ ДЛЯ BASELINE1:")
        print("-" * 60)

        worst_periods = b1_returns.nsmallest(5)
        analysis_results = []

        for date, b1_ret in worst_periods.items():
            pt_ret = pt_returns.get(date, np.nan)
            if date not in b1_weights.index or date not in pt_weights.index:
                continue

            b1_w = b1_weights.loc[date]
            pt_w = pt_weights.loc[date]
            diff = pt_w - b1_w

            # Топ изменения весов
            increased = diff.nlargest(3)
            decreased = diff.nsmallest(3)

            print(f"\n{date.strftime('%Y-%m-%d')}: B1={b1_ret:.2%}, PT={pt_ret:.2%} (разница: {pt_ret-b1_ret:+.2%})")
            print(f"  {pt_label} держит больше: {', '.join([f'{t}:{v:+.1%}' for t, v in increased.items()])}")
            print(f"  {pt_label} держит меньше: {', '.join([f'{t}:{v:+.1%}' for t, v in decreased.items()])}")

            analysis_results.append({
                'date': date,
                'b1_return': b1_ret,
                'pt_return': pt_ret,
                'diff': pt_ret - b1_ret,
                'increased': dict(increased),
                'decreased': dict(decreased)
            })

        # 2. Средние веса по активам
        print("\n\n📊 СРЕДНИЕ РАЗЛИЧИЯ В ВЕСАХ (стратегия - Baseline1):")
        print("-" * 60)

        avg_diff = (pt_weights - b1_weights).mean()
        avg_diff_sorted = avg_diff.sort_values()

        print(f"\n{pt_label} в среднем держит МЕНЬШЕ:")
        for ticker, diff in avg_diff_sorted.head(5).items():
            print(f"  {ticker}: {diff:+.1%}")

        print(f"\n{pt_label} в среднем держит БОЛЬШЕ:")
        for ticker, diff in avg_diff_sorted.tail(5).items():
            print(f"  {ticker}: {diff:+.1%}")

        # 3. Амплитуда изменений весов и оборот
        print("\n\n📈 АМПЛИТУДА ИЗМЕНЕНИЙ ВЕСОВ И ОБОРОТ:")
        print("-" * 60)

        b1_weight_vol = b1_weights.diff().abs().mean().mean()
        pt_weight_vol = pt_weights.diff().abs().mean().mean()
        b1_turnover = results['baseline1']['turnover'].mean()
        pt_turnover = results[pt_key]['turnover'].mean()

        print(f"  Среднее |Δвес| за ребалансировку: Baseline1 {b1_weight_vol:.2%}, "
              f"{pt_label} {pt_weight_vol:.2%} ({pt_weight_vol/b1_weight_vol:.2f}x)")
        print(f"  Средний оборот/мес (против дрейфованных весов): "
              f"Baseline1 {b1_turnover:.1%}, {pt_label} {pt_turnover:.1%}")
        print(f"  (ребалансировка у всех стратегий ежемесячная; различается размер сделок)")

        # 4. Сохраняем анализ в файл
        analysis_path = results_dir / f"weight_analysis_{pt_prefix}_{timestamp}.txt"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"АНАЛИЗ РАЗЛИЧИЙ ВЕСОВ: BASELINE1 vs {pt_label.upper()}\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. TOP-5 ХУДШИХ ПЕРИОДОВ ДЛЯ BASELINE1\n")
            f.write("-" * 40 + "\n")
            for r in analysis_results:
                f.write(f"\n{r['date'].strftime('%Y-%m-%d')}: B1={r['b1_return']:.2%}, PT={r['pt_return']:.2%}\n")
                f.write(f"  Держит больше: {r['increased']}\n")
                f.write(f"  Держит меньше: {r['decreased']}\n")

            f.write("\n\n2. СРЕДНИЕ РАЗЛИЧИЯ В ВЕСАХ\n")
            f.write("-" * 40 + "\n")
            f.write(f"\n{pt_label} в среднем держит МЕНЬШЕ:\n")
            for ticker, diff in avg_diff_sorted.head(5).items():
                f.write(f"  {ticker}: {diff:+.1%}\n")
            f.write(f"\n{pt_label} в среднем держит БОЛЬШЕ:\n")
            for ticker, diff in avg_diff_sorted.tail(5).items():
                f.write(f"  {ticker}: {diff:+.1%}\n")

            f.write("\n\n3. АМПЛИТУДА ИЗМЕНЕНИЙ ВЕСОВ И ОБОРОТ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Среднее |Δвес| за ребалансировку:\n")
            f.write(f"  Baseline1: {b1_weight_vol:.2%}\n")
            f.write(f"  {pt_label}: {pt_weight_vol:.2%} ({pt_weight_vol/b1_weight_vol:.2f}x)\n")
            f.write(f"Средний оборот/мес (против дрейфованных весов):\n")
            f.write(f"  Baseline1: {b1_turnover:.1%}\n")
            f.write(f"  {pt_label}: {pt_turnover:.1%}\n")
            f.write("\nПримечание: ребалансировка у всех стратегий строго ежемесячная,\n")
            f.write("различается только размер изменений весов. Интерпретация различий —\n")
            f.write("в тексте работы, на основе приведённых выше фактических данных.\n")

        print(f"\n✅ Анализ сохранён: {analysis_path}")

    except Exception as e:
        print(f"\n⚠️  Ошибка при анализе весов: {e}")


if __name__ == "__main__":
    main()
