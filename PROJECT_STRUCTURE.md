# Структура проекта

🇬🇧 **English version:** [PROJECT_STRUCTURE_EN.md](PROJECT_STRUCTURE_EN.md)

## Быстрый старт

```bash
pip install -r requirements.txt
python run_all.py
```

Результаты сохраняются в `results/`.

---

## Дерево файлов

```
VKR_Patch/
├── config/
│   └── config.yaml                  # Вся конфигурация эксперимента
│
├── data/
│   ├── raw/                         # Данные с Yahoo Finance
│   │   ├── prices.csv               # Цены акций (Adj Close)
│   │   ├── log_returns.csv          # Лог-доходности активов
│   │   └── benchmark_log_returns.csv # Лог-доходности бенчмарка (SPY)
│   └── cache/iceemdan/              # Кэш декомпозиций (в .gitignore)
│
├── src/
│   ├── data/
│   │   ├── downloader.py            # Загрузка данных + download_benchmark()
│   │   └── preprocessor.py          # Предобработка (log-returns)
│   │
│   ├── decomposition/
│   │   └── iceemdan.py              # CEEMDAN/ICEEMDAN + группировка IMF + кэш
│   │
│   ├── models/
│   │   ├── patchtst.py              # PatchTST: одноканальный + многоканальный (для ICEEMDAN)
│   │   └── patchtst_reference/      # Reference-реализация (yuqinie98/PatchTST)
│   │
│   ├── optimization/
│   │   ├── markowitz.py             # Оптимизатор Марковица (max Sharpe, SLSQP)
│   │   └── covariance.py            # Ковариация (sample / Ledoit-Wolf)
│   │
│   ├── utils/
│   │   ├── forecast_metrics.py      # Метрики прогнозов (RMSE, MAE, hit-rate)
│   │   └── portfolio_metrics.py     # Метрики портфеля, turnover, издержки, сплит по датам
│   │
│   └── backtesting/
│       ├── backtest.py              # Baseline 1: историческое среднее
│       ├── backtest_statsforecast.py # Baseline 2: AutoARIMA
│       ├── backtest_patchtst.py     # PatchTST (сырой ряд)
│       ├── backtest_patchtst_iceemdan.py # PatchTST + ICEEMDAN
│       └── benchmarks.py            # Equal Weight (1/N) и Buy & Hold (SPY)
│
├── scripts/
│   └── precompute_decompositions.py # Параллельный предрасчёт ICEEMDAN (CPU)
│
├── notebooks/
│   ├── colab_run.ipynb              # Запуск на GPU в Google Colab
│   └── portfolio_comparison.py      # Standalone-скрипт (устаревшая автономная копия)
│
├── results/                         # Результаты бэктестов + run_log_*.txt
│
├── README.md / README_EN.md         # Описание проекта (RU / EN)
├── RESULTS.md / RESULTS_EN.md       # Результаты исследования (RU / EN)
├── PROJECT_STRUCTURE.md / _EN.md    # Этот файл (RU / EN)
├── requirements.txt                 # Зависимости Python
├── LICENSE                          # MIT
└── run_all.py                       # Оркестратор: стратегии + бенчмарки + издержки
```

---

## Сравниваемые стратегии

Все используют **одинаковые** окна, ковариацию (Ledoit-Wolf), ограничения и оптимизатор; отличается только оценка μ.

| Стратегия | Оценка μ | Файл |
|---|---|---|
| Baseline 1 | mean(r) × 252 | `backtest.py` |
| Baseline 2 | AutoARIMA(21).mean × 252 | `backtest_statsforecast.py` |
| PatchTST | forecast(21).mean × 252 (сырой ряд) | `backtest_patchtst.py` |
| PatchTST + ICEEMDAN | forecast(21).mean × 252 (по компонентам) | `backtest_patchtst_iceemdan.py` |
| Equal Weight (1/N) | — | `benchmarks.py` |
| Buy & Hold (SPY) | — | `benchmarks.py` |

---

## Пайплайн PatchTST + ICEEMDAN

```
Train-окно актива (1260 дней лог-доходностей)
        │
        ▼  каузальная декомпозиция (только train)
CEEMDAN/ICEEMDAN → K переменных IMF + остаток
        │
        ▼  детерминированная группировка по среднему периоду
3 канала: noise (<5д) | cycle (5–63д) | trend (>63д + остаток)
        │
        ▼  многоканальный PatchTST (channel-independence, общие веса)
прогноз 3 каналов на 21 день  →  сумма каналов
        │
        ▼
μ = mean(сумма прогноза) × 252  →  оптимизация Марковица
```

Декомпозиции кэшируются на диск (`data/cache/iceemdan/`). Ключ кэша — хэш значений окна и параметров, поэтому при смене данных/параметров кэш инвалидируется автоматически и утечка из будущего невозможна.

---

## Оптимизация Марковица

```
max (w'μ - rf) / √(w'Σw)
s.t. Σw = 1,  min_w ≤ w ≤ max_w  (long-only, fully invested)
```

- **μ** — ожидаемые доходности (различаются по методам);
- **Σ** — ковариация Ledoit-Wolf (одинакова для всех);
- солвер — `scipy.optimize.minimize` (SLSQP).

---

## Метрики

**Портфельные** (`src/utils/portfolio_metrics.py`), на месячных простых доходностях с аннуализацией ×12 / √12:

| Метрика | Формула |
|---|---|
| Annual Return (CAGR) | ∏(1+r)^(12/N) − 1 |
| Annual Volatility | std(r) × √12 |
| Sharpe Ratio | (mean(r) − rf_мес) / std(r) × √12 |
| Max Drawdown | min по кумулятивной кривой |
| Calmar Ratio | CAGR / |MaxDD| |
| Turnover | Σ|w_t − w_дрейф| против дрейфованных весов |

Метрики приводятся без издержек (gross) и с издержками (net): `net = (exp(r) − 1) − cost_rate × turnover`.

**Прогнозов** (`src/utils/forecast_metrics.py`): RMSE, MAE, Hit Rate на месячных суммах.

---

## Конфигурация (`config/config.yaml`)

| Блок | Ключевые параметры |
|---|---|
| `data` | `tickers`, `start_date`, `end_date`, `benchmark_ticker` |
| `backtest` | `train_window`, `test_window`, `data_end` (обрезка для тюнинга) |
| `models.patchtst` | `mode` (fast/full), `padding_patch`, `n_workers`, гиперпараметры режима |
| `models.iceemdan` | `trials`, `epsilon`, `seed`, `grouping`, `forecast_noise`, `cache` |
| `optimization` | `risk_free_rate`, `covariance`, `constraints` (веса, long-only) |
| `evaluation` | `transaction_costs.cost_rate`, `split.validation_end` |

### Как изменить набор акций

1. Отредактируйте `data.tickers` в `config/config.yaml`.
2. Перекачайте данные: `python src/data/downloader.py` (или ответьте `y` в `run_all.py`).
3. Кэш декомпозиций чистить не нужно — он привязан к значениям данных.

---

## Запуск

```bash
# Все стратегии + бенчмарки (Enter = все)
python run_all.py

# Фоновый прогон (macOS, не уснёт, переживёт закрытие терминала)
nohup caffeinate -is bash -c 'printf "n\n\n" | python3 run_all.py' > run.log 2>&1 &
tail -f results/run_log_*.txt

# Отдельные стратегии
python src/backtesting/backtest.py                    # Baseline 1
python src/backtesting/backtest_statsforecast.py      # AutoARIMA
python src/backtesting/backtest_patchtst.py           # PatchTST
python src/backtesting/backtest_patchtst_iceemdan.py  # PatchTST + ICEEMDAN

# Предрасчёт декомпозиций (ускоряет ICEEMDAN-прогон)
python scripts/precompute_decompositions.py --all
```

### Выходные файлы (в `results/`, с меткой времени)

- `comparison_full|validation|holdout_<ts>.csv` — сводные таблицы по периодам;
- `metrics_<ts>.json` — метрики (gross/net по периодам) + параметры прогона;
- `<strategy>_returns[_net]_<ts>.csv`, `<strategy>_weights_<ts>.csv`, `<strategy>_forecasts_<ts>.csv`;
- `cumulative_returns_<ts>.png` — график; `weight_analysis_*_<ts>.txt` — анализ весов; `run_log_<ts>.txt` — полный лог.

---

## Зависимости (`requirements.txt`)

```
Data:          yfinance, pandas, numpy
ML/DL:         torch
Time Series:   statsforecast (AutoARIMA), EMD-signal (CEEMDAN/ICEEMDAN)
Optimization:  scipy
Visualization: matplotlib, seaborn
Utils:         pyyaml, tqdm, scikit-learn
```
