# VKR: PatchTST для портфельной оптимизации Марковица

## Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Запуск всех моделей
python run_all.py
```

Результаты сохраняются в `results/`.

---

## Тема исследования

Применение модели PatchTST Self-Supervised для прогнозирования ожидаемых доходностей в задаче портфельной оптимизации по Марковицу.

## Цель

Эмпирически проверить гипотезу о том, что замена исторических средних на прогнозы модели PatchTST в качестве оценки ожидаемых доходностей улучшает качество портфеля Марковица.

## Формальная постановка

**Классическая задача Марковица** - максимизация коэффициента Шарпа:

```
max  (w'μ - r_f) / sqrt(w'Σw)
s.t. Σw_i = 1, 0.05 <= w_i <= 0.25
```

![Markowitz formula](docs/markowitz_formula.png)

где:
- **w** — вектор весов активов
- **μ** — вектор ожидаемых доходностей (в проекте сравниваются способы оценки)
- **Σ** — ковариационная матрица доходностей
- **r_f** — безрисковая ставка (берется из `config/config.yaml`)

## Подходы к оценке μ

Все три метода используют одинаковые окна бэктеста:
- Окно обучения: `backtest.train_window` (по умолчанию 1260 дней, 5 лет)
- Горизонт прогноза: `backtest.test_window` (по умолчанию 21 день, 1 месяц)

| Подход | Оценка μ | Описание |
|--------|----------|----------|
| **Baseline 1** | mean(r) × 252 | Классический Марковиц (историческое среднее) |
| **Baseline 2** | AutoARIMA.mean × 252 | StatsForecast AutoARIMA |
| **PatchTST** | forecast(21).mean × 252 | Self-Supervised Transformer |

## PatchTST Self-Supervised

**Источник:** https://github.com/yuqinie98/PatchTST

- Реализация: `src/models/patchtst_reference/`
- Режимы `fast` / `full` настраиваются в `config/config.yaml`
- Авто-выбор устройства: MPS (macOS) → CUDA → CPU

## Данные

- **Активы:** 20 акций из 10 секторов S&P 500 (задаются в `config/config.yaml`)
- **Период:** 2010-01-01 — 2025-01-01
- **Источник:** Yahoo Finance (Adjusted Close)
- **Файлы:** `data/raw/prices.csv`, `data/raw/log_returns.csv`

## Конфигурация

Все параметры проекта задаются в `config/config.yaml`:
- `data` — тикеры и период данных
- `backtest` — окна обучения и теста
- `models` — настройки PatchTST и AutoARIMA (StatsForecast)
- `optimization` — безрисковая ставка, метод ковариации, ограничения весов
  - `covariance`: `sample` или `ledoit_wolf`
  - `gross_exposure` используется только при `long_only=false`

## Структура проекта

```
VKR_Patch/
├── run_all.py                    # Запуск всех моделей
├── config/config.yaml            # Конфигурация эксперимента
├── data/raw/                     # Данные (prices, log_returns)
├── src/
│   ├── data/                     # Загрузка и предобработка
│   ├── models/
│   │   ├── patchtst.py           # PatchTST Self-Supervised
│   │   └── patchtst_reference/   # Reference реализация
│   ├── optimization/
│   │   ├── markowitz.py          # Оптимизатор Марковица
│   │   └── covariance.py         # Оценка ковариации
│   ├── backtesting/
│   │   ├── backtest.py           # Baseline 1 (историческое среднее)
│   │   ├── backtest_statsforecast.py  # Baseline 2 (AutoARIMA)
│   │   └── backtest_patchtst.py  # PatchTST
│   └── utils/
│       └── forecast_metrics.py   # Метрики прогнозов
├── notebooks/
│   └── 01_portfolio_comparison.ipynb  # Colab notebook
├── docs/                         # Документация
└── results/                      # Результаты экспериментов
```

## Установка

```bash
python3 -m pip install -r requirements.txt
```

## Запуск

### Полный запуск (интерактивно):

```bash
python3 run_all.py
```

Скрипт спросит:
- нужно ли скачать данные
- какие модели запускать

### Отдельные бэктесты:

```bash
# Baseline 1: Историческое среднее
python3 src/backtesting/backtest.py

# Baseline 2: StatsForecast AutoARIMA
python3 src/backtesting/backtest_statsforecast.py

# PatchTST Self-Supervised
python3 src/backtesting/backtest_patchtst.py
```

### Colab Notebook:

Открыть `notebooks/01_portfolio_comparison.ipynb` в Google Colab.

## Результаты и метрики

- Метрики считаются по месячным доходностям (ребалансировка раз в месяц).
- Веса портфеля сохраняются в `results/*_weights.csv`.

| Метрика | Описание |
|---------|----------|
| Sharpe Ratio | Доходность на единицу риска |
| Annual Return | Годовая доходность (CAGR) |
| Annual Volatility | Годовая волатильность |
| Max Drawdown | Максимальная просадка |
| Total Return | Общая доходность за период |
