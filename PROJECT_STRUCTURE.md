# Структура проекта VKR_Patch

## Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Запуск всех моделей (рекомендуется)
python run_all.py

# 3. Или быстрый режим PatchTST (для отладки)
python run_all.py --fast
```

Результаты сохраняются в `results/`.

---

## Дерево файлов

```
VKR_Patch/
├── config/
│   └── config.yaml                 # Конфигурация эксперимента
│
├── data/
│   └── raw/                        # Сырые данные с Yahoo Finance
│       ├── prices.csv              # Цены акций
│       └── log_returns.csv         # Лог-доходности
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py           # Загрузка данных с Yahoo Finance
│   │   └── preprocessor.py         # Предобработка (log-returns)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patchtst.py             # PatchTST Self-Supervised модель
│   │   └── patchtst_reference/     # Reference реализация PatchTST
│   │       ├── PatchTST_backbone.py
│   │       ├── PatchTST_layers.py
│   │       └── patchTST_selfsupervised.py
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── markowitz.py            # Оптимизатор Марковица (max Sharpe)
│   │   └── covariance.py           # Оценка ковариации (sample / Ledoit-Wolf)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── forecast_metrics.py     # Метрики прогнозов (MAE, RMSE, DA)
│   │
│   └── backtesting/
│       ├── __init__.py
│       ├── backtest.py             # Baseline 1: историческое среднее
│       ├── backtest_statsforecast.py  # Baseline 2: StatsForecast AutoARIMA
│       └── backtest_patchtst.py    # PatchTST Self-Supervised
│
├── notebooks/
│   └── portfolio_comparison.py     # Standalone скрипт (все три метода)
│
├── results/                        # Результаты бэктестов
│
├── .gitignore
├── LICENSE                         # MIT лицензия
├── README.md                       # Описание проекта
├── RESULTS.md                      # Результаты исследования
├── PROJECT_STRUCTURE.md            # Этот файл
├── requirements.txt                # Зависимости Python
└── run_all.py                      # Запуск всех моделей + сохранение в results/
```

---

## Три подхода к оценке μ

Все три метода используют **одинаковые параметры бэктеста** (из `config/config.yaml`):
- **TRAIN_WINDOW = 1260 дней** (5 лет)
- **TEST_WINDOW = 21 день** (1 месяц)
- **RF = 0.04** (безрисковая ставка)

| Подход | Оценка μ | Вход | Файл |
|--------|----------|------|------|
| **Baseline 1** | mean(r) × 252 | 1260 дней | `backtest.py` |
| **Baseline 2** | AutoARIMA(21).mean × 252 | 1260 дней | `backtest_statsforecast.py` |
| **PatchTST** | forecast(21).mean × 252 | 1260 дней | `backtest_patchtst.py` |

---

## PatchTST Self-Supervised

**Источник:** https://github.com/yuqinie98/PatchTST

### Архитектура

```
Вход: 252 дня (1 год)
    ↓
Patching: 30 патчей (patch=16, stride=8)
    ↓
Embedding: Linear(16 → 128)
    ↓
Positional Encoding
    ↓
Transformer Encoder (3 слоя, 16 голов)
    ↓
Prediction Head → 21 день
    ↓
μ = mean(forecast) × 252
```

### Self-Supervised Pre-training

```
1. Маскируем 15% патчей случайно
2. Модель учится восстанавливать замаскированные патчи
3. Loss = MSE(predicted_patches, real_patches)
```

### Параметры модели (full mode)

```yaml
patchtst:
  input_length: 252         # 1 год
  pred_length: 21           # 1 месяц
  patch_length: 16
  stride: 8
  d_model: 128
  n_heads: 16
  n_layers: 3
  d_ff: 512
  dropout: 0.1
  use_revin: true
  mask_ratio: 0.15          # Для self-supervised pretraining
  pretrain_epochs: 20
  finetune_epochs: 10
  pretrain_lr: 0.005
  batch_size: 64
```

---

## Оптимизация Марковица

```
max (w'μ - rf) / √(w'Σw)
s.t. Σw = 1, w ≥ 0
```

- **μ** — вектор ожидаемых доходностей (различается по методам)
- **Σ** — ковариационная матрица (одинаковая для всех методов)
- **Ограничения:** long-only, fully invested

---

## Метрики для сравнения

| Метрика | Описание | Формула |
|---------|----------|---------|
| **Sharpe Ratio** | Доходность на единицу риска | (R - Rf) / σ |
| **Annual Return** | Годовая доходность | mean(r) × 12 |
| **Annual Volatility** | Годовая волатильность | std(r) × √12 |
| **Max Drawdown** | Максимальная просадка | max(peak - trough) |
| **Total Return** | Общая доходность | exp(Σr) - 1 |

---

## Пайплайн эксперимента

```
┌─────────────────────────────────────────────────────────────────┐
│                         ДАННЫЕ                                   │
│  Yahoo Finance → Adj Close → Log Returns                         │
│  20 акций S&P 500 из 10 секторов, 2010-2025                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                СКОЛЬЗЯЩЕЕ ОКНО (одинаковое для всех)            │
│                                                                  │
│  Train: 1260 дней (5 лет) → Test: 21 день (1 месяц)             │
│  Сдвиг: 21 день                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ПРОГНОЗ μ (ожидаемые доходности)             │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Baseline 1  │  │  Baseline 2  │  │   PatchTST   │          │
│  │  Historical  │  │    ARIMA     │  │Self-Supervised│          │
│  │    Mean      │  │              │  │              │          │
│  │  (1260 дней) │  │  (1260 дней) │  │  (1260 дней) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ОПТИМИЗАЦИЯ МАРКОВИЦА                        │
│                                                                  │
│  max (w'μ - rf) / √(w'Σw)                                       │
│  s.t. Σw = 1, w ≥ 0                                             │
│                                                                  │
│  Σ — ковариационная матрица (одинаковая для всех подходов)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    СРАВНЕНИЕ РЕЗУЛЬТАТОВ                        │
│                                                                  │
│  Sharpe, MaxDD, Annual Return, Annual Volatility, Total Return  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Запуск

### Все модели сразу (рекомендуется):

```bash
# Запуск всех трёх моделей + сохранение результатов в results/
python run_all.py

# Быстрый режим PatchTST (для отладки)
python run_all.py --fast
```

Результаты сохраняются в `results/`:
- `comparison_YYYYMMDD_HHMMSS.csv` — сводная таблица метрик
- `metrics_YYYYMMDD_HHMMSS.json` — метрики в JSON
- `*_returns_YYYYMMDD_HHMMSS.csv` — доходности каждой модели

### Отдельные бэктесты:

```bash
# Baseline 1: Историческое среднее
python src/backtesting/backtest.py

# Baseline 2: StatsForecast AutoARIMA
python src/backtesting/backtest_statsforecast.py

# PatchTST Self-Supervised
python src/backtesting/backtest_patchtst.py
```

---

## Зависимости (requirements.txt)

```
# Data
yfinance, pandas, numpy

# ML/DL
torch, pytorch-lightning

# Time Series
statsforecast (AutoARIMA)

# Optimization
scipy, cvxpy

# Visualization
matplotlib, seaborn, plotly

# Utils
pyyaml, tqdm, scikit-learn

# Testing
pytest
```
