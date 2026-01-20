# План воссоздания проекта с нуля: Портфельная оптимизация с PatchTST

**Версия:** v1.1
**Последнее обновление:** 20.01.2026

Этот план рассчитан на пошаговое воссоздание проекта. Каждый шаг выполняется, проверяется и согласуется перед переходом к следующему.

**ВАЖНО:** Модель PatchTST используется в режиме Self-Supervised из официального репозитория:
https://github.com/yuqinie98/PatchTST

---

## Шаг 1. Подготовка окружения

**Цель:** Создать воспроизводимую среду для экспериментов

### Что сделать:

1. **Установить Python 3.x** (рекомендуется 3.10+)

2. **Создать виртуальное окружение:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. **Установить библиотеки:**
```bash
pip install torch numpy pandas scipy scikit-learn
pip install statsforecast  # для AutoARIMA
pip install matplotlib seaborn  # для визуализации
pip install pyyaml jupytext  # для конфигов и notebook
```

4. **Проверить установку:**
```python
import torch
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.backends.mps.is_available()}")  # Apple Silicon
print(f"CUDA: {torch.cuda.is_available()}")  # NVIDIA GPU
```

**Пауза:** Проверь, что все пакеты установлены и импортируются без ошибок.

---

## Шаг 2. Создать структуру проекта

**Цель:** Организовать файлы проекта

### Создать папки:
```bash
mkdir -p config data/raw src/{backtesting,models,optimization,utils} notebooks results
```

### Структура должна быть:
```
VKR_Patch/
├── config/
│   └── config.yaml
├── data/
│   └── raw/
├── src/
│   ├── backtesting/
│   ├── models/
│   ├── optimization/
│   └── utils/
├── notebooks/
├── results/
└── run_all.py
```

**Пауза:** Убедись, что структура создана.

---

## Шаг 3. Создать конфигурационный файл

**Цель:** Единый конфиг для всех параметров

### Создать `config/config.yaml`:
```yaml
# Параметры данных
data:
  tickers:
    - AAPL
    - CVX
    - JNJ
    - JPM
    - KO
    - MSFT
    - PG
    - UNH
    - WFC
    - XOM
  start_date: "2000-01-01"
  end_date: "2025-01-01"

# Параметры бэктеста
backtest:
  train_window: 1260  # 5 лет (252 дня * 5)
  test_window: 21     # 1 месяц (21 торговый день)

# Параметры оптимизации Марковица
optimization:
  risk_free_rate: 0.04              # 4% годовых
  covariance: "ledoit_wolf"         # sample | ledoit_wolf

  constraints:
    long_only: true                 # Только длинные позиции
    fully_invested: true            # Сумма весов = 100%
    min_weight: 0.05                # Минимум 5% в каждый актив
    max_weight: 0.25                # Максимум 25% в один актив
    gross_exposure: null            # Для long-only не используется

# Параметры моделей
models:
  arima:
    max_p: 3
    max_d: 0  # Лог-доходности уже стационарны
    max_q: 3
    stepwise: true  # Ускоряет подбор ~10x

  patchtst:
    mode: "full"  # fast | full

    full:
      input_length: 1260    # 5 лет (согласовано с train_window)
      pred_length: 21       # 1 месяц прогноз
      patch_length: 16
      stride: 8
      d_model: 128
      n_heads: 16
      n_layers: 3
      d_ff: 512
      dropout: 0.2
      use_revin: true       # Reversible Instance Normalization
      mask_ratio: 0.4       # 40% патчей маскируем (self-supervised)
      pretrain_epochs: 30   # Pre-training epochs
      finetune_epochs: 10   # Fine-tuning epochs
      pretrain_lr: 0.0001   # Learning rate
      batch_size: 64
```

**Важно:**
- `train_window: 1260` - согласовано для всех моделей
- `constraints.min_weight: 0.05` - с 10 активами это означает минимум 50% инвестировано
- `pretrain_epochs: 30` для full режима (для экспериментов можно уменьшить)

**Пауза:** Проверь, что файл создан и корректно парсится:
```python
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
print(config['backtest']['train_window'])  # должно вывести 1260
```

---

## Шаг 4. Загрузка данных

**Цель:** Получить цены активов (Adjusted Close)

### Создать скрипт загрузки данных:

```python
# src/data/downloader.py
import yfinance as yf
import pandas as pd
from pathlib import Path
import yaml

# Загрузка конфига
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

tickers = config['data']['tickers']
start = config['data']['start_date']
end = config['data']['end_date']

# Скачиваем данные
data = yf.download(tickers, start=start, end=end)['Adj Close']

# Сохраняем
output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "prices.csv"
data.to_csv(output_path)
print(f"Данные сохранены: {output_path}")
print(f"Форма: {data.shape}")
print(f"Период: {data.index[0]} — {data.index[-1]}")
```

### Запустить загрузку:
```bash
python src/data/downloader.py
```

**Проверки:**
- Нет пустых колонок (все тикеры скачались)
- Используется Adjusted Close (учитывает дивиденды и сплиты)
- Нет большого количества NaN

**Пауза:** Проверь файл `data/raw/prices.csv`.

---

## Шаг 5. Построение лог-доходностей

**Цель:** Подготовить данные для бэктеста

### Создать скрипт преобразования:

```python
# src/data/preprocessor.py
import pandas as pd
import numpy as np
from pathlib import Path

# Загружаем цены
prices_path = Path(__file__).parent.parent.parent / "data" / "raw" / "prices.csv"
prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

# Вычисляем лог-доходности
log_returns = np.log(prices / prices.shift(1))

# Удаляем первую строку (NaN) и любые оставшиеся NaN
log_returns = log_returns.dropna()

# Проверка на inf
if np.isinf(log_returns.values).any():
    print("ВНИМАНИЕ: Есть inf значения!")
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()

# Сохраняем
output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "log_returns.csv"
log_returns.to_csv(output_path)

print(f"Лог-доходности сохранены: {output_path}")
print(f"Форма: {log_returns.shape}")
print(f"Период: {log_returns.index[0]} — {log_returns.index[-1]}")
print(f"NaN: {log_returns.isna().sum().sum()}")
print(f"Inf: {np.isinf(log_returns.values).sum()}")
```

### Запустить:
```bash
python src/data/preprocessor.py
```

**Проверки:**
- Длина данных = N-1 относительно цен
- Нет NaN, inf
- Синхронизация по датам

**Пауза:** Проверь файл `data/raw/log_returns.csv`.

---

## Шаг 6. Оптимизация Марковица

**Цель:** Реализовать максимизацию Sharpe Ratio

### Создать `src/optimization/markowitz.py`:

```python
import numpy as np
from scipy.optimize import minimize

def maximize_sharpe(mu, cov, rf=0.02, min_weight=0.0, max_weight=1.0,
                   long_only=True, fully_invested=True, gross_exposure=None):
    """
    Максимизация Sharpe Ratio с ограничениями.

    Args:
        mu: вектор ожидаемых годовых доходностей (N,)
        cov: матрица ковариации годовая (N, N)
        rf: безрисковая ставка (годовая)
        min_weight: минимальный вес на актив
        max_weight: максимальный вес на актив
        long_only: только длинные позиции
        fully_invested: сумма весов = 1
        gross_exposure: максимальная сумма |весов| (для long/short)

    Returns:
        weights: оптимальные веса (N,)
    """
    n = len(mu)

    # Целевая функция: минимизируем -Sharpe
    def objective(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        # Защита от деления на 0
        if vol < 1e-10:
            return 1e10
        return -(ret - rf) / vol

    # Ограничения
    constraints = []
    if fully_invested:
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    if gross_exposure is not None and not long_only:
        constraints.append({'type': 'ineq', 'fun': lambda w: gross_exposure - np.sum(np.abs(w))})

    # Границы весов
    if long_only:
        bounds = [(min_weight, max_weight) for _ in range(n)]
    else:
        bounds = [(-max_weight, max_weight) for _ in range(n)]

    # Начальное приближение
    w0 = np.ones(n) / n

    # Оптимизация
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if not result.success:
        print(f"ВНИМАНИЕ: Оптимизация не сошлась: {result.message}")

    return result.x
```

### Создать `src/optimization/covariance.py`:

```python
import numpy as np
from sklearn.covariance import LedoitWolf

def compute_covariance(returns, method='ledoit_wolf', annualize=252):
    """
    Оценка ковариационной матрицы.

    Args:
        returns: DataFrame с лог-доходностями
        method: 'sample' | 'ledoit_wolf'
        annualize: множитель для годовой ковариации (252 для дневных данных)

    Returns:
        cov: ковариационная матрица (годовая)
    """
    if method == 'sample':
        cov = returns.cov().values
    elif method == 'ledoit_wolf':
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    return cov * annualize
```

**Пауза:** Проверь импорты и базовую работу функций.

---

## Шаг 7. Baseline 1: Историческое среднее

**Цель:** Реализовать бэктест с μ = историческое среднее

### Создать `src/backtesting/backtest.py`:

Смотри текущую реализацию в проекте. Ключевые моменты:

```python
# Основной цикл бэктеста
while i + TRAIN_WINDOW + TEST_WINDOW <= n:
    # 1. Train данные: последние 1260 дней
    train_data = returns.iloc[i:i + TRAIN_WINDOW]
    # 2. Test данные: следующие 21 день
    test_data = returns.iloc[i + TRAIN_WINDOW:i + TRAIN_WINDOW + TEST_WINDOW]

    # 3. Оценка μ: историческое среднее × 252
    daily_mean = train_data.mean()
    mu = daily_mean.values * 252

    # 4. Оценка Σ: ковариация × 252
    cov = compute_covariance(train_data, method='ledoit_wolf', annualize=252)

    # 5. Оптимизация
    weights = maximize_sharpe(mu, cov, rf=0.04, ...)

    # 6. Доходность портфеля за месяц (buy-and-hold)
    asset_gross = np.exp(test_data.sum(axis=0).values)
    portfolio_gross = np.dot(weights, asset_gross)
    month_return = np.log(portfolio_gross)

    # 7. Сдвиг окна на 21 день
    i += TEST_WINDOW
```

**Важно:**
- Ребалансировка раз в месяц (21 день)
- Внутри месяца buy-and-hold (без ребалансировки)
- Сохранение весов на каждом шаге для анализа

**Пауза:** Запусти короткий тест (1-2 тикера или укороченный период).

---

## Шаг 8. Baseline 2: StatsForecast AutoARIMA

**Цель:** Прогноз μ с помощью ARIMA

### Создать `src/backtesting/backtest_statsforecast.py`:

Ключевое отличие от Baseline 1:

```python
# Вместо исторического среднего делаем прогноз
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Прогноз на 21 день
def forecast_returns_arima(train_returns, horizon=21):
    # Преобразование в long-формат
    df_long = train_returns.stack().reset_index()
    df_long.columns = ['ds', 'unique_id', 'y']

    # ARIMA модель
    model = AutoARIMA(max_p=3, max_d=0, max_q=3, seasonal=False, stepwise=True)
    sf = StatsForecast(models=[model], freq='B', n_jobs=1)

    # Прогноз
    forecast_df = sf.forecast(h=horizon, df=df_long)

    # μ = среднее прогноза × 252
    preds = forecast_df.groupby('unique_id')['AutoARIMA'].mean()
    mu = preds.values * 252

    return mu
```

**Важно:**
- `max_d=0` потому что лог-доходности уже стационарны
- `stepwise=True` ускоряет подбор ~10x
- Fallback на историческое среднее если ARIMA не сошлась

**Пауза:** Проверь, что ARIMA работает на одном тикере.

---

## Шаг 9. PatchTST Self-Supervised

**Цель:** Реализовать прогноз с PatchTST

### Создать `src/models/patchtst.py`:

Архитектура на основе официального репозитория:

```python
import torch
import torch.nn as nn

class PatchTST_SelfSupervised(nn.Module):
    def __init__(self, input_len=1260, pred_len=21, patch_len=16, stride=8,
                 d_model=128, n_heads=16, n_layers=3, d_ff=512,
                 dropout=0.2, mask_ratio=0.4, use_revin=True):
        super().__init__()

        # Число патчей: (1260 - 16) // 8 + 1 = 156
        num_patches = (input_len - patch_len) // stride + 1

        # RevIN (Reversible Instance Normalization)
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=1)

        # Патчинг
        self.patch_len = patch_len
        self.stride = stride

        # Embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.pretrain_head = nn.Linear(d_model, patch_len)  # Для pre-training
        self.forecast_head = nn.Linear(d_model * num_patches, pred_len)  # Для прогноза
```

### Процесс обучения:

1. **Pre-training (self-supervised):**
```python
def pretrain_step(model, x):
    # Маскирование 40% патчей
    patches = create_patches(x)  # (batch, 156, 16)
    masked_patches, mask = mask_random_patches(patches, ratio=0.4)

    # Прогноз замаскированных патчей
    encoded = model.encoder(model.patch_embedding(masked_patches))
    predicted = model.pretrain_head(encoded)

    # Loss только на замаскированных патчах
    loss = mse_loss(predicted[mask], patches[mask])
    return loss
```

2. **Fine-tuning (supervised):**
```python
def finetune_step(model, x, y_true):
    # y_true = следующие 21 день (для supervised обучения prediction head)
    patches = create_patches(x)
    encoded = model.encoder(model.patch_embedding(patches))
    predicted = model.forecast_head(encoded.flatten(1))
    loss = mse_loss(predicted, y_true)
    return loss
```

3. **Forecasting:**
```python
def forecast(model, x):
    # x = последние 1260 дней
    patches = create_patches(x)
    encoded = model.encoder(model.patch_embedding(patches))
    forecast = model.forecast_head(encoded.flatten(1))
    return forecast  # 21 день прогноз
```

**Важно:**
- Pre-training на каждом шаге бэктеста (вычислительно затратно!)
- Device: автовыбор MPS (Apple) / CUDA / CPU
- Параметры из config.yaml

**Пауза:** Проверь, что модель создается и forward pass работает.

---

## Шаг 10. Метрики прогноза

**Цель:** Сравнить прогнозы с фактом

### Создать `src/utils/forecast_metrics.py`:

```python
import numpy as np

def calculate_forecast_metrics(actual, predicted):
    """
    Метрики качества прогноза.

    Args:
        actual: фактические месячные доходности (сумма за 21 день)
        predicted: прогнозные месячные доходности (сумма прогнозов за 21 день)

    Returns:
        dict с RMSE, MAE, Hit Rate
    """
    # Убираем NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # MAE
    mae = np.mean(np.abs(actual - predicted))

    # Hit Rate (совпадение знаков)
    nonzero_mask = (actual != 0) & (predicted != 0)
    if nonzero_mask.sum() > 0:
        hits = np.sign(actual[nonzero_mask]) == np.sign(predicted[nonzero_mask])
        hit_rate = hits.mean()
    else:
        hit_rate = np.nan

    return {'rmse': rmse, 'mae': mae, 'hit_rate': hit_rate}
```

**Как использовать в бэктесте:**

```python
# На каждом шаге сохраняем прогноз и факт
actual_monthly = test_data.sum(axis=0)  # Фактическая месячная доходность
predicted_monthly = forecast.sum(axis=0)  # Прогнозная месячная доходность

# Сохраняем для каждого тикера
forecast_records.append({
    'date': test_data.index[0],
    'ticker': ticker,
    'actual': actual_monthly[ticker],
    'predicted': predicted_monthly[ticker],
    'model': 'PatchTST'
})
```

**Пауза:** Проверь расчет метрик на тестовых данных.

---

## Шаг 11. Портфельные метрики

**Цель:** Рассчитать метрики эффективности портфеля

### Функция расчета метрик:

```python
def calculate_metrics(returns, rf=0.04):
    """
    Метрики портфеля.

    Args:
        returns: Series с месячными лог-доходностями
        rf: безрисковая ставка (годовая)
    """
    # Преобразуем в простые доходности
    simple_returns = np.exp(returns) - 1
    monthly_rf = (1 + rf) ** (1 / 12) - 1
    excess = simple_returns - monthly_rf

    # Annual Return (CAGR)
    annual_return = (1 + simple_returns).prod() ** (12 / len(simple_returns)) - 1

    # Annual Volatility
    annual_vol = simple_returns.std() * np.sqrt(12)

    # Sharpe Ratio
    sharpe = (excess.mean() / simple_returns.std() * np.sqrt(12))

    # Max Drawdown
    cumulative = (1 + simple_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Total Return
    total_return = (1 + simple_returns).prod() - 1

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown,
        'Total Return': total_return
    }
```

**Пауза:** Проверь корректность расчетов.

---

## Шаг 12. Анализ ребалансировки

**Цель:** Понять, как модели меняют веса

### Метрика Turnover:

```python
def calculate_turnover(weights_df):
    """
    Turnover = среднее изменение весов между периодами.

    Args:
        weights_df: DataFrame с весами (index=dates, columns=tickers)

    Returns:
        turnover: Series с turnover для каждого периода
    """
    diff = weights_df.diff().abs()
    turnover = diff.sum(axis=1) / 2  # Делим на 2 (продажа+покупка)
    return turnover
```

### Визуализация в Jupyter:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap весов
sns.heatmap(weights_df.T, cmap='RdYlGn', center=0.1, vmin=0, vmax=0.25)
plt.title('Веса портфеля во времени')
plt.show()

# Траектории весов
for ticker in weights_df.columns:
    plt.plot(weights_df.index, weights_df[ticker], label=ticker)
plt.legend()
plt.show()
```

**Пауза:** Создай визуализации для анализа.

---

## Шаг 13. Пакетный запуск

**Цель:** Запускать все модели одной командой

### Создать `run_all.py`:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--baseline1', action='store_true')
parser.add_argument('--baseline2', action='store_true')
parser.add_argument('--patchtst', action='store_true')
parser.add_argument('--fast', action='store_true', help='PatchTST fast mode')
args = parser.parse_args()

# Запуск выбранных моделей
if args.baseline1:
    run_baseline1(...)
if args.baseline2:
    run_baseline2(...)
if args.patchtst:
    run_patchtst(mode='fast' if args.fast else 'full', ...)

# Сравнение результатов
# Сохранение в CSV, JSON
# Визуализация графиков
```

**Запуск:**
```bash
python run_all.py --baseline1 --baseline2 --patchtst
```

**Пауза:** Проверь, что скрипт работает.

---

## Шаг 14. Jupyter Notebook для Colab

**Цель:** Создать интерактивный анализ

### Создать `notebooks/01_portfolio_comparison.py`:

Используй формат `.py` с магическими комментариями для jupytext:

```python
# %%
# # Портфельная оптимизация с PatchTST
#
# Сравнение трёх подходов к оценке μ

# %%
import pandas as pd
import numpy as np
# ... импорты

# %% [markdown]
# ## 1. Загрузка данных

# %%
log_returns = pd.read_csv('data/raw/log_returns.csv', index_col=0, parse_dates=True)

# ... и так далее
```

### Конвертация в .ipynb:

```bash
jupytext --to ipynb notebooks/01_portfolio_comparison.py
```

**Пауза:** Загрузи в Google Colab и проверь.

---

## Финальные проверки

### 1. Согласованность параметров:
- ✓ Все модели используют `train_window=1260`
- ✓ Все модели используют `test_window=21`
- ✓ Одинаковые constraints для оптимизации
- ✓ Одинаковая безрисковая ставка (`rf=0.04`)

### 2. Корректность метрик:
- ✓ Лог-доходности преобразуются в простые для метрик
- ✓ Risk-free корректно преобразуется (годовая → месячная)
- ✓ Все три модели считают метрики одинаково

### 3. Сохранение результатов:
- ✓ Все результаты сохраняются с timestamp
- ✓ CSV: доходности, веса, прогнозы
- ✓ JSON: метрики
- ✓ PNG: графики

---

## Ключевые параметры проекта

### Данные:
- **Тикеры:** 10 акций S&P 500
- **Период:** 2000-2025
- **Частота:** Дневные данные

### Бэктест:
- **Train:** 1260 дней (5 лет)
- **Test:** 21 день (1 месяц)
- **Ребалансировка:** Раз в месяц

### Оптимизация:
- **Метод:** Максимизация Sharpe Ratio
- **Ковариация:** Ledoit-Wolf (робастная оценка)
- **Risk-free:** 4% годовых
- **Constraints:** min=5%, max=25%, long-only, fully-invested

### Модели:
1. **Baseline 1:** μ = историческое среднее × 252
2. **Baseline 2:** μ = mean(ARIMA_forecast_21d) × 252
3. **PatchTST:** μ = mean(PatchTST_forecast_21d) × 252

### PatchTST параметры:
- Input: 1260 (согласовано с train window)
- Patches: (1260-16)/8+1 = 156
- Pretrain: 30 epochs (self-supervised, mask_ratio=0.4)
- Finetune: 10 epochs (supervised prediction head)

---

## Возможные проблемы и решения

### 1. PatchTST медленно обучается
**Решение:**
- Используй fast mode (`pretrain_epochs=10`)
- Проверь device (MPS/CUDA vs CPU)
- Уменьши batch_size если не хватает памяти

### 2. Constraints слишком жесткие
**Проблема:** min=5% × 10 активов = 50% минимум инвестировано

**Решение:** В `config.yaml` измени:
```yaml
min_weight: 0.0  # Разрешить нулевые веса
max_weight: 0.40  # Больше концентрации
```

### 3. ARIMA не сходится
**Решение:** В коде есть fallback на историческое среднее

### 4. Отрицательный Sharpe
**Причина:** Портфельная доходность < risk-free rate

**Решение:** Это нормально, показывает что стратегия не работает в данный период

---

## Для дипломной работы (ВКР)

### Структура описания:

1. **Введение:**
   - Проблема оценки μ в портфельной теории
   - Сравнение классических и ML подходов

2. **Методология:**
   - Марковиц mean-variance optimization
   - Walk-forward backtesting
   - Три подхода к прогнозу μ

3. **Модели:**
   - Baseline 1: Historical mean (naive benchmark)
   - Baseline 2: ARIMA (statistical approach)
   - PatchTST: Self-supervised transformer (ML approach)

4. **Результаты:**
   - Портфельные метрики
   - Метрики прогнозов
   - Анализ ребалансировки

5. **Выводы:**
   - Сравнение эффективности
   - Trade-offs (точность vs стабильность vs вычислительная сложность)

### Ключевые графики:
- Кумулятивные доходности
- Drawdown траектории
- Heatmap весов портфеля
- Turnover comparison

---

**Автор:** Iurii Krotov
**Репозиторий:** https://github.com/iuriikrot/VKR_Patch
**Лицензия:** Apache 2.0 (PatchTST reference code)
