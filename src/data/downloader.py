"""
Загрузка дневных данных акций с Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


# Параметры данных берём из config/config.yaml
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)
_data_cfg = _config.get("data", {})

TICKERS = _data_cfg.get("tickers", [])
START_DATE = _data_cfg.get("start_date")
END_DATE = _data_cfg.get("end_date")
PRICE_COLUMN = _data_cfg.get("price_column", "Adj Close")
REQUIRED_PRICE_COLUMN = "Adj Close"


def download_stock_data(
    tickers,
    start_date,
    end_date,
    price_column=REQUIRED_PRICE_COLUMN,
    auto_adjust=False
):
    """Скачать ценовой ряд (Adjusted Close) для заданных тикеров."""
    if price_column != REQUIRED_PRICE_COLUMN:
        raise ValueError(
            f"Требуется '{REQUIRED_PRICE_COLUMN}', а запрошено '{price_column}'."
        )

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust
    )

    if price_column not in data.columns:
        raise ValueError(f"Нет колонки '{price_column}' в данных. Доступно: {list(data.columns)}")

    return data[price_column].dropna()


def save_data(data: pd.DataFrame, path):
    """Сохранить данные в CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)


def load_data(path) -> pd.DataFrame:
    """Загрузить данные из CSV."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_data(
    tickers,
    start_date,
    end_date,
    cache_path,
    price_column=REQUIRED_PRICE_COLUMN,
    auto_adjust=False
):
    """
    Получить данные из кеша или скачать.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_data(cache_path)

    prices = download_stock_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        price_column=price_column,
        auto_adjust=auto_adjust
    )
    save_data(prices, cache_path)
    return prices


def download_and_prepare_data():
    """
    Скачивает ценовую колонку из config и добавляет лог-доходности.
    Сохраняет в data/raw/prices.csv и data/raw/log_returns.csv
    """
    if PRICE_COLUMN != REQUIRED_PRICE_COLUMN:
        raise ValueError(
            f"Требуется '{REQUIRED_PRICE_COLUMN}', а в config указано '{PRICE_COLUMN}'."
        )
    print(f"Скачиваем данные для {len(TICKERS)} акций...")
    print(f"Период: {START_DATE} - {END_DATE}")

    # 1. Скачиваем Adjusted Close
    prices = download_stock_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        price_column=PRICE_COLUMN,
        auto_adjust=False
    )

    print(f"\nЗагружено {len(prices)} торговых дней")

    # 2. Добавляем лог-доходности
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()

    # 3. Сохраняем
    save_path = Path(__file__).parent.parent.parent / "data" / "raw"
    save_path.mkdir(parents=True, exist_ok=True)

    prices.to_csv(save_path / "prices.csv")
    log_returns.to_csv(save_path / "log_returns.csv")

    print(f"\nСохранено:")
    print(f"  - prices.csv ({len(prices)} строк)")
    print(f"  - log_returns.csv ({len(log_returns)} строк)")

    return prices, log_returns


if __name__ == "__main__":
    prices, returns = download_and_prepare_data()

    print("\n" + "="*50)
    print("Цены (первые 5 строк):")
    print(prices.head())

    print("\n" + "="*50)
    print("Лог-доходности (первые 5 строк):")
    print(returns.head())
