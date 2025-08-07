import numpy as np
import pandas as pd
import pytest

from app.features.processors.ta_indicators import compute_cci, compute_batch_indicators
from app.features.processors.orderbook_features import compute_imbalance_series

@pytest.fixture(scope="module")
def large_df():
    # Create a large synthetic time series for TA and order-book tests
    n = 100_000
    dates = pd.date_range("2025-01-01", periods=n, freq="S")
    price = pd.Series(np.random.rand(n) * 100, index=dates)
    high = price * 1.01
    low = price * 0.99
    close = price
    volume = pd.Series(np.random.rand(n) * 10, index=dates)
    df = pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    # For order-book, duplicate rows with bid/ask
    ob_df = pd.DataFrame({
        "ts": np.repeat(dates.values, 2),
        "price": np.tile(price.values, 2),
        "amount": np.tile(volume.values, 2),
        "side": np.tile(["bid", "ask"], n)
    })
    return df, ob_df

def test_cci_speed(benchmark, large_df):
    df, _ = large_df
    # Benchmark compute_cci with window=20
    result = benchmark(compute_cci, df, 20, 0.015)
    assert isinstance(result, pd.Series)

def test_imbalance_speed(benchmark, large_df):
    _, ob_df = large_df
    # Benchmark imbalance for order-book
    result = benchmark(compute_imbalance_series, ob_df)
    assert isinstance(result, pd.Series)


def test_batch_indicators_speed(benchmark, large_df):
    df, _ = large_df
    # Benchmark the fused CCI+ROC batch processor
    result = benchmark(compute_batch_indicators, df, 20, 0.015)
    assert isinstance(result, pd.DataFrame)
    assert set(["cci", "roc"]).issubset(result.columns)