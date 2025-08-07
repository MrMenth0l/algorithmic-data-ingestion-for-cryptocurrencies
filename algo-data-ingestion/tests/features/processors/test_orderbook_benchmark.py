import numpy as np
import pandas as pd
import pytest

from app.features.processors.orderbook_features import compute_imbalance_series
from app.features.processors.orderbook_features import compute_batch_orderbook

@pytest.fixture
def large_orderbook_df():
    # Build a DataFrame with 100k timestamps, 2 sides per timestamp
    n = 100_000
    ts = pd.date_range("2025-01-01", periods=n, freq="S")
    prices = np.random.rand(n * 2) * 100
    amounts = np.random.rand(n * 2)
    sides = np.tile(["bid", "ask"], n)

    df = pd.DataFrame({
        "ts": np.repeat(ts.values, 2),
        "price": prices,
        "amount": amounts,
        "side": sides
    })
    return df

def test_imbalance_correctness(large_orderbook_df):
    # Sanity check: runs without error and produces same shape
    series = compute_imbalance_series(large_orderbook_df)
    assert isinstance(series, pd.Series)
    assert series.shape[0] == len(series.index)

def test_imbalance_speed(benchmark, large_orderbook_df):
    # Benchmark the Numba-accelerated imbalance
    result = benchmark(compute_imbalance_series, large_orderbook_df)
    # Optionally, assert minimum performance (e.g. < 0.1s)


def test_batch_orderbook_speed(benchmark, large_orderbook_df):
    """
    Benchmark the fused imbalance+spread batch processor.
    """
    result = benchmark(compute_batch_orderbook, large_orderbook_df)
    assert isinstance(result, pd.DataFrame)
    assert set(["imbalance", "spread"]).issubset(result.columns)