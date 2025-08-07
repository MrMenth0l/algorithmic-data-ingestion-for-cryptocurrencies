import numpy as np
import pandas as pd
import pytest

from app.features.processors.ta_indicators import (
    compute_cci,
    compute_rsi,
    compute_macd,
    compute_bollinger,
    compute_vwap,
    compute_atr,
    compute_obv,
    compute_roc,
    compute_stochastic,
    compute_adx,
    compute_mfi
)

# Helper to create a small price series
@pytest.fixture
def price_df():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    # create a simple increasing price for test
    price = pd.Series(np.arange(1, 11, dtype=float), index=dates)
    high = price + 1
    low = price - 1
    close = price
    volume = pd.Series(1.0, index=dates)
    df = pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    return df

def test_compute_cci(price_df):
    cci = compute_cci(price_df, window=3, constant=0.015)
    # For window=3 on a linear series, the first two are NaN
    assert np.isnan(cci.iloc[0])
    assert np.isnan(cci.iloc[1])
    # Check third value is 100 because of linear price series CCI calculation
    assert pytest.approx(cci.iloc[2], rel=1e-6) == 100.0

def test_compute_rsi(price_df):
    rsi = compute_rsi(price_df["close"], window=3)
    # First value is NaN until enough data
    assert np.isnan(rsi.iloc[0])
    # Since price is strictly increasing, RSI should be 100 after enough data
    assert pytest.approx(rsi.iloc[3], rel=1e-6) == 100.0

def test_compute_macd(price_df):
    df_macd = compute_macd(price_df["close"], fast=3, slow=6, signal=3)
    # If a DataFrame is returned, extract the macd and signal columns
    assert isinstance(df_macd, pd.DataFrame)
    assert set(["macd", "signal"]).issubset(df_macd.columns)
    macd = df_macd["macd"]
    signal = df_macd["signal"]
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)

def test_compute_bollinger(price_df):
    res = compute_bollinger(price_df["close"], 3, 2)
    # Allow for either tuple of Series or DataFrame
    if isinstance(res, tuple):
        upper, middle, lower = res
        assert all(isinstance(s, pd.Series) for s in (upper, middle, lower))
    else:
        # assume DataFrame
        assert isinstance(res, pd.DataFrame)
        assert set(["upper", "middle", "lower"]).issubset(res.columns)
        upper, middle, lower = res["upper"], res["middle"], res["lower"]
    # first two values should be NaN
    assert np.isnan(upper.iloc[0]) and np.isnan(middle.iloc[0]) and np.isnan(lower.iloc[0])

def test_compute_vwap(price_df):
    vwap = compute_vwap(price_df, window=3)
    assert isinstance(vwap, pd.Series)
    # NaNs for first periods
    assert np.isnan(vwap.iloc[0])

def test_compute_atr(price_df):
    atr = compute_atr(price_df, window=3)
    assert isinstance(atr, pd.Series)
    # first two are NaN
    assert np.isnan(atr.iloc[0]) and np.isnan(atr.iloc[1])

def test_compute_obv(price_df):
    obv = compute_obv(price_df)
    assert isinstance(obv, pd.Series)
    # Since price increases, OBV should monotonically increase or stay same
    diffs = obv.diff().dropna()
    assert all(diffs >= 0)

def test_compute_roc(price_df):
    roc = compute_roc(price_df["close"], window=3)
    assert isinstance(roc, pd.Series)
    # NaNs until window
    assert np.isnan(roc.iloc[0]) and np.isnan(roc.iloc[2])

def test_compute_stochastic(price_df):
    numb = pd.DataFrame({
        "high": price_df["high"],
        "low": price_df["low"],
        "close": price_df["close"]
    })
    stoch = compute_stochastic(numb, 3, 2)
    assert isinstance(stoch, pd.DataFrame)
    # First values NaN
    assert np.isnan(stoch.iloc[0, 0]) and np.isnan(stoch.iloc[1, 0])

def test_compute_adx(price_df):
    adx = compute_adx(price_df, window=3)
    assert isinstance(adx, pd.Series)
    # NaNs until enough data
    assert np.isnan(adx.iloc[0])

def test_compute_mfi(price_df):
    mfi = compute_mfi(price_df, window=3)
    assert isinstance(mfi, pd.Series)
    # NaNs until enough data
    assert np.isnan(mfi.iloc[0])