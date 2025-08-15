import pytest
import pandas as pd
import numpy as np

from app.features.factory.market_factory import (
    build_market_features,
    FEATURE_VERSION,
    FEATURE_SCHEMA,
)


def _mk_ohlcv(n=30, start="2025-08-01 00:00:00+00:00", symbol="BTC/USDT", exchange="binance", timeframe="1m"):
    ts = pd.date_range(start=start, periods=n, freq="1min", tz="UTC")
    # simple ramp so returns/ema/rsi are deterministic-ish
    close = pd.Series(np.linspace(100.0, 100.0 + (n - 1), n))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(np.linspace(10, 10 + (n - 1) * 0.1, n))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": pd.Series([symbol] * n, dtype="string"),
            "exchange": pd.Series([exchange] * n, dtype="string"),
            "timeframe": pd.Series([timeframe] * n, dtype="string"),
        }
    )
    return df


def test_market_factory_happy_path_basic_shape_and_types():
    df = _mk_ohlcv(n=30)
    out = build_market_features(df)

    # Not empty; with v1 features (e.g., hl_spread), every row has at least one non-NaN feature
    assert not out.empty
    assert len(out) == len(df)

    # Exact schema & ordering
    assert list(out.columns) == list(FEATURE_SCHEMA.keys())

    # Timestamp is tz-aware UTC
    assert "UTC" in str(out["timestamp"].dtype)

    # dt partition exists and looks like YYYY-MM-DD
    assert "dt" in out.columns
    assert out["dt"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()

    # Version tag
    assert out["feature_version"].nunique() == 1
    assert out["feature_version"].iloc[0] == FEATURE_VERSION

    # At least one non-null across feature columns
    feat_cols = [c for c in out.columns if c not in ("timestamp", "dt", "symbol", "exchange", "timeframe", "feature_version")]
    assert out[feat_cols].notna().any(axis=1).all()

    # Monotonic by time
    assert out["timestamp"].is_monotonic_increasing


def test_market_factory_missing_required_column_raises():
    df = _mk_ohlcv(n=10).drop(columns=["volume"])
    with pytest.raises(ValueError) as exc:
        build_market_features(df)
    assert "Missing required columns" in str(exc.value)
    assert "volume" in str(exc.value)


def test_market_factory_standardizes_timestamp_from_ts_column():
    # Provide 'ts' instead of 'timestamp' to exercise standardize_time_column
    base = _mk_ohlcv(n=10).rename(columns={"timestamp": "ts"})
    out = build_market_features(base)
    assert "timestamp" in out.columns
    assert "UTC" in str(out["timestamp"].dtype)
    assert "dt" in out.columns
    # Sanity on metadata carried through
    assert out["symbol"].dtype.name == "string"
    assert out["exchange"].dtype.name == "string"
    assert out["timeframe"].dtype.name == "string"


def test_market_factory_ret1_is_deterministic_on_small_series():
    # 5 points linear ramp: 100,101,102,103,104 → first valid pct change at second row is ~0.01
    df = _mk_ohlcv(n=5, start="2025-08-01 00:00:00+00:00")
    out = build_market_features(df, dropna_final=False)  # keep early NaNs for direct inspect

    # Find the row matching the 2nd timestamp
    t1 = pd.Timestamp("2025-08-01 00:01:00+00:00", tz="UTC")
    row = out[out["timestamp"] == t1]
    assert not row.empty

    # pct change close(101) / close(100) - 1 == 0.01
    val = row["ret_1"].iloc[0]
    assert pytest.approx(val, rel=1e-9, abs=1e-9) == 0.01

    # log return ~ log(101) - log(100)
    lv = row["logret_1"].iloc[0]
    assert pytest.approx(lv, rel=1e-9, abs=1e-9) == np.log(101.0) - np.log(100.0)