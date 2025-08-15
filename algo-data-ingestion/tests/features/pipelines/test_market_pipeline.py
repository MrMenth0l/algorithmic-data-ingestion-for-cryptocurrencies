import pytest
import pandas as pd
import numpy as np

fakeredis = pytest.importorskip("fakeredis.aioredis")
from fakeredis.aioredis import FakeRedis  # type: ignore

from app.features.pipelines.market_pipeline import build_and_store_market_features
from app.features.store.redis_store import RedisFeatureStore
from app.features.factory.market_factory import FEATURE_VERSION


def _mk_ohlcv(n=8, start="2025-08-01 00:00:00+00:00", symbol="BTC/USDT", exchange="binance", timeframe="1m"):
    ts = pd.date_range(start=start, periods=n, freq="1min", tz="UTC")
    # deterministic ramp
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


@pytest.mark.asyncio
async def test_pipeline_build_and_store_roundtrip():
    # Fake redis and store injected (so we don't touch real Redis)
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)

    ohlcv = _mk_ohlcv(n=8, symbol="BTC/USDT", timeframe="1m")
    feats, keys = await build_and_store_market_features(ohlcv, store=store, ttl=10)

    # Features produced and cached (1 key per row kept)
    assert not feats.empty
    assert len(keys) == len(feats)

    # First key shape: features:market:BTC-USDT:1m:{epoch}
    t0 = feats.loc[0, "timestamp"]
    epoch0 = int(pd.Timestamp(t0).value // 1_000_000_000)
    assert keys[0] == f"features:market:BTC-USDT:1m:{epoch0}"

    # Read back payload for the first row via API (uses same key derivation)
    payload = await store.read(domain="market", symbol="BTC/USDT", timeframe="1m", ts=t0)
    assert isinstance(payload, dict)
    assert payload.get("feature_version") == FEATURE_VERSION
    # A couple of representative fields exist
    for fld in ("ret_1", "macd", "rsi_14", "hl_spread", "oi_obv"):
        assert fld in payload

    await store.aclose()


@pytest.mark.asyncio
async def test_pipeline_empty_input_returns_empty_and_no_keys():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)

    empty = pd.DataFrame()
    feats, keys = await build_and_store_market_features(empty, store=store)
    assert feats.empty
    assert keys == []

    await store.aclose()


@pytest.mark.asyncio
async def test_pipeline_symbol_sanitization_and_read():
    # Ensure symbols with special chars are sanitized in key but still readable via read()
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)

    ohlcv = _mk_ohlcv(n=3, symbol="avax:usdt", timeframe="5m")
    feats, keys = await build_and_store_market_features(ohlcv, store=store)

    assert len(keys) == len(feats)
    # Key should contain AVAX-USDT (colon replaced with dash, uppercased)
    assert "AVAX-USDT" in keys[0]

    # Read back using the original symbol spelling; store.read() applies same sanitization
    t0 = feats.loc[0, "timestamp"]
    payload = await store.read(domain="market", symbol="avax:usdt", timeframe="5m", ts=t0)
    assert payload is not None
    assert payload.get("feature_version") == FEATURE_VERSION

    await store.aclose()