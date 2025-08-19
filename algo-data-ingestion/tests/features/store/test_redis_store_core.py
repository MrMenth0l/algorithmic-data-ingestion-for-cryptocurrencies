import asyncio
import pytest
import pandas as pd

import os
from prometheus_client import generate_latest
from app.features.store import redis_store as rs
from app.ingestion_service.utils import _METRICS_REGISTRY

fakeredis = pytest.importorskip("fakeredis.aioredis")  # ensures dev dep present
from fakeredis.aioredis import FakeRedis  # type: ignore

from app.features.store.redis_store import RedisFeatureStore


@pytest.mark.asyncio
async def test_single_write_read_roundtrip():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(
        url="redis://fake", namespace="features", default_ttl=5, redis_client=r
    )

    key = await store.write(
        domain="market",
        symbol="BTC/USDT",
        timeframe="1m",
        ts=0,
        payload={"feat": 1.23},
    )
    assert key == "features:market:BTC-USDT:1m:0"

    out = await store.read(
        domain="market", symbol="BTC/USDT", timeframe="1m", ts=0
    )
    assert out == {"feat": 1.23}

    await store.aclose()


@pytest.mark.asyncio
async def test_batch_write_and_batch_read():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)

    items = [
        {
            "domain": "market",
            "symbol": "ETH/USDT",
            "timeframe": "1m",
            "ts": 1,
            "payload": {"x": 2},
        },
        {
            "domain": "market",
            "symbol": "ETH/USDT",
            "timeframe": "1m",
            "ts": 2,
            "payload": {"x": 3},
        },
    ]
    keys = await store.batch_write(items)
    assert keys == [
        "features:market:ETH-USDT:1m:1",
        "features:market:ETH-USDT:1m:2",
    ]

    vals = await store.batch_read(
        [
            ("market", "ETH/USDT", "1m", 1),
            ("market", "ETH/USDT", "1m", 2),
            ("market", "ETH/USDT", "1m", 3),  # miss
        ]
    )
    assert vals == [{"x": 2}, {"x": 3}, None]

    await store.aclose()


@pytest.mark.asyncio
async def test_ttl_expiry():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", default_ttl=1, redis_client=r)

    await store.write(
        domain="market",
        symbol="SOL/USDT",
        timeframe="5m",
        ts=3,
        payload={"v": 7},
        ttl=1,  # explicit
    )
    # allow TTL to elapse
    await asyncio.sleep(1.1)

    out = await store.read(
        domain="market", symbol="SOL/USDT", timeframe="5m", ts=3
    )
    assert out is None

    await store.aclose()


@pytest.mark.asyncio
async def test_epoch_parsing_variants():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)

    # Variants for the same instant (Unix epoch start)
    variants = [
        0,                                  # seconds
        0.0,                                # float seconds
        "1970-01-01T00:00:00Z",             # ISO string
        pd.Timestamp("1970-01-01T00:00:00Z"),    # pandas Timestamp tz-aware
    ]
    for i, ts in enumerate(variants):
        key = await store.write(
            domain="market",
            symbol=f"X{i}/USDT",
            timeframe="1m",
            ts=ts,
            payload={"i": i},
        )
        # key should always resolve to epoch 0
        assert key.endswith(":0"), f"bad key for {ts!r}: {key}"
        out = await store.read(
            domain="market", symbol=f"X{i}/USDT", timeframe="1m", ts=ts
        )
        assert out == {"i": i}

    await store.aclose()


@pytest.mark.asyncio
async def test_symbol_sanitization_in_key():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)

    key = await store.write(
        domain="market",
        symbol="avax:usdt",  # colon should be sanitized too
        timeframe="15m",
        ts=42,
        payload={"ok": True},
    )
    assert key == "features:market:AVAX-USDT:15m:42"
    await store.aclose()


# Additional tests
@pytest.mark.asyncio
async def test_epoch_ms_key():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)
    # milliseconds should be normalized down to seconds in the key
    key = await store.write(
        domain="market",
        symbol="BTC/USDT",
        timeframe="1m",
        ts=1_700_000_000_000,  # ms
        payload={"ok": True},
    )
    assert key.endswith(":1700000000")
    await store.aclose()


def test_missing_url_raises():
    with pytest.raises(RuntimeError):
        RedisFeatureStore(url="")


def test_get_store_synthesizes_url(monkeypatch):
    # Reset singleton
    rs._store_singleton = None

    # Dummy settings object with no URL so env vars are used
    class Dummy:
        REDIS_URL = None
        REDIS_HOST = None
        REDIS_PORT = None
        REDIS_DB = None
        FEATURE_TTL_SEC = "5"
        FEATURE_NAMESPACE = "featns"

    monkeypatch.setattr(rs, "settings", Dummy())

    # Env vars synthesize URL
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("REDIS_HOST", "myredis")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")

    store = rs.get_store()
    assert isinstance(store, RedisFeatureStore)
    assert store._url == "redis://myredis:6380/2"

    # Cleanup
    rs._store_singleton = None


@pytest.mark.asyncio
async def test_metrics_exposed_after_ops():
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", redis_client=r)

    # perform a write and a read to trigger counters/histogram
    await store.write(
        domain="market",
        symbol="BTC/USDT",
        timeframe="1m",
        ts=0,
        payload={"v": 1},
    )
    await store.read(domain="market", symbol="BTC/USDT", timeframe="1m", ts=0)

    # Scrape the custom registry and ensure metric families exist
    text = generate_latest(_METRICS_REGISTRY).decode("utf-8")
    assert "feature_writes_total" in text
    assert "feature_reads_total" in text
    assert "feature_hits_total" in text or "feature_misses_total" in text
    assert "feature_op_latency_seconds_bucket" in text

    await store.aclose()