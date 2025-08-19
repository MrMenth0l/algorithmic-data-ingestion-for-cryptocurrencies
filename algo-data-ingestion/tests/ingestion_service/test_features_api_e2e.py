import asyncio
import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

fakeredis = pytest.importorskip("fakeredis.aioredis")
from fakeredis.aioredis import FakeRedis  # type: ignore

from app.ingestion_service.main import app
from app.features.store.redis_store import RedisFeatureStore, get_store
from app.features.store import redis_store as rs


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake_store(monkeypatch):
    """
    Replace DI singleton with a RedisFeatureStore that uses fakeredis.
    """
    r = FakeRedis(decode_responses=True)
    store = RedisFeatureStore(url="redis://fake", namespace="features", default_ttl=None, redis_client=r)

    # Make the singleton return our fake store
    rs._store_singleton = store

    # Also patch the symbol the route calls directly
    monkeypatch.setattr("app.ingestion_service.routes.get_store", lambda: store, raising=True)

    # (Keeping this doesn't hurt, but it's not what the route uses)
    app.dependency_overrides[get_store] = lambda: store

    yield store

    app.dependency_overrides.clear()
    asyncio.get_event_loop().run_until_complete(store.aclose())

def _mk_df(n: int = 30):
    # Generate n 1-min bars, tz-naive (route will tz_localize to UTC)
    base = pd.Timestamp("2025-08-01 00:00:00")
    ts = pd.date_range(base, periods=n, freq="1min", tz=None)
    # simple, monotonic OHLCV
    opens = pd.Series(range(100, 100 + n), dtype="float")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": opens + 0.5,
            "low": opens - 0.5,
            "close": opens + 0.01,    # tiny drift so ret1 != 0
            "volume": pd.Series([10.0 + i for i in range(n)], dtype="float"),
            "symbol": pd.Series(["BTCUSDT"] * n, dtype="string"),
            "exchange": pd.Series(["binance"] * n, dtype="string"),
            "timeframe": pd.Series(["1m"] * n, dtype="string"),
        }
    )
    return df


def test_ingest_then_retrieve_features(client, fake_store, monkeypatch):
    """
    /ingest/market -> features written to Redis by route -> /ingest/features/market returns them.
    """
    # Mock CCXTAdapter in routes
    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return _mk_df(30)

    from types import SimpleNamespace
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )

    # 1) Ingest
    r = client.post(
        "/ingest/market/binance",
        json={"symbol": "BTC/USDT", "granularity": "1m", "limit": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["features_written"] >= 1

    # 2) Build the two epoch seconds we expect were written (route localizes to UTC)
    t0 = int(datetime(2025, 8, 1, 0, 14, tzinfo=timezone.utc).timestamp())
    t1 = t0 + 60

    # 3) Retrieve features for both timestamps
    r2 = client.get(
        "/ingest/features/market",
        params={"symbol": "BTC/USDT", "timeframe": "1m", "ts": [t0, t1]},
    )
    assert r2.status_code == 200, r2.text
    out = r2.json()
    assert out["rows"] in (1, 2)  # depending on NaN sanitization some rows may be dropped
    assert isinstance(out["data"], list)
    for row in out["data"]:
        assert "timestamp" in row
        # payload keys may vary, but at least one engineered feature should exist
        assert any(k in row for k in ("ret1", "rsi_14", "hl_spread", "oi_obv"))


def test_cache_miss_returns_empty(client, fake_store):
    """
    If keys are not present in Redis, the GET should return rows=0 data=[].
    """
    now = int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())
    r = client.get(
        "/ingest/features/market",
        params={"symbol": "ETH/USDT", "timeframe": "1m", "ts": [now, now + 60]},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["rows"] == 0
    assert j["data"] == []


def test_metrics_increment_after_ops(client, fake_store, monkeypatch):
    """
    After a write + read flow, /metrics should include our custom counters/histogram buckets.
    """
    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return _mk_df(30)

    from types import SimpleNamespace
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv),
    )

    # do an ingest (writes to Redis)
    r1 = client.post("/ingest/market/binance", json={"symbol": "BTC/USDT", "granularity": "1m", "limit": 2})
    assert r1.status_code == 200

    # do a read (reads from Redis)
    t0 = int(datetime(2025, 8, 1, 0, 14, tzinfo=timezone.utc).timestamp())
    client.get("/ingest/features/market", params={"symbol": "BTC/USDT", "timeframe": "1m", "ts": [t0]})

    # scrape metrics text
    r2 = client.get("/metrics/")
    assert r2.status_code == 200
    text = r2.text

    # basic presence checks
    assert "feature_writes_total" in text
    assert "feature_reads_total" in text
    # one of hits/misses must be present (depending on retrieval result)
    assert ("feature_hits_total" in text) or ("feature_misses_total" in text)
    assert "feature_op_latency_seconds_bucket" in text