# tests/ingestion_service/test_feature_ranges.py
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# -------- minimal in-memory feature store matching RedisFeatureStore API -----

def _safe_symbol(symbol: str) -> str:
    return (symbol or "").strip().replace("/", "-").replace(":", "-").upper()

def _epoch_seconds(ts: Any) -> int:
    if isinstance(ts, (int, float)):
        ts = int(ts)
        return ts // 1000 if ts > 10_000_000_000 else ts
    ser = pd.Series([ts])
    if getattr(ser.dt, "tz", None) is not None:
        dt = ser.dt.tz_convert("UTC")
    else:
        dt = ser.dt.tz_localize("UTC")
    return int(dt.astype("int64").iloc[0] // 1_000_000_000)

class _MemStore:
    def __init__(self, namespace: str = "features", default_ttl: Optional[int] = None) -> None:
        self.ns = namespace
        self.ttl = default_ttl
        self.kv: Dict[str, str] = {}
        self.idx: Dict[str, List[Tuple[int, str]]] = {}

    def _key(self, domain: str, symbol: str, timeframe: str, epoch_s: int) -> str:
        return f"{self.ns}:{domain.strip().lower()}:{_safe_symbol(symbol)}:{timeframe.strip().lower()}:{int(epoch_s)}"

    def _index_key(self, domain: str, symbol: str, timeframe: str) -> str:
        return f"{self.ns}:{domain.strip().lower()}:{_safe_symbol(symbol)}:{timeframe.strip().lower()}:_idx"

    async def batch_write(self, items: Sequence[Dict[str, Any]]) -> List[str]:
        keys = []
        for it in items:
            epoch_s = _epoch_seconds(it["ts"])
            k = self._key(it["domain"], it["symbol"], it["timeframe"], epoch_s)
            self.kv[k] = json.dumps(it["payload"], default=str)
            ik = self._index_key(it["domain"], it["symbol"], it["timeframe"])
            self.idx.setdefault(ik, []).append((epoch_s, k))
            self.idx[ik].sort(key=lambda t: t[0])  # keep sorted
            keys.append(k)
        return keys

    async def batch_read(self, queries: Sequence[Tuple[str, str, str, Any]]):
        out: List[Optional[Dict[str, Any]]] = []
        for (domain, symbol, timeframe, ts) in queries:
            epoch_s = _epoch_seconds(ts)
            k = self._key(domain, symbol, timeframe, epoch_s)
            raw = self.kv.get(k)
            out.append(None if raw is None else json.loads(raw))
        return out

    async def range_read(
        self, *, domain: str, symbol: str, timeframe: str, start: int, end: int, limit: int = 200, reverse: bool = False
    ) -> List[Dict[str, Any]]:
        ik = self._index_key(domain, symbol, timeframe)
        rows = self.idx.get(ik, [])
        selected_keys = [k for (score, k) in rows if start <= score <= end]
        if reverse:
            selected_keys = list(reversed(selected_keys))
        selected_keys = selected_keys[:limit]
        out: List[Dict[str, Any]] = []
        for k in selected_keys:
            raw = self.kv.get(k)
            if raw:
                try:
                    out.append(json.loads(raw))
                except Exception:
                    pass
        return out

# ---------- fixtures -------------

@pytest.fixture
def mem_store():
    return _MemStore()

@pytest.fixture
def client(monkeypatch, mem_store):
    # import here so app/routes load only once
    from app.ingestion_service.main import app
    from app.ingestion_service.routes import get_store as routes_get_store

    # Override DI dependency and also monkeypatch direct calls in helpers
    app.dependency_overrides[routes_get_store] = lambda: mem_store
    monkeypatch.setattr("app.ingestion_service.routes.get_store", lambda: mem_store)

    with TestClient(app) as c:
        yield c

    # cleanup overrides for safety
    app.dependency_overrides.clear()

# ---------- test helpers to build dataframes ----------

def _mk_market_df(symbol="BTC/USDT", timeframe="1m", n=5):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    ts = [now - timedelta(minutes=(n - 1 - i)) for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low":  [ 99.0 + i for i in range(n)],
        "close":[100.5 + i for i in range(n)],
        "volume":[10.0 + i for i in range(n)],
        "symbol":[symbol]*n,
        "timeframe":[timeframe]*n,
    })

def _mk_onchain_df(n=3):
    # Keep all timestamps within the same day to satisfy Parquet partition checks
    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    ts = [now + timedelta(minutes=i) for i in range(n)]  # same-day, minute steps
    return pd.DataFrame({
        "timestamp": ts,                      # tz-aware UTC
        "symbol": ["BTC"] * n,
        "metric": ["active_addresses"] * n,
        "value": [1000 + i for i in range(n)],

        # Fields required by the on-chain parquet schema even for Glassnode
        "source": ["glassnode"] * n,
        "contract_address": [None] * n,
        "contract_name": [None] * n,

        # Often present/used downstream; safe default
        "timeframe": ["1d"] * n,
    })


def _mk_social_df(query="bitcoin", n=3):
    # Keep timestamps within the same day and provide all fields the writer schema expects
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    ts = [now + timedelta(minutes=i) for i in range(n)]  # same-day
    return pd.DataFrame({
        # time + core fields used by feature writer
        "ts": ts,                                   # tz-aware UTC
        "user": [f"user{i}" for i in range(n)],
        "text": [f"{query} #{i}" for i in range(n)],
        "sentiment_score": [0.1 * i for i in range(n)],
        "symbol": ["twitter"] * n,
        "timeframe": ["1m"] * n,

        # fields commonly required by social parquet schema
        "id": [f"id-{i}" for i in range(n)],
        "author": [f"user{i}" for i in range(n)],
        "title": [f"{query} title {i}" for i in range(n)],
        "selftext": [None] * n,
        "likes": [0] * n,
        "retweets": [0] * n,
        "score": [0] * n,
        "num_comments": [0] * n,
        "subreddit": ["bitcoin"] * n,
        "content": [f"{query} #{i}" for i in range(n)],
    })

# ---------- tests -------------

def test_market_range_after_ingest(client, mem_store, monkeypatch):
    # stub CCXTAdapter used inside the route
    async def fake_fetch(symbol, timeframe, since=None, limit=None):
        return _mk_market_df(symbol=symbol, timeframe=timeframe, n=5)
    from types import SimpleNamespace
    monkeypatch.setattr("app.ingestion_service.routes.CCXTAdapter", lambda exch: SimpleNamespace(fetch_ohlcv=fake_fetch))

    # 1) Ingest -> writes features into mem_store via route (_write_market_features_to_store)
    r = client.post("/ingest/market/binance", json={"symbol":"BTC/USDT","granularity":"1m","limit":5})
    assert r.status_code == 200, r.text
    assert r.json()["features_written"] >= 1

    # 2) Compute a window and query range endpoint
    df = _mk_market_df(n=5)
    start = int(df["timestamp"].iloc[0].timestamp())
    end   = int(df["timestamp"].iloc[-1].timestamp())
    r2 = client.get("/ingest/features/market/range", params={
        "symbol":"BTC/USDT","timeframe":"1m","start":start,"end":end,"limit":10
    })
    assert r2.status_code == 200, r2.text
    out = r2.json()
    assert out["rows"] >= 1
    assert isinstance(out["data"], list)

def test_onchain_range_after_ingest(client, mem_store, monkeypatch):
    # stub onchain fetch function used in route
    monkeypatch.setattr("app.ingestion_service.routes.fetch_glassnode", lambda *a, **k: _mk_onchain_df(n=3))

    r = client.post("/ingest/onchain/glassnode", json={"symbol":"BTC","metric":"active_addresses","days":1})
    assert r.status_code == 200, r.text
    assert r.json()["features_written"] >= 1

    df = _mk_onchain_df(n=3)
    start = int(df["timestamp"].iloc[0].timestamp())
    end   = int(df["timestamp"].iloc[-1].timestamp())
    r2 = client.get("/ingest/features/onchain/range", params={
        "symbol":"BTC","timeframe":"1d","start":start,"end":end,"limit":10
    })
    assert r2.status_code == 200, r2.text
    out = r2.json()
    assert out["rows"] >= 1

def test_social_range_after_ingest(client, mem_store, monkeypatch):
    # stub social fetch used in route
    monkeypatch.setattr(
        "app.ingestion_service.routes.fetch_twitter_sentiment",
        lambda query, since, until, max_results: _mk_social_df(query=query, n=3),
    )

    r = client.post("/ingest/social/twitter", json={"query":"bitcoin","max_results":5})
    assert r.status_code == 200, r.text
    assert r.json()["features_written"] >= 1

    df = _mk_social_df("bitcoin", n=3)
    start = int(df["ts"].iloc[0].timestamp())
    end   = int(df["ts"].iloc[-1].timestamp())
    r2 = client.get("/ingest/features/social/range", params={
        "topic":"twitter","timeframe":"1m","start":start,"end":end,"limit":10
    })
    assert r2.status_code == 200, r2.text
    out = r2.json()
    assert out["rows"] >= 1