# tests/ingestion_service/test_onchain_social_e2e.py
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.ingestion_service import routes
from app.ingestion_service.main import app  # whatever module exposes your FastAPI `app`

import pytest

@pytest.fixture(autouse=True)
def _override_store(mem_store):
    # Ensure all /features/* endpoints use the in-memory store
    app.dependency_overrides[routes.provide_store] = lambda: mem_store
    yield
    app.dependency_overrides.pop(routes.provide_store, None)

# --- local TestClient fixture ---
@pytest.fixture
def client():
    from app.ingestion_service.main import app
    return TestClient(app)

# --- stub parquet writes so tests don't touch disk ---
@pytest.fixture(autouse=True)
def _stub_parquet(monkeypatch, tmp_path):
    out = tmp_path / "ok.parquet"
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, parts: str(out)
    )

# --- tiny in-memory async store that mimics RedisFeatureStore ---
class _MemStore:
    def __init__(self):
        # key: (domain, symbol, timeframe, epoch_s) -> payload dict
        self._kv = {}

    @staticmethod
    def _epoch_seconds(ts):
        if isinstance(ts, (int, float)):
            # int could be ms, normalize like production helper
            return int(ts // 1000) if ts > 10_000_000_000 else int(ts)
        s = pd.Series([ts])
        dt = pd.to_datetime(s, utc=True)
        ns = dt.astype("int64").iloc[0]
        return int(ns // 1_000_000_000)

    async def batch_write(self, items):
        for it in items:
            k = (
                it["domain"],
                str(it["symbol"]),
                str(it["timeframe"]),
                self._epoch_seconds(it["ts"]),
            )
            self._kv[k] = dict(it["payload"])
        return list(self._kv.keys())

    async def batch_read(self, queries):
        out = []
        for (domain, symbol, timeframe, ts) in queries:
            k = (domain, str(symbol), str(timeframe), self._epoch_seconds(ts))
            out.append(self._kv.get(k))
        return out


@pytest.fixture
def mem_store(monkeypatch):
    store = _MemStore()
    # Override get_store used inside routes to return our in-mem store
    monkeypatch.setattr("app.ingestion_service.routes.get_store", lambda: store)
    return store


def _mk_onchain_df():
    # Two daily points ending "now" (NAIVE timestamps, the app will localize)
    now = pd.Timestamp.utcnow().floor("D")
    ts = pd.date_range(end=now, periods=2, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTC", "BTC"],
            "metric": ["active_addresses", "active_addresses"],
            "value": [123.0, 125.5],
            "timeframe": ["1d", "1d"],
        }
    )


def _mk_social_df(query="bitcoin", n=3):
    # n minute-spaced posts (NAIVE timestamps, the app will localize)
    now = pd.Timestamp.utcnow().floor("T")
    ts = pd.date_range(end=now, periods=n, freq="T")
    return pd.DataFrame(
        {
            "ts": ts,
            "user": [f"user{i}" for i in range(n)],
            "text": [f"{query} sample {i}" for i in range(n)],
            "sentiment_score": [0.1, 0.2, -0.1][:n],
            "timeframe": ["1m"] * n,
            # symbol/topic omitted to exercise default "twitter"
        }
    )


def test_onchain_ingest_then_retrieve(client, mem_store, monkeypatch):
    """
    POST /ingest/onchain/glassnode writes rows -> GET /ingest/features/onchain returns them.
    """
    # Monkeypatch the function imported in routes.py
    monkeypatch.setattr("app.ingestion_service.routes.fetch_glassnode", lambda *a, **k: _mk_onchain_df())

    # 1) Ingest
    r = client.post(
        "/ingest/onchain/glassnode",
        json={"symbol": "BTC", "metric": "active_addresses", "days": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["features_written"] >= 1

    # 2) One of the timestamps we wrote
    df = _mk_onchain_df()
    ts_epoch = int(df["timestamp"].iloc[-1].value // 1_000_000_000)

    # 3) Retrieve
    r2 = client.get(
        "/ingest/features/onchain",
        params={"symbol": "BTC", "metric": "active_addresses", "ts": [ts_epoch]},
    )
    assert r2.status_code == 200, r2.text
    out = r2.json()
    assert out["rows"] >= 1
    assert isinstance(out["data"][0]["timestamp"], int)
    # payload contains "metric" and "value"
    assert "value" in out["data"][0]


def test_social_ingest_then_retrieve(client, mem_store, monkeypatch):
    """
    POST /ingest/social/twitter writes rows -> GET /ingest/features/social returns them.
    """
    # Monkeypatch the function imported in routes.py
    monkeypatch.setattr(
        "app.ingestion_service.routes.fetch_twitter_sentiment",
        lambda query, since, until, max_results: _mk_social_df(query=query, n=3),
    )

    # 1) Ingest
    r = client.post(
        "/ingest/social/twitter",
        json={"query": "bitcoin", "max_results": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["features_written"] >= 1

    # 2) Pick a written timestamp
    df = _mk_social_df("bitcoin", 3)
    ts_epoch = int(df["ts"].iloc[-1].value // 1_000_000_000)

    # 3) Retrieve (topic defaults to 'twitter' in the writer; timeframe '1m')
    r2 = client.get(
        "/ingest/features/social",
        params={"topic": "twitter", "timeframe": "1m", "ts": [ts_epoch]},
    )
    assert r2.status_code == 200, r2.text
    out = r2.json()
    assert out["rows"] >= 1
    row = out["data"][0]
    assert isinstance(row["timestamp"], int)
    # writer sets payload keys: user, text, sentiment
    assert "user" in row and "text" in row and "sentiment" in row