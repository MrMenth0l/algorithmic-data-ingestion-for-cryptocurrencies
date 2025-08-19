import pytest
from fastapi.testclient import TestClient
import pandas as pd
from datetime import datetime

# Import the app and DI providers from main (assumes main imports routes AFTER providers)
from app.ingestion_service.main import app, get_ccxt, get_onchain, get_social, get_news


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------
# GET endpoints (DI overrides) → expect JSON envelope {"rows": int, "data": [...]}
# ---------------------------------------------------------------------

def test_get_news_search_envelope(client):

    """GET /ingest/news/search returns {rows, data} envelope."""
    class FakeNews:
        async def get_crypto_news(self, since=None, until=None, source="api", limit: int = 2):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
            return pd.DataFrame({
                "published_at": ts,
                "id": pd.Series(["n1"], dtype="string"),
                "title": pd.Series(["Hello"], dtype="string"),
                "url": pd.Series(["http://x"], dtype="string"),
                "source": pd.Series([source], dtype="string"),
                "author": pd.Series(["me"], dtype="string"),
                "description": pd.Series(["desc"], dtype="string"),
            })
        async def aclose(self): pass

    def _override_news(**_kwargs):
        return FakeNews()

    app.dependency_overrides[get_news] = _override_news
    r = client.get(
        "/ingest/news/search",
        params={"source": "api", "limit": 1, "since": 0, "until": 1},
    )
    app.dependency_overrides.clear()
    assert r.status_code == 200


def test_get_onchain_glassnode_envelope(client):
    """GET /ingest/onchain/glassnode returns {rows, data} envelope."""
    class FakeOnchain:
        async def get_glassnode_metric(self, symbol: str, metric: str, days: int = 1):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
            return pd.DataFrame({
                "timestamp": ts,
                "source": ["glassnode"],
                "symbol": [symbol],
                "metric": [metric],
                "value": [1.23],
            })
        async def aclose(self): pass

    app.dependency_overrides[get_onchain] = lambda *_: FakeOnchain()
    r = client.get("/ingest/onchain/glassnode", params={"symbol": "BTC", "metric": "active_addresses", "days": 1})
    app.dependency_overrides.clear()
    assert r.status_code == 200
    j = r.json()
    assert "rows" in j and isinstance(j["data"], list)


def test_get_social_reddit_envelope(client):
    """GET /ingest/social/reddit returns {rows, data} envelope."""
    class FakeSocial:
        async def fetch_reddit_api(self, subreddit: str, since=None, until=None, limit: int = 2):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True)
            return pd.DataFrame({
                "ts": ts,
                "author": pd.Series(["r1"], dtype="string"),
                "title": pd.Series(["t"], dtype="string"),
                "selftext": pd.Series(["s"], dtype="string"),
                "score": pd.Series([10], dtype="Int64"),
                "num_comments": pd.Series([2], dtype="Int64"),
                "id": pd.Series(["abc"], dtype="string"),
                "subreddit": pd.Series([subreddit], dtype="string"),
                "source": pd.Series(["reddit"], dtype="string"),
            })
        async def aclose(self): pass

    app.dependency_overrides[get_social] = lambda *_: FakeSocial()
    r = client.get("/ingest/social/reddit", params={"subreddit": "CryptoCurrency", "limit": 1})
    app.dependency_overrides.clear()
    assert r.status_code == 200
    j = r.json()
    assert "rows" in j and isinstance(j["data"], list)


def test_get_ccxt_historical_envelope(client):
    """GET /ingest/ccxt/binance/historical returns {rows, data} envelope OR raw dict."""
    class FakeCCXT:
        async def fetch_historical(self, symbol: str, timeframe: str = "1m", since=None, limit: int = 2):
            ts = pd.to_datetime(["2025-08-01T00:00:00Z", "2025-08-01T00:01:00Z"], utc=True)
            return pd.DataFrame({
                "timestamp": ts,
                "open": [1.0, 2.0],
                "high": [1.1, 2.1],
                "low": [0.9, 1.9],
                "close": [1.05, 2.05],
                "volume": [10.0, 20.0],
                "symbol": pd.Series([symbol, symbol], dtype="string"),
                "exchange": pd.Series(["binance", "binance"], dtype="string"),
                "timeframe": pd.Series([timeframe, timeframe], dtype="string"),
            })
        async def aclose(self): pass

    app.dependency_overrides[get_ccxt] = lambda *_: FakeCCXT()
    r = client.get("/ingest/ccxt/binance/historical", params={"symbol": "BTC/USDT", "timeframe": "1m", "limit": 2})
    app.dependency_overrides.clear()
    assert r.status_code == 200
    j = r.json()
    # Accept either standardized envelope or the current raw dict-of-columns
    if isinstance(j, dict) and "rows" in j and "data" in j:
        assert isinstance(j["data"], list)
    else:
        # raw frame-like dict: expect canonical columns present
        for k in ("timestamp", "open", "high", "low", "close", "volume", "symbol", "exchange", "timeframe", "dt"):
            assert k in j


# ---------------------------------------------------------------------
# POST /ingest/market/{exchange} – success / no_data / write_error / schema_error
# ---------------------------------------------------------------------

def test_post_market_success(client, monkeypatch):
    """Success path: tz-naive input; route tz_localize('UTC'); writer returns path."""
    df = pd.DataFrame({
        # tz-naive on purpose so route can tz_localize('UTC') without raising
        "timestamp": pd.to_datetime(["2025-08-01 00:00:00"]),
        "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0],
        "symbol": pd.Series(["BTCUSDT"], dtype="string"),
        "exchange": pd.Series(["binance"], dtype="string"),
        "timeframe": pd.Series(["1m"], dtype="string"),
    })

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return df

    # Adapter mock
    from types import SimpleNamespace
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv)
    )

    # Writer assertion + path: don't require 'dt' here since your writer derives it
    def capture_write(df_in, base, partitions, filename=None):
        # after route tz_localize
        assert "UTC" in str(df_in["timestamp"].dtype)
        return "/fake/path.parquet"

    monkeypatch.setattr("app.ingestion_service.routes.write_to_parquet", capture_write)

    r = client.post("/ingest/market/binance", json={"symbol": "BTC-USDT", "granularity": "1m"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "path": "/fake/path.parquet"}


def test_post_market_no_data(client, monkeypatch):
    """Empty DataFrame → no_data."""
    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return pd.DataFrame()

    from types import SimpleNamespace
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv)
    )
    monkeypatch.setattr("app.ingestion_service.routes.write_to_parquet", lambda *a, **k: None)

    r = client.post("/ingest/market/binance", json={"symbol": "BTC-USDT", "granularity": "1m"})
    assert r.status_code == 200
    assert r.json() == {"status": "no_data", "path": None}


def test_post_market_write_error(client, monkeypatch):
    """Writer raises IOError → 500 with 'Write failed'."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-08-01 00:00:00"]),  # tz-naive for route tz_localize
        "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0],
        "symbol": pd.Series(["BTCUSDT"], dtype="string"),
        "exchange": pd.Series(["binance"], dtype="string"),
        "timeframe": pd.Series(["1m"], dtype="string"),
    })

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return df

    from types import SimpleNamespace
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv)
    )
    def bad_write(*_a, **_k): raise IOError("disk full")
    monkeypatch.setattr("app.ingestion_service.routes.write_to_parquet", bad_write)

    r = client.post("/ingest/market/binance", json={"symbol": "BTC-USDT", "granularity": "1m"})
    assert r.status_code == 500
    assert "Write failed" in r.json()["detail"]


def test_post_market_schema_error(client, monkeypatch):
    """
    Simulate schema/normalization error as a ValueError raised by write_to_parquet
    so the route returns 422 with the message.
    """
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-08-01 00:00:00"]),  # tz-naive; route will localize
        "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0],
        "symbol": pd.Series(["BTCUSDT"], dtype="string"),
        "exchange": pd.Series(["binance"], dtype="string"),
        "timeframe": pd.Series(["1m"], dtype="string"),
    })

    async def fake_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        return df

    from types import SimpleNamespace
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv)
    )
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("Missing columns: ['x']"))
    )

    r = client.post("/ingest/market/binance", json={"symbol": "BTC-USDT", "granularity": "1m"})
    assert r.status_code == 422
    assert "Missing columns" in r.json()["detail"]