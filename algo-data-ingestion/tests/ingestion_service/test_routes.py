import asyncio
import pytest
from fastapi.testclient import TestClient
from app.ingestion_service.main import app
from app.ingestion_service import utils
import pandas as pd
from types import SimpleNamespace

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_ingest_market_success(client, monkeypatch):
    # Mock adapter
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-08-01T00:00:00Z"]),
        "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]
    })
    async def fake_fetch_ohlcv(symbol, gr):
        return df
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv)
    )
    # Mock write_to_parquet
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: "/fake/path.parquet"
    )
    response = client.post("/ingest/market/binance", json={"symbol":"BTC-USDT","granularity":"1m"})
    assert response.status_code == 200
    assert response.json() == {"status":"ok","path":"/fake/path.parquet"}

def test_ingest_market_no_data(client, monkeypatch):
    # Mock empty df
    df = pd.DataFrame()
    async def fake_fetch_ohlcv(symbol, gr):
        return df
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv)
    )
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: None
    )
    response = client.post("/ingest/market/binance", json={"symbol":"BTC-USDT","granularity":"1m"})
    assert response.status_code == 200
    assert response.json() == {"status":"no_data","path":None}

def test_ingest_market_write_error(client, monkeypatch):
    df = pd.DataFrame({"a":[1]})
    async def fake_fetch_ohlcv(symbol, gr):
        return df
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch_ohlcv)
    )
    def bad_write(df, base, partitions):
        raise IOError("disk full")
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        bad_write
    )
    response = client.post("/ingest/market/binance", json={"symbol":"BTC-USDT","granularity":"1m"})
    assert response.status_code == 500
    assert "Write failed" in response.json()["detail"]

def test_ingest_onchain_invalid_source(client):
    response = client.post("/ingest/onchain/unknown", json={"source":"unknown","chain_id":1,"days":1})
    assert response.status_code == 400

def test_ingest_social_invalid_platform(client):
    response = client.post("/ingest/social/unknown", json={"platform":"unknown","query":"a","since":"2025-08-01T00:00:00Z","until":"2025-08-02T00:00:00Z","max_results":1})
    assert response.status_code == 400

def test_ingest_news_invalid_source(client):
    response = client.post("/ingest/news", json={"source_type":"bad"})
    assert response.status_code == 400


# On-Chain Success
def test_ingest_onchain_glassnode_success(client, monkeypatch):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-08-01T00:00:00Z"]),
        "source": ["glassnode"],
        "symbol": ["BTC"],
        "metric": ["balance"],
        "value": [123.45]
    })
    # Mock fetch_glassnode
    async def fake_fetch_glassnode(symbol, metric, days):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_glassnode", fake_fetch_glassnode)
    # Mock write_to_parquet
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: "/fake/onchain.parquet"
    )
    response = client.post("/ingest/onchain/glassnode", json={
        "source": "glassnode", "chain_id": 1, "symbol": "BTC", "metric": "balance", "days": 1
    })
    assert response.status_code == 200
    assert response.json() == {"status":"ok","path":"/fake/onchain.parquet"}

def test_ingest_onchain_glassnode_no_data(client, monkeypatch):
    df = pd.DataFrame()
    async def fake_fetch_glassnode(symbol, metric, days):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_glassnode", fake_fetch_glassnode)
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: None
    )
    response = client.post("/ingest/onchain/glassnode", json={
        "source": "glassnode", "chain_id": 1, "symbol": "BTC", "metric": "balance", "days": 1
    })
    assert response.status_code == 200
    assert response.json() == {"status":"no_data","path":None}

def test_ingest_onchain_glassnode_write_error(client, monkeypatch):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-08-01T00:00:00Z"]),
        "source": ["glassnode"],
        "symbol": ["BTC"],
        "metric": ["balance"],
        "value": [123.45]
    })
    async def fake_fetch_glassnode(symbol, metric, days):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_glassnode", fake_fetch_glassnode)
    def bad_write(df, base, partitions):
        raise IOError("disk full")
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        bad_write
    )
    response = client.post("/ingest/onchain/glassnode", json={
        "source": "glassnode", "chain_id": 1, "symbol": "BTC", "metric": "balance", "days": 1
    })
    assert response.status_code == 500
    assert "Write failed" in response.json()["detail"]


def test_ingest_social_twitter_success(client, monkeypatch):
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2025-08-01T00:00:00Z","2025-08-01T01:00:00Z"]),
        "user": ["u1","u2"],
        "text": ["t1","t2"],
        "sentiment_score": [0.1, -0.2]
    })
    async def fake_twitter(query, since, until, max_results):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_twitter_sentiment", fake_twitter)
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: "/fake/social.parquet"
    )
    response = client.post("/ingest/social/twitter", json={
        "platform": "twitter", "query": "test", "since": "2025-08-01T00:00:00Z",
        "until": "2025-08-01T01:00:00Z", "max_results": 2
    })
    assert response.status_code == 200
    assert response.json() == {"status":"ok","path":"/fake/social.parquet"}

def test_ingest_social_twitter_no_data(client, monkeypatch):
    df = pd.DataFrame()
    async def fake_twitter(query, since, until, max_results):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_twitter_sentiment", fake_twitter)
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: None
    )
    response = client.post("/ingest/social/twitter", json={
        "platform": "twitter", "query": "test", "since": "2025-08-01T00:00:00Z",
        "until": "2025-08-01T01:00:00Z", "max_results": 2
    })
    assert response.status_code == 200
    assert response.json() == {"status":"no_data","path":None}

def test_ingest_social_twitter_write_error(client, monkeypatch):
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2025-08-01T00:00:00Z"]),
        "user": ["u1"],
        "text": ["t1"],
        "sentiment_score": [0.1]
    })
    async def fake_twitter(query, since, until, max_results):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_twitter_sentiment", fake_twitter)
    def bad_write(df, base, partitions):
        raise IOError("disk full")
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        bad_write
    )
    response = client.post("/ingest/social/twitter", json={
        "platform": "twitter", "query": "test", "since": "2025-08-01T00:00:00Z",
        "until": "2025-08-01T01:00:00Z", "max_results": 2
    })
    assert response.status_code == 500
    assert "Write failed" in response.json()["detail"]


def test_ingest_news_api_success(client, monkeypatch):
    df = pd.DataFrame({
        "published_at": pd.to_datetime(["2025-08-01T00:00:00Z"]),
        "source_type": ["api"],
        "source_name": ["crypto.news"],
        "title": ["Hello"],
        "url": ["http://"],
        "author": ["a"],
        "summary": ["s"]
    })
    async def fake_fetch_news_api(category):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_news_api", fake_fetch_news_api)
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: "/fake/news.parquet"
    )
    response = client.post("/ingest/news", json={"source_type":"api","category":"crypto"})
    assert response.status_code == 200
    assert response.json() == {"status":"ok","path":"/fake/news.parquet"}

def test_ingest_news_api_no_data(client, monkeypatch):
    df = pd.DataFrame()
    async def fake_fetch_news_api(category):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_news_api", fake_fetch_news_api)
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        lambda df, base, partitions: None
    )
    response = client.post("/ingest/news", json={"source_type":"api","category":"crypto"})
    assert response.status_code == 200
    assert response.json() == {"status":"no_data","path":None}

def test_ingest_news_api_write_error(client, monkeypatch):
    df = pd.DataFrame({
        "published_at": pd.to_datetime(["2025-08-01T00:00:00Z"]),
        "source_type": ["api"],
        "source_name": ["crypto.news"],
        "title": ["Hello"],
        "url": ["http://"],
        "author": ["a"],
        "summary": ["s"]
    })
    async def fake_fetch_news_api(category):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_news_api", fake_fetch_news_api)
    def bad_write(df, base, partitions):
        raise IOError("disk full")
    monkeypatch.setattr(
        "app.ingestion_service.routes.write_to_parquet",
        bad_write
    )
    response = client.post("/ingest/news", json={"source_type":"api","category":"crypto"})
    assert response.status_code == 500
    assert "Write failed" in response.json()["detail"]


# Schema validation error for Market
def test_ingest_market_schema_validation_error(client, monkeypatch):
    # Missing 'timestamp'
    df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]})
    async def fake_fetch(symbol, gr):
        return df
    monkeypatch.setattr(
        "app.ingestion_service.routes.CCXTAdapter",
        lambda exchange: SimpleNamespace(fetch_ohlcv=fake_fetch)
    )
    # write_to_parquet shouldn't be reached; route should return 422
    response = client.post(
        "/ingest/market/binance",
        json={"symbol": "BTC-USDT", "granularity": "1m"}
    )
    assert response.status_code == 422
    assert "Missing columns" in response.json()["detail"]

# Schema validation error for On-Chain
def test_ingest_onchain_schema_validation_error(client, monkeypatch):
    # Missing 'timestamp'
    df = pd.DataFrame({"source": ["glassnode"], "symbol": ["BTC"], "metric": ["balance"], "value": [1.0]})
    async def fake_fetch_glassnode(symbol, metric, days):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_glassnode", fake_fetch_glassnode)
    response = client.post(
        "/ingest/onchain/glassnode",
        json={"source": "glassnode", "chain_id": 1, "symbol": "BTC", "metric": "balance", "days": 1}
    )
    assert response.status_code == 422
    assert "Missing columns" in response.json()["detail"]

# Schema validation error for Social
def test_ingest_social_schema_validation_error(client, monkeypatch):
    # Missing 'ts'
    df = pd.DataFrame({"user": ["u"], "text": ["t"], "sentiment_score": [0.0]})
    async def fake_twitter(query, since, until, max_results):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_twitter_sentiment", fake_twitter)
    response = client.post(
        "/ingest/social/twitter",
        json={
            "platform": "twitter",
            "query": "test",
            "since": "2025-08-01T00:00:00Z",
            "until": "2025-08-01T01:00:00Z",
            "max_results": 1
        }
    )
    assert response.status_code == 422
    assert "Missing columns" in response.json()["detail"]

# Schema validation error for News
def test_ingest_news_schema_validation_error(client, monkeypatch):
    # Missing 'published_at'
    df = pd.DataFrame({"source_type": ["api"], "source_name": ["n"], "title": ["t"], "url": ["u"], "author": ["a"], "summary": ["s"]})
    async def fake_fetch_news_api(category):
        return df
    monkeypatch.setattr("app.ingestion_service.routes.fetch_news_api", fake_fetch_news_api)
    response = client.post("/ingest/news", json={"source_type": "api", "category": "crypto"})
    assert response.status_code == 422
    assert "Missing columns" in response.json()["detail"]
