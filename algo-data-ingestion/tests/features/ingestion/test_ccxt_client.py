import pytest
from fastapi.testclient import TestClient
from ratelimit import RateLimitException

from app.ingestion_service.main import app
from app.features.ingestion.ccxt_client import CCXTClient

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limit():
    # Ensure rate limit state is reset before each test
    CCXTClient._rate_limit_counter = 0
    yield

def test_historical_endpoint_success(monkeypatch):
    dummy = [{"timestamp": 1620000000, "price": 50000}]
    monkeypatch.setattr(
        "app.features.ingestion.ccxt_client.CCXTClient.fetch_historical",
        lambda self, symbol, since=None, limit=1: dummy
    )
    response = client.get("/ingest/ccxt/historical", params={"symbol": "BTC/USDT", "limit": 1})
    assert response.status_code == 200
    assert response.json() == dummy

def test_rate_limit_exceeded():
    ccxt = CCXTClient()
    # Make RATE_LIMIT_CALLS successful calls
    for _ in range(CCXTClient.RATE_LIMIT_CALLS):
        ccxt.fetch_historical("BTC/USDT", None, limit=1)
    # Next call should raise RateLimitException
    with pytest.raises(RateLimitException):
        ccxt.fetch_historical("BTC/USDT", None, limit=1)