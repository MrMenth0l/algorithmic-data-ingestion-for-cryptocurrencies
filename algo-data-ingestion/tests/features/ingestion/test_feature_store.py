import time
import pytest
from fastapi.testclient import TestClient
from app.ingestion_service.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_redis():
    # Flush Redis before each test to ensure a clean slate
    from app.ingestion_service.routes import _redis
    _redis.flushdb()
    yield

def test_feature_store_smoke():
    # 1) Write a vector
    payload = {
        "symbol": "BTC",
        "timestamp": 1620000000,
        "features": {"rsi": 55.2, "macd": 1.3}
    }
    write_resp = client.post("/ingest/features/write", json=payload)
    assert write_resp.status_code == 200
    data = write_resp.json()
    assert data["status"] == "ok"
    key = data["key"]
    assert key == "features:BTC:1620000000"

    # 2) Read it back immediately
    read_resp = client.get(
        f"/ingest/features/read?symbol=BTC&timestamp=1620000000"
    )
    assert read_resp.status_code == 200
    assert read_resp.json() == payload

def test_feature_store_ttl_eviction(monkeypatch):
    # Override TTL to 1 second for this test
    from app.ingestion_service.config import settings
    monkeypatch.setattr(settings, "feature_ttl_seconds", 1)

    payload = {
        "symbol": "ETH",
        "timestamp": 1630000000,
        "features": {"ema": 2.5}
    }
    client.post("/ingest/features/write", json=payload)

    # Immediately readable
    resp1 = client.get(
        "/ingest/features/read?symbol=ETH&timestamp=1630000000"
    )
    assert resp1.status_code == 200

    # Sleep past TTL
    time.sleep(1.1)

    # Now should be gone
    resp2 = client.get(
        "/ingest/features/read?symbol=ETH&timestamp=1630000000"
    )
    assert resp2.status_code == 404
    assert resp2.json()["detail"] == "Feature vector not found"