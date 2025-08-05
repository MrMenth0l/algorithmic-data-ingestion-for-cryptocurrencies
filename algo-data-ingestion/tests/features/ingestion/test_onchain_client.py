import pytest
from fastapi.testclient import TestClient
from ratelimit.exception import RateLimitException

from app.ingestion_service.main import app
from app.features.ingestion.onchain_client import OnchainClient, RATE_LIMIT_CALLS

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limit():
    # Reset rate limit counters before each test
    OnchainClient._rate_limit_counter = 0
    yield

def test_glassnode_endpoint_success(monkeypatch):
    dummy = [{"timestamp": 1620000000, "value": 123.45}]
    monkeypatch.setattr(
        "app.features.ingestion.onchain_client.OnchainClient.get_glassnode_metric",
        lambda self, symbol, metric, days=1: dummy
    )
    response = client.get("/ingest/onchain/glassnode", params={"symbol": "BTC", "metric": "volume", "days": 1})
    assert response.status_code == 200
    assert response.json() == dummy

def test_covalent_endpoint_success(monkeypatch):
    dummy = [{"token": "0xABC", "balance": 1000}]
    monkeypatch.setattr(
        "app.features.ingestion.onchain_client.OnchainClient.get_covalent_balances",
        lambda self, chain_id, address: dummy
    )
    response = client.get("/ingest/onchain/covalent", params={"chain_id": 1, "address": "0xAddress"})
    assert response.status_code == 200
    assert response.json() == dummy

def test_rate_limit_exceeded():
    client_obj = OnchainClient()
    # Exceed rate limit for glassnode_metric
    for _ in range(RATE_LIMIT_CALLS):
        client_obj.get_glassnode_metric("BTC", "volume", days=1)
    with pytest.raises(RateLimitException):
        client_obj.get_glassnode_metric("BTC", "volume", days=1)

    # Reset and exceed rate limit for covalent_balances
    OnchainClient._rate_limit_counter = 0
    for _ in range(RATE_LIMIT_CALLS):
        client_obj.get_covalent_balances(chain_id=1, address="0xAddress")
    with pytest.raises(RateLimitException):
        client_obj.get_covalent_balances(chain_id=1, address="0xAddress")