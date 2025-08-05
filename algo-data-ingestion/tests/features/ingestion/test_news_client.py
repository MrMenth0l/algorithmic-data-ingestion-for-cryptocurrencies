import pytest
from fastapi.testclient import TestClient
from ratelimit.exception import RateLimitException

from app.ingestion_service.main import app
from app.features.ingestion.news_client import NewsClient, RATE_LIMIT_CALLS

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limit():
    # Reset rate limit counter before each test
    NewsClient._rate_limit_counter = 0
    yield

def test_news_endpoint_success(monkeypatch):
    dummy = [{"ts": 1620000000, "title": "Test News", "url": "http://example.com", "summary": "Summary", "source": "crypto_news_api"}]
    monkeypatch.setattr(
        "app.features.ingestion.news_client.NewsClient.get_crypto_news",
        lambda self, since=None, until=None, source="crypto_news_api", limit=1: dummy
    )
    response = client.get("/ingest/news", params={"source": "crypto_news_api", "limit": 1})
    assert response.status_code == 200
    assert response.json() == dummy

def test_rate_limit_exceeded():
    news_client = NewsClient()
    # Make RATE_LIMIT_CALLS successful calls
    for _ in range(RATE_LIMIT_CALLS):
        news_client.get_crypto_news(None, None, "crypto_news_api", 1)
    # Next call should raise RateLimitException
    with pytest.raises(RateLimitException):
        news_client.get_crypto_news(None, None, "crypto_news_api", 1)