import pytest
from fastapi.testclient import TestClient
from ratelimit.exception import RateLimitException

from app.ingestion_service.main import app
from app.features.ingestion.social_client import SocialClient, RATE_LIMIT_CALLS

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limit():
    # Reset rate limit counter before each test
    SocialClient._rate_limit_counter = 0
    yield

def test_twitter_endpoint_success(monkeypatch):
    dummy = [{"ts": 1620000000, "author": "user", "content": "tweet content"}]
    monkeypatch.setattr(
        "app.features.ingestion.social_client.SocialClient.fetch_tweets",
        lambda self, query, since=None, until=None, limit=100: dummy
    )
    response = client.get("/ingest/social/twitter", params={"query": "test", "limit": 1})
    assert response.status_code == 200
    assert response.json() == dummy

def test_reddit_endpoint_success(monkeypatch):
    dummy = [{"ts": 1620000000, "author": "user", "content": "post content"}]
    monkeypatch.setattr(
        "app.features.ingestion.social_client.SocialClient.fetch_reddit_api",
        lambda self, subreddit, since=None, until=None, limit=100: dummy
    )
    response = client.get("/ingest/social/reddit", params={"subreddit": "testsub", "limit": 1})
    assert response.status_code == 200
    assert response.json() == dummy

def test_rate_limit_exceeded_twitter():
    sc = SocialClient()
    for _ in range(RATE_LIMIT_CALLS):
        sc.fetch_tweets("test", None, None, limit=1)
    with pytest.raises(RateLimitException):
        sc.fetch_tweets("test", None, None, limit=1)

def test_rate_limit_exceeded_reddit():
    sc = SocialClient()
    # reset counter
    SocialClient._rate_limit_counter = 0
    for _ in range(RATE_LIMIT_CALLS):
        sc.fetch_reddit("testsub", None, None, limit=1)
    with pytest.raises(RateLimitException):
        sc.fetch_reddit("testsub", None, None, limit=1)
