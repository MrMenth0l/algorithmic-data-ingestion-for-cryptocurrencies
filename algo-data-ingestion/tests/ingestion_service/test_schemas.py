import pytest
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError

from app.ingestion_service.schemas import (
    MarketIngestRequest,
    OnchainIngestRequest,
    SocialIngestRequest,
    NewsIngestRequest,
)

def test_market_ingest_request_valid():
    req = MarketIngestRequest(symbol="BTC-USDT", granularity="1m")
    assert req.symbol == "BTC-USDT"
    assert req.granularity == "1m"

def test_market_ingest_request_missing_field():
    with pytest.raises(ValidationError):
        MarketIngestRequest(granularity="1m")  # missing symbol

def test_onchain_ingest_request_valid():
    req = OnchainIngestRequest(
        source="glassnode",
        chain_id=1,
        symbol="BTC",
        address=None,
        metric="balance",
        days=7
    )
    assert req.source == "glassnode"
    assert req.chain_id == 1
    assert req.days == 7

def test_onchain_ingest_request_invalid_days():
    with pytest.raises(ValidationError):
        OnchainIngestRequest(source="covalent", chain_id=1, days=0)

def test_social_ingest_request_valid():
    now = datetime.now(timezone.utc)
    req = SocialIngestRequest(
        platform="twitter",
        query="crypto",
        since=now - timedelta(hours=1),
        until=now,
        max_results=50
    )
    assert req.platform == "twitter"
    assert req.max_results == 50

def test_social_ingest_request_invalid_max_results():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        SocialIngestRequest(
            platform="reddit",
            query="blockchain",
            since=now - timedelta(hours=1),
            until=now,
            max_results=0
        )

def test_news_ingest_request_valid_api():
    req = NewsIngestRequest(source_type="api", category="crypto")
    assert req.source_type == "api"
    assert req.category == "crypto"
    assert req.feed_url is None

def test_news_ingest_request_valid_rss():
    req = NewsIngestRequest(source_type="rss", feed_url="http://example.com/rss")
    assert req.source_type == "rss"
    assert req.feed_url == "http://example.com/rss"
    assert req.category is None

def test_news_ingest_request_missing_source_type():
    with pytest.raises(ValidationError):
        NewsIngestRequest()