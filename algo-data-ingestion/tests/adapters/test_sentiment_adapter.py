import os
import pytest
import pandas as pd
import asyncio
from types import SimpleNamespace
from datetime import datetime, timezone

# Patch 1: reload module and import as mod
import importlib
import app.adapters.sentiment_adapter as mod
importlib.reload(mod)

# Adjust to your project layout
from app.adapters.sentiment_adapter import fetch_twitter_sentiment, CLIENT

@pytest.fixture(autouse=True)
def fix_env_and_reload(monkeypatch):
    # Ensure the CLIENT was initialized with dummy env vars
    monkeypatch.setenv("TWITTER_BEARER", "bearer-token")
    monkeypatch.setenv("TWITTER_API_KEY", "api-key")
    monkeypatch.setenv("TWITTER_API_SECRET", "api-secret")
    # Reload the module so CLIENT picks them up
    import importlib
    import app.adapters.sentiment_adapter as mod
    importlib.reload(mod)
    yield

@pytest.mark.asyncio
async def test_fetch_twitter_sentiment_empty(monkeypatch):
    # Simulate no tweets returned
    fake_response = SimpleNamespace(data=[])
    monkeypatch.setattr(mod.CLIENT, "search_recent_tweets", lambda *args, **kwargs: fake_response)

    since = datetime(2025, 8, 1, tzinfo=timezone.utc)
    until = datetime(2025, 8, 1,  tzinfo=timezone.utc)
    df = await mod.fetch_twitter_sentiment("test", since, until, max_results=5)

    # Expect an empty DataFrame with the right columns
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["ts", "user", "text", "sentiment_score"]
    assert df.shape[0] == 0

@pytest.mark.asyncio
async def test_fetch_twitter_sentiment_populated(monkeypatch):
    # Build fake tweets
    now = datetime(2025, 8, 1, 12, 0, tzinfo=timezone.utc)
    tweets = [
        SimpleNamespace(
            id="1", text="Hello world", author_id=123, created_at=now
        ),
        SimpleNamespace(
            id="2", text="Another tweet", author_id=456, created_at=now
        ),
    ]
    fake_response = SimpleNamespace(data=tweets)
    # Patch the search_recent_tweets method
    monkeypatch.setattr(mod.CLIENT, "search_recent_tweets", lambda *args, **kwargs: fake_response)

    since = datetime(2025, 8, 1, 11, 0, tzinfo=timezone.utc)
    until = datetime(2025, 8, 1, 12, 0, tzinfo=timezone.utc)
    df = await mod.fetch_twitter_sentiment("foo", since, until, max_results=2)

    # Validate DataFrame
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["ts", "user", "text", "sentiment_score"]
    assert df.shape[0] == 2

    # Check contents
    assert df["text"].tolist() == ["Hello world", "Another tweet"]
    assert df["user"].tolist() == ["123", "456"]
    # Timestamps should match and be timezone-aware UTC
    assert all(ts.tzinfo == timezone.utc for ts in df["ts"])
    assert df["ts"].tolist() == [now, now]
    # The placeholder sentiment score remains 0.0
    assert df["sentiment_score"].tolist() == [0.0, 0.0]