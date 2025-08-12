import os

import analyzer
import pytest
import pandas as pd
import asyncio
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta


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

    # Expect an empty DataFrame with normalized columns
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected = {"ts", "author", "text", "likes", "retweets", "sentiment_score", "id", "source", "dt"}
    assert expected.issubset(set(df.columns))
    assert str(df["ts"].dtype).endswith("UTC]")

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
    # Stub out sentiment analyzer for predictable scores
    monkeypatch.setattr(mod, "sentiment_analyzer", lambda text: [{"score": 0.0}])

    since = datetime(2025, 8, 1, 11, 0, tzinfo=timezone.utc)
    until = datetime(2025, 8, 1, 12, 0, tzinfo=timezone.utc)
    df = await mod.fetch_twitter_sentiment("foo", since, until, max_results=2)

    # Validate DataFrame
    assert isinstance(df, pd.DataFrame)
    expected = {"ts", "author", "text", "likes", "retweets", "sentiment_score", "id", "source", "dt"}
    assert expected.issubset(set(df.columns))
    assert df.shape[0] == 2

    # Check contents
    assert df["text"].tolist() == ["Hello world", "Another tweet"]
    assert df["author"].tolist() == ["123", "456"]
    # Timestamps should match and be timezone-aware UTC
    assert all(getattr(ts, 'tzinfo', None) is not None for ts in df["ts"])  # tz-aware
    assert df["ts"].tolist() == [now, now]
    # The placeholder sentiment score remains 0.0
    assert df["sentiment_score"].tolist() == [0.0, 0.0]
    # Normalization extras
    assert df["source"].nunique() == 1 and df["source"].iloc[0] == "twitter"
    assert df["dt"].nunique() == 1


# Additional tests for parameter validation, retry logic, and parse error handling
import tweepy

@pytest.mark.asyncio
async def test_parameter_validation(monkeypatch):
    now = datetime(2025, 8, 1, tzinfo=timezone.utc)
    # Out-of-range max_results should return an empty, schema-correct DF (no exception)
    df = await mod.fetch_twitter_sentiment("q", now - timedelta(hours=1), now, max_results=0)
    assert isinstance(df, pd.DataFrame) and df.empty
    df = await mod.fetch_twitter_sentiment("q", now - timedelta(hours=1), now, max_results=101)
    assert isinstance(df, pd.DataFrame) and df.empty


@pytest.mark.asyncio
async def test_retry_logic_and_calls(monkeypatch):
    # Setup: first call raises TweepyException, second returns data
    calls = {"n": 0}
    monkeypatch.setattr('app.adapters.sentiment_adapter.TWITTER_CALLS', SimpleNamespace(inc=lambda: calls.__setitem__('n', calls['n'] + 1)))
    tweets = [SimpleNamespace(id="1", text="Hi", author_id=1, created_at=datetime(2025,8,1,12,0,tzinfo=timezone.utc))]
    fake_responses = [tweepy.TweepyException("Rate limit"), SimpleNamespace(data=tweets)]
    async def fake_search(*args, **kwargs):
        resp = fake_responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp
    monkeypatch.setattr(mod.CLIENT, "search_recent_tweets", fake_search)
    df = await mod.fetch_twitter_sentiment("test", datetime(2025,8,1,11,0,tzinfo=timezone.utc), datetime(2025,8,1,12,0,tzinfo=timezone.utc), max_results=1)
    assert isinstance(df, pd.DataFrame)
    # Should increment calls once on initial attempt
    assert calls["n"] == 1
    assert df.shape[0] == 1


@pytest.mark.asyncio
async def test_sentiment_parse_error(monkeypatch):
    errors = {"n": 0}
    monkeypatch.setattr('app.adapters.sentiment_adapter.TWITTER_PARSE_ERRORS', SimpleNamespace(inc=lambda: errors.__setitem__('n', errors['n'] + 1)))
    # Simulate pipeline error
    monkeypatch.setattr('app.adapters.sentiment_adapter.sentiment_analyzer', lambda text: (_ for _ in ()).throw(Exception("model fail")))
    # Simulate one tweet
    async def fake_search(*args, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(id="1", text="Bad", author_id=1, created_at=datetime(2025,8,1,12,0,tzinfo=timezone.utc))])
    monkeypatch.setattr(mod.CLIENT, "search_recent_tweets", fake_search)
    df = await mod.fetch_twitter_sentiment("test", datetime(2025,8,1,11,0,tzinfo=timezone.utc), datetime(2025,8,1,12,0,tzinfo=timezone.utc), max_results=1)
    # parse error should increment once for model failure
    assert errors["n"] == 1
    assert df.iloc[0]["sentiment_score"] == 0.0
    expected = {"ts", "author", "text", "likes", "retweets", "sentiment_score", "id", "source", "dt"}
    assert expected.issubset(set(df.columns))
    assert str(df["ts"].dtype).endswith("UTC]")