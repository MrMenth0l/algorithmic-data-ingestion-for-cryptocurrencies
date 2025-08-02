import os
import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

# We'll need httpx to patch
import httpx

# Adjust to your project layout
from app.adapters.reddit_adapter import (
    fetch_reddit_api,
    fetch_pushshift,
    fetch_reddit,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)

# Async context manager helper
class FakeClient:
    def __init__(self, method_fn, method_name="get", auth=None, headers=None):
        setattr(self, method_name, method_fn)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

# Combined client for auth and get in fetch_reddit_api
class FakeAuthClient:
    def __init__(self, post_fn, get_fn):
        self.post = post_fn
        self.get = get_fn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

# Fixture to seed env vars and reload module
@pytest.fixture(autouse=True)
def fix_env(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id-token")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret-token")
    monkeypatch.setenv("REDDIT_USER_AGENT", "TestAgent/0.1")
    # reload module so envs take effect
    import importlib
    import app.adapters.reddit_adapter as mod
    importlib.reload(mod)
    yield

@pytest.mark.asyncio
async def test_fetch_reddit_api_token_and_posts(monkeypatch):
    # 1) Mock token endpoint
    async def fake_post(url, data):
        assert url == "https://www.reddit.com/api/v1/access_token"
        assert data["grant_type"] == "client_credentials"
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"access_token": "fake-token"},
            raise_for_status=lambda: None
        )

    # 2) Mock posts endpoint
    now = datetime.utcnow()
    fake_posts = {
        "data": {
            "children": [
                {"data": {"id": "aaa", "title": "T1", "selftext": "body1", "author": "bob", "created_utc": int(now.timestamp())}},
                {"data": {"id": "bbb", "title": "T2", "selftext": "",   "author": "alice", "created_utc": int((now - timedelta(minutes=1)).timestamp())}},
            ]
        }
    }
    async def fake_get(url, params):
        assert url.startswith("https://oauth.reddit.com/r/testsub/new")
        assert params["limit"] == 2
        return SimpleNamespace(
            status_code=200,
            json=lambda: fake_posts,
            raise_for_status=lambda: None
        )

    # Patch the AsyncClient to use FakeAuthClient for both post and get
    monkeypatch.setattr(
        "app.adapters.reddit_adapter.httpx.AsyncClient",
        lambda *args, auth=None, headers=None, **kwargs: FakeAuthClient(fake_post, fake_get)
    )

    # Call fetch_reddit_api
    since = now - timedelta(hours=1)
    until = now
    df = await fetch_reddit_api("testsub", since, until, limit=2)

    # Validate DataFrame
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"ts", "author", "title", "selftext", "id"}
    assert df.shape[0] == 2
    # Check timestamp ordering and values
    # Ensure timestamps are positive epoch nanoseconds
    assert (df["ts"].astype('int64') > 0).all()

@pytest.mark.asyncio
async def test_fetch_pushshift(monkeypatch):
    # Prepare fake pushshift data
    now = datetime.utcnow()
    fake_data = [
        {"id": "xxx", "title": "P1", "selftext": "s1", "author": "u1", "created_utc": int(now.timestamp())},
        {"id": "yyy", "title": "P2", "selftext": "",   "author": "u2", "created_utc": int((now - timedelta(minutes=2)).timestamp())},
    ]
    async def fake_get(url, params):
        assert url == "https://api.pushshift.io/reddit/search/submission"
        assert params["subreddit"] == "othersub"
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"data": fake_data},
            raise_for_status=lambda: None
        )

    monkeypatch.setattr(
        "app.adapters.reddit_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get, method_name="get")
    )

    since = now - timedelta(hours=2)
    until = now
    df = await fetch_pushshift("othersub", since, until, limit=2)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"ts", "author", "title", "selftext", "id"}
    assert df.shape[0] == 2

@pytest.mark.asyncio
async def test_fetch_reddit_unified(monkeypatch):
    # Simply ensure fetch_reddit delegates correctly
    sentinel = pd.DataFrame({"x": [1]})
    monkeypatch.setattr("app.adapters.reddit_adapter.fetch_reddit_api", lambda *args, **kwargs: asyncio.sleep(0, result=sentinel))
    monkeypatch.setattr("app.adapters.reddit_adapter.fetch_pushshift", lambda *args, **kwargs: asyncio.sleep(0, result=sentinel))

    res_api = await fetch_reddit("any", datetime.utcnow(), datetime.utcnow(), source="api")
    res_ps  = await fetch_reddit("any", datetime.utcnow(), datetime.utcnow(), source="pushshift")
    assert res_api is sentinel
    assert res_ps  is sentinel

    with pytest.raises(ValueError):
        await fetch_reddit("any", datetime.utcnow(), datetime.utcnow(), source="bogus")