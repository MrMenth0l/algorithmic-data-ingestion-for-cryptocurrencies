import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.adapters.reddit_adapter import fetch_reddit_api, fetch_pushshift, REDDIT_API_CALLS, PUSHSHIFT_CALLS, REDDIT_PARSE_ERRORS

# Helpers
class DummyResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")
    def json(self):
        if self._json is None:
            raise Exception("Invalid JSON")
        return self._json

class FakeClient:
    def __init__(self, post_fn=None, get_fn=None):
        self.post = post_fn
        self.get = get_fn
    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc, tb): pass

@pytest.mark.asyncio
async def test_fetch_reddit_api_param_validation():
    now = datetime.utcnow()
    with pytest.raises(ValueError):
        await fetch_reddit_api("test", now, now - timedelta(seconds=1), 10)
    with pytest.raises(ValueError):
        await fetch_reddit_api("test", now - timedelta(days=1), now, 0)
    with pytest.raises(ValueError):
        await fetch_reddit_api("test", now - timedelta(days=1), now, 1001)

@pytest.mark.asyncio
async def test_fetch_reddit_api_retry_and_metric(monkeypatch):
    # Track metric
    calls = {"n": 0}
    now = datetime.utcnow()
    monkeypatch.setattr('app.adapters.reddit_adapter.REDDIT_API_CALLS', SimpleNamespace(inc=lambda: calls.__setitem__('n', calls['n'] + 1)))
    # Simulate token fetch failure then success
    tokens = [DummyResponse(200, json_data=None), DummyResponse(200, json_data={"access_token": "abc"})]
    async def fake_post(url, data):
        resp = tokens.pop(0)
        if resp._json is None:
            raise Exception("Token error")
        return resp
    # Simulate get posts
    posts_payload = {"data": {"children": [{"data": {"id": "1", "title": "t", "selftext": "s", "author": "a", "created_utc": int(now.timestamp())}}]}}
    async def fake_get(url, params):
        return DummyResponse(200, json_data=posts_payload)
    monkeypatch.setattr('app.adapters.reddit_adapter.httpx.AsyncClient', lambda *args, **kwargs: FakeClient(post_fn=fake_post, get_fn=fake_get))
    df = await fetch_reddit_api("test", now - timedelta(days=1), now, 1)
    assert isinstance(df, pd.DataFrame)
    assert calls['n'] == 1
    assert df.iloc[0]['id'] == '1'

@pytest.mark.asyncio
async def test_fetch_reddit_api_parse_error(monkeypatch):
    # Track parse errors
    errors = {"n": 0}
    monkeypatch.setattr('app.adapters.reddit_adapter.REDDIT_PARSE_ERRORS', SimpleNamespace(inc=lambda: errors.__setitem__('n', errors['n'] + 1)))
    now = datetime.utcnow()
    # Simulate token success but JSON error on posts
    async def fake_post(url, data): return DummyResponse(200, json_data={"access_token": "abc"})
    async def fake_get(url, params): return DummyResponse(200, json_data=None)
    monkeypatch.setattr('app.adapters.reddit_adapter.httpx.AsyncClient', lambda *args, **kwargs: FakeClient(post_fn=fake_post, get_fn=fake_get))
    df = await fetch_reddit_api("test", now - timedelta(days=1), now, 1)
    assert df.empty
    assert errors['n'] == 1

@pytest.mark.asyncio
async def test_fetch_pushshift_retry_and_metric(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr('app.adapters.reddit_adapter.PUSHSHIFT_CALLS', SimpleNamespace(inc=lambda: calls.__setitem__('n', calls['n'] + 1)))
    now = datetime.utcnow()
    posts = [[{"id": "2", "title": "t2", "selftext": "s2", "author": "b", "created_utc": int(now.timestamp())}]]
    responses = [Exception("Network fail"), DummyResponse(200, json_data={"data": posts[0]})]
    async def fake_get(url, params):
        resp = responses.pop(0)
        if isinstance(resp, Exception): raise resp
        return resp
    monkeypatch.setattr('app.adapters.reddit_adapter.httpx.AsyncClient', lambda *args, **kwargs: FakeClient(get_fn=fake_get))
    df = await fetch_pushshift("test", now - timedelta(days=1), now, 1)
    assert isinstance(df, pd.DataFrame)
    assert calls['n'] == 1

@pytest.mark.asyncio
async def test_fetch_pushshift_parse_error(monkeypatch):
    errors = {"n": 0}
    monkeypatch.setattr('app.adapters.reddit_adapter.REDDIT_PARSE_ERRORS', SimpleNamespace(inc=lambda: errors.__setitem__('n', errors['n'] + 1)))
    now = datetime.utcnow()
    async def fake_get(url, params): return DummyResponse(200, json_data={})
    monkeypatch.setattr('app.adapters.reddit_adapter.httpx.AsyncClient', lambda *args, **kwargs: FakeClient(get_fn=fake_get))
    df = await fetch_pushshift("test", now - timedelta(days=1), now, 1)
    assert df.empty
    assert errors['n'] == 1