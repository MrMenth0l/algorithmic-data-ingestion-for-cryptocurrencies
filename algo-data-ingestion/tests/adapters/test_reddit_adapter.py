import pytest
from datetime import datetime
import httpx
from app.adapters.reddit_adapter import fetch_reddit_api, fetch_pushshift

@pytest.mark.asyncio
async def test_fetch_reddit_api(monkeypatch):
    """
    Test fetch_reddit_api using a dummy OAuth flow and dummy posts.
    """
    # Dummy token response
    class DummyTokenResponse:
        def raise_for_status(self): pass
        def json(self): return {"access_token": "fake-token"}

    # Dummy posts response
    dummy_children = [
        {"data": {
            "id": "abc123",
            "title": "Test",
            "selftext": "Body",
            "author": "user1",
            "created_utc": 1620000000
        }}
    ]

    class DummyOAuthClient:
        async def post(self, url, data):
            # Ensure grant_type present
            assert data.get("grant_type") == "client_credentials"
            return DummyTokenResponse()
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    class DummyAPIClient:
        async def get(self, url, params):
            # Ensure limit parameter is passed
            assert "limit" in params
            class R:
                def raise_for_status(self): pass
                def json(self): return {"data": {"children": dummy_children}}
            return R()
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    # Monkeypatch both HTTP clients
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *args, auth=None, headers=None, **kwargs: DummyOAuthClient() if auth else DummyAPIClient()
    )

    posts = await fetch_reddit_api("testsub", 2)
    assert isinstance(posts, list)
    assert len(posts) == 1
    post = posts[0]
    assert post["id"] == "abc123"
    assert post["title"] == "Test"
    assert isinstance(post["created_utc"], datetime)

@pytest.mark.asyncio
async def test_fetch_pushshift(monkeypatch):
    """
    Test fetch_pushshift using a dummy Pushshift response.
    """
    dummy_data = [
        {
            "id": "xyz789",
            "title": "PushTest",
            "selftext": "BodyPS",
            "author": "user2",
            "created_utc": 1620000000
        }
    ]

    class DummyClient:
        async def get(self, url, params):
            class R:
                def raise_for_status(self): pass
                def json(self): return {"data": dummy_data}
            return R()
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient())

    posts = await fetch_pushshift("testsub", 3)
    assert isinstance(posts, list)
    assert len(posts) == 1
    post = posts[0]
    assert post["id"] == "xyz789"
    assert isinstance(post["created_utc"], datetime)