import asyncio
from datetime import datetime
import pytest
from types import SimpleNamespace
from httpx import Response
import json

import feedparser

# FakeClient for patching httpx.AsyncClient
class FakeClient:
    def __init__(self, get_fn):
        self.get = get_fn

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# Adjust this import to your module path
from app.adapters.news_adapter import (
    fetch_crypto_news,
    fetch_rss_feed,
    NEWS_API_KEY,
)


class DummyResponse:
    def __init__(self, status_code: int, json_data=None, text_data=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def fix_env(monkeypatch):
    # ensure NEWS_API_KEY is set
    monkeypatch.setenv("NEWS_API_KEY", "test-token")
    import importlib
    import app.adapters.news_adapter as news_mod
    importlib.reload(news_mod)
    yield


@pytest.mark.asyncio
async def test_fetch_crypto_news_success(monkeypatch):
    # Prepare fake articles
    data = {
        "data": [
            {
                "news_url": "https://example.com/abcd1234",
                "title": "Test Title",
                "source_name": "ExampleSource",
                "author": "Alice",
                "text": "Description here",
                "date": "2025-08-01T12:00:00"
            }
        ]
    }

    async def fake_get(url, params):
        # check parameters passed correctly
        assert params["section"] == "general"
        assert params["items"] == 1
        assert params["token"] == "test-token"
        return DummyResponse(200, json_data=data)

    # patch httpx.AsyncClient.get
    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )

    results = await fetch_crypto_news("general", limit=1)
    assert len(results) == 1
    item = results[0]
    assert item["id"] == "abcd1234"  # last 10 chars of URL
    assert item["title"] == "Test Title"
    assert item["url"] == "https://example.com/abcd1234"
    assert item["source"] == "ExampleSource"
    assert item["author"] == "Alice"
    assert item["description"] == "Description here"
    assert isinstance(item["published_at"], datetime)
    assert item["published_at"] == datetime.fromisoformat("2025-08-01T12:00:00")


@pytest.mark.asyncio
async def test_fetch_crypto_news_http_error(monkeypatch):
    async def fake_get(url, params):
        return DummyResponse(500)

    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(fake_get)
    )

    with pytest.raises(Exception):
        await fetch_crypto_news("exchange", limit=5)


@pytest.mark.asyncio
async def test_fetch_rss_feed(monkeypatch):
    # Fake RSS content
    xml = "<rss><channel><item><guid>1</guid><title>News 1</title><link>url1</link><description>Sum1</description><pubDate>Wed, 01 Aug 2025 12:00:00 GMT</pubDate></item></channel></rss>"

    async def fake_get(url):
        return DummyResponse(200, text_data=xml)

    # patch httpx.AsyncClient.get
    monkeypatch.setattr(
        "app.adapters.news_adapter.httpx.AsyncClient",
        lambda *args, **kwargs: FakeClient(lambda url: fake_get(url))
    )

    # patch feedparser.parse
    monkeypatch.setattr(feedparser, "parse", lambda content: SimpleNamespace(entries=[
        {
            "id": "1",
            "title": "News 1",
            "link": "url1",
            "summary": "Sum1",
            "published": "Wed, 01 Aug 2025 12:00:00 GMT"
        }
    ]))

    updates = []
    # replace asyncio.sleep to break after first loop
    monkeypatch.setattr("app.adapters.news_adapter.asyncio.sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        await fetch_rss_feed("http://fake.feed/", lambda item: updates.append(item), poll_interval=1)

    assert len(updates) == 1
    item = updates[0]
    assert item["id"] == "1"
    assert item["title"] == "News 1"
    assert item["url"] == "url1"
    assert item["summary"] == "Sum1"
    assert item["published_at"] == "Wed, 01 Aug 2025 12:00:00 GMT"