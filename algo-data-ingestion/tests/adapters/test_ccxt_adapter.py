# File: tests/adapters/test_ccxt_adapter.py
from app.adapters.ccxt_adapter import CCXTAdapter, get_ticker_raw

import pytest
import ccxt.async_support as ccxt

# sample ticker payload used by both tests
sample = {
    "symbol": "BTC/USDT",
    "bid": 50000,
    "ask": 50010,
    "last": 50005,
    "timestamp": 1620000000000
}

@pytest.mark.asyncio
async def test_fetch_ticker_monkeypatched(monkeypatch):
    class DummyExchange:
        async def fetch_ticker(self, symbol):
            assert symbol == "BTC/USDT"
            return sample
        async def close(self):
            pass

    # Monkeypatch ccxt.binance
    monkeypatch.setattr(
        ccxt, "binance",
        lambda *args, **kwargs: DummyExchange()
    )

    adapter = CCXTAdapter("binance")
    async def dummy_close(self):
        pass
    monkeypatch.setattr(CCXTAdapter, "close", dummy_close)
    result = await adapter.fetch_ticker("BTC/USDT")
    await adapter.close()

    assert result == sample

@pytest.mark.asyncio
async def test_get_ticker_raw(monkeypatch):
    async def dummy_fetch(self, symbol):
        return sample
    monkeypatch.setattr(CCXTAdapter, "fetch_ticker", dummy_fetch)
    async def dummy_close(self):
        pass
    monkeypatch.setattr(CCXTAdapter, "close", dummy_close)

    raw = await get_ticker_raw("BTC/USDT")
    assert raw == sample

# File: tests/adapters/test_reddit_adapter.py
import pytest
from datetime import datetime
from app.adapters.reddit_adapter import fetch_pushshift

@pytest.mark.asyncio
async def test_fetch_pushshift(monkeypatch):
    dummy_response = [
        {
            "id": "123",
            "title": "Hello",
            "selftext": "",
            "author": "u/test",
            "created_utc": 1620000000
        }
    ]

    class DummyClient:
        async def get(self, url, params):
            class R:
                def raise_for_status(self): pass
                def json(self): return {"data": dummy_response}
            return R()
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: DummyClient())

    posts = await fetch_pushshift("cryptocurrency", 1)
    assert len(posts) == 1
    post = posts[0]
    assert post["id"] == "123"
    assert isinstance(post["created_utc"], datetime)