import asyncio
import pytest
import pandas as pd

from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.adapters.ccxt_adapter import CCXTAdapter, get_ticker_raw

@pytest.fixture
def dummy_exchange(monkeypatch):
    fake = AsyncMock()
    fake.fetch_ticker.return_value = {"symbol": "BTC/USDT", "last": 123.45}
    fake.fetch_ohlcv.return_value = [
        [1609459200000, 29000, 29500, 28800, 29300, 100],
        [1609545600000, 29300, 30000, 29000, 29500, 150],
    ]
    fake.fetch_order_book.return_value = {
        "bids": [[29500, 1], [29400, 2]],
        "asks": [[29600, 0.5], [29700, 0.8]],
    }
    fake.load_markets.return_value = {"BTC/USDT": {}}
    fake.watch_ticker.side_effect = [
        {"symbol": "BTC/USDT", "last": 29300},
        asyncio.CancelledError()  # to break the loop after first callback
    ]
    fake.fetch_balance.return_value = {"free": {"USDT": 1000}, "used": {}}
    fake.load_markets.return_value = {"BTC/USDT": {}}
    # patch ccxt.binance to return instance of our fake
    monkeypatch.setattr("app.adapters.ccxt_adapter.ccxt.binance", lambda *args, **kwargs: fake)
    return fake

@pytest.mark.asyncio
async def test_fetch_ticker(dummy_exchange):
    adapter = CCXTAdapter("binance")
    ticker = await adapter.fetch_ticker("BTC/USDT")
    assert ticker["last"] == 123.45

@pytest.mark.asyncio
async def test_fetch_ohlcv(dummy_exchange):
    adapter = CCXTAdapter("binance")
    since = datetime(2021, 1, 1)
    df = await adapter.fetch_ohlcv("BTC/USDT", "1d", since, limit=2)
    # should return a DataFrame with proper dtypes
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dtype == "datetime64[ns]"
    assert df.shape == (2, 6)

@pytest.mark.asyncio
async def test_fetch_order_book(dummy_exchange):
    adapter = CCXTAdapter("binance")
    df = await adapter.fetch_order_book("BTC/USDT", limit=2)
    # expect 4 rows (2 bids + 2 asks)
    assert df.shape[0] == 4
    assert set(df["side"]) == {"bid", "ask"}

@pytest.mark.asyncio
async def test_list_symbols_and_balance(dummy_exchange):
    adapter = CCXTAdapter("binance")
    syms = await adapter.list_symbols()
    assert "BTC/USDT" in syms
    bal = await adapter.fetch_balance()
    assert bal["free"]["USDT"] == 1000

@pytest.mark.asyncio
async def test_close(dummy_exchange):
    adapter = CCXTAdapter("binance")
    # ensure close calls underlying client's close()
    await adapter.close()
    dummy_exchange.close.assert_awaited()

def test_get_ticker_raw(monkeypatch, dummy_exchange):
    # ensure get_ticker_raw spins up adapter, calls fetch, then closes
    # patch CCXTAdapter to use our dummy
    class DummyAdapter:
        def __init__(self, *args, **kwargs):
            self.client = dummy_exchange
        async def fetch_ticker(self, sym):
            return {"symbol": sym, "last": 999}
        async def close(self):
            pass

    monkeypatch.setattr("app.adapters.ccxt_adapter.CCXTAdapter", DummyAdapter)

    result = asyncio.get_event_loop().run_until_complete(get_ticker_raw("ETH/USDT"))
    assert result["last"] == 999

@pytest.mark.asyncio
async def test_watch_ticker(dummy_exchange):
    adapter = CCXTAdapter("binance")
    events = []

    def cb(tick):
        events.append(tick)

    # we expect it to append one update, then the CancelledError stops it
    with pytest.raises(asyncio.CancelledError):
        await adapter.watch_ticker("BTC/USDT", cb)
    assert len(events) == 1
    assert events[0]["last"] == 29300