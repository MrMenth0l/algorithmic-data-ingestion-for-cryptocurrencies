# Status: ✅ Exists & wired into async pipeline

import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Any
import pandas as pd
from datetime import datetime
import logging


async def _retry_async(func, *args, retries: int = 3, backoff_factor: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"CCXT call {func.__name__} failed on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                await asyncio.sleep(backoff_factor * 2 ** (attempt - 1))
    # Last attempt
    return await func(*args, **kwargs)

logger = logging.getLogger(__name__)

class CCXTAdapter:
    def __init__(self, exchange_id: str = "binance"):
        exchange_cls = getattr(ccxt, exchange_id)
        self.client = exchange_cls({
            "enableRateLimit": True,
            # you can inject your API keys here if needed
            # "apiKey": "...",
            # "secret": "..."
        })

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches the ticker for a given symbol via CCXT
        Returns the raw JSON dict from the exchange
        """
        return await _retry_async(self.client.fetch_ticker, symbol)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.
        Returns a DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        since_ms = int(since.timestamp() * 1000)
        ohlcv = await _retry_async(self.client.fetch_ohlcv, symbol, timeframe, since_ms, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch current order book snapshot for a symbol.
        Returns a DataFrame with bids and asks.
        """
        ob = await _retry_async(self.client.fetch_order_book, symbol, limit)
        bids = pd.DataFrame(ob['bids'], columns=['price', 'amount'])
        bids['side'] = 'bid'
        asks = pd.DataFrame(ob['asks'], columns=['price', 'amount'])
        asks['side'] = 'ask'
        return pd.concat([bids, asks], ignore_index=True)

    async def watch_ticker(
        self,
        symbol: str,
        callback: callable
    ):
        """
        Stream live ticker updates via WebSocket.
        Calls callback with each ticker update dict.
        """
        await self.client.load_markets()
        while True:
            ticker = await self.client.watch_ticker(symbol)
            callback(ticker)

    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balances.
        Returns the raw balance dict.
        """
        return await _retry_async(self.client.fetch_balance)

    async def list_symbols(self) -> list:
        """
        Return list of symbols available on the exchange.
        """
        markets = await _retry_async(self.client.load_markets)
        return list(markets.keys())

    async def close(self):
        await self.client.close()

# Convenience function if you prefer not to manage client lifecycles
async def get_ticker_raw(symbol: str) -> Dict[str, Any]:
    adapter = CCXTAdapter()
    try:
        return await adapter.fetch_ticker(symbol)
    finally:
        await adapter.close()