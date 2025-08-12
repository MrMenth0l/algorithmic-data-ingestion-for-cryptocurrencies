# Status: ✅ Exists & wired into async pipeline

import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import logging
from app.common.time_norm import to_utc_dt, add_dt_partition, coerce_schema


async def _retry_async(func, *args, retries: int = 3, backoff_factor: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            try:
                logger.warning(f"CCXT call {func.__name__} failed on attempt {attempt}/{retries}: {e}")
            except Exception:
                pass
            if attempt < retries:
                await asyncio.sleep(backoff_factor * 2 ** (attempt - 1))
    # Last attempt
    return await func(*args, **kwargs)

logger = logging.getLogger("app.adapters.ccxt_adapter")

class CCXTAdapter:
    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id
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
        since: Optional[datetime] = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.
        Returns a DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        if since is not None:
            since_ms = int(since.timestamp() * 1000)
            ohlcv = await _retry_async(self.client.fetch_ohlcv, symbol, timeframe, since_ms, limit)
        else:
            ohlcv = await _retry_async(self.client.fetch_ohlcv, symbol, timeframe, None, limit)
        df = pd.DataFrame(ohlcv, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        # Make timestamp tz-aware UTC, add identifying columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['symbol'] = symbol
        df['exchange'] = self.exchange_id
        df['timeframe'] = timeframe

        # Enforce schema and add dt partition
        schema = {
            'timestamp': 'datetime64[ns, UTC]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'symbol': 'string',
            'exchange': 'string',
            'timeframe': 'string',
        }
        df = coerce_schema(df, schema)
        add_dt_partition(df, ts_col='timestamp')
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
        # Stamp snapshot time and identifiers, normalize schema
        now_utc = pd.Timestamp.now(tz='UTC')
        bids['timestamp'] = now_utc
        asks['timestamp'] = now_utc
        bids['symbol'] = symbol
        asks['symbol'] = symbol
        bids['exchange'] = self.exchange_id
        asks['exchange'] = self.exchange_id

        df = pd.concat([bids, asks], ignore_index=True)
        schema = {
            'timestamp': 'datetime64[ns, UTC]',
            'price': 'float64',
            'amount': 'float64',
            'side': 'string',
            'symbol': 'string',
            'exchange': 'string',
        }
        df = coerce_schema(df, schema)
        add_dt_partition(df, ts_col='timestamp')
        return df

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