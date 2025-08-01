import pandas as pd
import asyncio
from app.adapters.ccxt_adapter import CCXTAdapter
from datetime import datetime

class CCXTClient:
    """
    Wrapper around CCXT for fetching historical OHLCV, orderbook snapshots,
    and streaming live data via WebSocket.
    """
    def __init__(self, exchange_name: str, api_key: str = None, secret: str = None):
        """
        Initialize the CCXTClient by wrapping the CCXTAdapter.
        """
        self.adapter = CCXTAdapter(exchange_name)

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a symbol.
        Returns a DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.fetch_ohlcv(symbol, timeframe, since, limit)
        )

    def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch current order book snapshot for a symbol.
        Returns a DataFrame with bids and asks.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.fetch_order_book(symbol, limit)
        )

    async def _watch_ticker(self, symbol: str, handle_update):
        """
        Internal: uses CCXT async WebSocket to watch ticker updates.
        """
        await self.adapter.watch_ticker(symbol, handle_update)

    def stream_ticker(
        self,
        symbol: str,
        handle_update: callable
    ):
        """
        Stream live ticker updates via WebSocket.
        handle_update will be called with each update dict.
        """
        asyncio.get_event_loop().run_until_complete(
            self.adapter.watch_ticker(symbol, handle_update)
        )

    def list_symbols(self) -> list:
        """
        Return list of symbols available on the exchange.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.list_symbols()
        )

    def get_ticker(
        self,
        symbol: str
    ) -> dict:
        """
        Fetch a one-off ticker snapshot via REST.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.fetch_ticker(symbol)
        )

    def fetch_balance(self) -> dict:
        """
        Fetch account balances (if API keys provided).
        """
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.fetch_balance()
        )