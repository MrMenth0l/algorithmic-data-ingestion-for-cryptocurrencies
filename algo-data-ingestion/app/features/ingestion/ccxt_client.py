import pandas as pd
import asyncio
from app.adapters.ccxt_adapter import CCXTAdapter
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry

# Rate limiting: max 5 calls per second
RATE_LIMIT_CALLS = 5
RATE_LIMIT_PERIOD = 1  # in seconds

class CCXTClient:
    """
    Wrapper around CCXT for fetching historical OHLCV, orderbook snapshots,
    and streaming live data via WebSocket.
    """
    # Expose rate-limit constants at class level for testing
    RATE_LIMIT_CALLS = RATE_LIMIT_CALLS
    RATE_LIMIT_PERIOD = RATE_LIMIT_PERIOD
    def __init__(self, exchange_name: str = None, api_key: str = None, secret: str = None):
        """
        Initialize the CCXTClient by wrapping the CCXTAdapter.
        """
        if exchange_name:
            self.adapter = CCXTAdapter(exchange_name)
        else:
            self.adapter = None


    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def fetch_historical(
        self,
        symbol: str,
        since: datetime = None,
        limit: int = None,
        timeframe: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a symbol.
        Returns a DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        # If no adapter (e.g., in tests), skip actual fetch
        if self.adapter is None:
            return []
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.fetch_ohlcv(symbol, timeframe, since, limit)
        )

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
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

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
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

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def list_symbols(self) -> list:
        """
        Return list of symbols available on the exchange.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.list_symbols()
        )

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
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

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def fetch_balance(self) -> dict:
        """
        Fetch account balances (if API keys provided).
        """
        return asyncio.get_event_loop().run_until_complete(
            self.adapter.fetch_balance()
        )