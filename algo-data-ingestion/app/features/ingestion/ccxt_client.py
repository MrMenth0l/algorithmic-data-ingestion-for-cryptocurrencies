from __future__ import annotations
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime
import inspect
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters.ccxt_adapter import CCXTAdapter

# Keeping these constants for tests that reference them, but
# we no longer use blocking decorators in async code.
RATE_LIMIT_CALLS = 5
RATE_LIMIT_PERIOD = 1  # seconds


class CCXTClient:
    """
    Async wrapper around CCXTAdapter for fetching historical OHLCV,
    order book snapshots, and streaming live data.
    """

    # Expose rate-limit constants at class level for testing
    RATE_LIMIT_CALLS = RATE_LIMIT_CALLS
    RATE_LIMIT_PERIOD = RATE_LIMIT_PERIOD

    def __init__(self, exchange_name: Optional[str] = None, api_key: Optional[str] = None, secret: Optional[str] = None):
        """
        Initialize the CCXTClient by wrapping the CCXTAdapter.
        Note: adapter creds handled internally if supported; we keep signature for future.
        """
        if exchange_name:
            # Preserve existing adapter construction to avoid signature drift
            self.adapter = CCXTAdapter(exchange_name)
        else:
            self.adapter = None

    async def aclose(self) -> None:
        """Attempt to gracefully close underlying adapter resources."""
        if not self.adapter:
            return
        # Support either async aclose() or close() (sync/async)
        close = getattr(self.adapter, "aclose", None) or getattr(self.adapter, "close", None)
        if callable(close):
            result = close()
            if inspect.iscoroutine(result):
                await result

    # -----------------
    # Async API methods
    # -----------------

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def fetch_historical(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a symbol.
        Returns a DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        if timeframe is None:
            timeframe = "1m"
        if self.adapter is None:
            return pd.DataFrame()
        return await self.adapter.fetch_ohlcv(symbol, timeframe, since, limit)

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch current order book snapshot for a symbol.
        Returns a DataFrame with bids and asks.
        """
        if self.adapter is None:
            return pd.DataFrame()
        return await self.adapter.fetch_order_book(symbol, limit)

    async def _watch_ticker(self, symbol: str, handle_update: Callable[[Dict[str, Any]], None]):
        """Internal: uses adapter to watch ticker updates."""
        if self.adapter is None:
            return
        await self.adapter.watch_ticker(symbol, handle_update)

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def stream_ticker(self, symbol: str, handle_update: Callable[[Dict[str, Any]], None]):
        """
        Stream live ticker updates via WebSocket. Calls handle_update per item.
        """
        if self.adapter is None:
            return
        await self.adapter.watch_ticker(symbol, handle_update)

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def list_symbols(self) -> List[str]:
        """Return list of symbols available on the exchange."""
        if self.adapter is None:
            return []
        return await self.adapter.list_symbols()

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch a one-off ticker snapshot via REST."""
        if self.adapter is None:
            return {}
        return await self.adapter.fetch_ticker(symbol)

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balances (if API keys provided)."""
        if self.adapter is None:
            return {}
        return await self.adapter.fetch_balance()