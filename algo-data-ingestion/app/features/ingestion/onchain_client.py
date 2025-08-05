import asyncio
from app.adapters.onchain_adapter import fetch_glassnode, fetch_covalent
import pandas as pd
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
from httpx import HTTPStatusError

# Rate limiting: max 10 calls per minute
RATE_LIMIT_CALLS = 10
RATE_LIMIT_PERIOD = 60  # in seconds

class OnchainClient:
    """
    Wrapper around on-chain adapters for fetching blockchain metrics.
    """

    def __init__(self):
        """
        Initialize OnchainClient. Currently, no credentials are required,
        as adapters read from environment variables.
        """
        pass

    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def get_glassnode_metric(
        self,
        symbol: str,
        metric: str,
        days: int = 1
    ) -> pd.DataFrame:
        """
        Fetch a time series from Glassnode for a given symbol and metric.
        Returns a DataFrame with columns [source, symbol, metric, timestamp, value].
        """
        try:
            return asyncio.get_event_loop().run_until_complete(
                fetch_glassnode(symbol, metric, days)
            )
        except HTTPStatusError:
            return pd.DataFrame()

    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def get_covalent_balances(
        self,
        chain_id: int,
        address: str
    ) -> pd.DataFrame:
        """
        Fetch token balances for an address from Covalent.
        Returns a DataFrame with columns [timestamp, value, contract_address, contract_name, etc.].
        """
        try:
            return asyncio.get_event_loop().run_until_complete(
                fetch_covalent(chain_id, address)
            )
        except HTTPStatusError:
            return pd.DataFrame()