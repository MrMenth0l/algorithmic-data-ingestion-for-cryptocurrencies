from __future__ import annotations
from typing import Optional
import logging
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters.onchain_adapter import fetch_glassnode, fetch_covalent

# Keep constants for compatibility with any tests referencing them.
RATE_LIMIT_CALLS = 10
RATE_LIMIT_PERIOD = 60  # seconds


def _empty_glassnode_df() -> pd.DataFrame:
    """Schema-stable empty frame for Glassnode responses."""
    return pd.DataFrame(columns=["source", "symbol", "metric", "timestamp", "value"])


def _empty_covalent_df() -> pd.DataFrame:
    """Schema-stable empty frame for Covalent responses."""
    return pd.DataFrame(columns=["timestamp", "contract_address", "contract_name", "value"])


class OnchainClient:
    """
    Async wrapper around on-chain adapters for fetching blockchain metrics.

    Guarantees:
    - Never raises out of these methods; returns an EMPTY DataFrame on error.
    - Sets `self.last_error` with a brief message if an error occurred.
    - DataFrames always have a stable schema so FastAPI JSON serialization won't choke.
    """

    def __init__(self) -> None:
        # Adapters read creds/config from environment as needed.
        self.last_error: Optional[str] = None

    async def aclose(self) -> None:
        """Lifecycle symmetry; nothing to close yet."""
        return None

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def get_glassnode_metric(
        self,
        symbol: str,
        metric: str,
        days: int = 1,
    ) -> pd.DataFrame:
        """
        Fetch a time series from Glassnode for a given symbol and metric.
        Returns a DataFrame with columns [source, symbol, metric, timestamp, value].
        Never raises; returns empty DataFrame on error.
        """
        self.last_error = None
        try:
            df = await fetch_glassnode(symbol, metric, days)
            if df is None:
                return _empty_glassnode_df()
            # Basic schema guard to keep responses JSON-serializable downstream
            required = {"timestamp", "value"}
            if not required.issubset(set(df.columns)):
                return _empty_glassnode_df()
            return df
        except Exception as e:  # broad catch to keep API stable without keys/config
            logging.warning("Glassnode fetch failed: %s", e)
            self.last_error = str(e)
            return _empty_glassnode_df()

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def get_covalent_balances(
        self,
        chain_id: int,
        address: str,
    ) -> pd.DataFrame:
        """
        Fetch token balances for an address from Covalent.
        Returns a DataFrame with columns [timestamp, value, contract_address, contract_name, ...].
        Never raises; returns empty DataFrame on error.
        """
        self.last_error = None
        try:
            df = await fetch_covalent(chain_id, address)
            if df is None:
                return _empty_covalent_df()
            required = {"timestamp", "value"}
            if not required.issubset(set(df.columns)):
                return _empty_covalent_df()
            return df
        except Exception as e:
            logging.warning("Covalent fetch failed: %s", e)
            self.last_error = str(e)
            return _empty_covalent_df()