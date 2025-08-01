import asyncio
from app.adapters.onchain_adapter import fetch_glassnode, fetch_covalent
import pandas as pd
from datetime import datetime

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
        return asyncio.get_event_loop().run_until_complete(
            fetch_glassnode(symbol, metric, days)
        )

    def get_covalent_balances(
        self,
        chain_id: int,
        address: str
    ) -> pd.DataFrame:
        """
        Fetch token balances for an address from Covalent.
        Returns a DataFrame with columns [timestamp, value, contract_address, contract_name, etc.].
        """
        return asyncio.get_event_loop().run_until_complete(
            fetch_covalent(chain_id, address)
        )