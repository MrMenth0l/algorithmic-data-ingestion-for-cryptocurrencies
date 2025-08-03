import os, httpx
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import logging
from prometheus_client import Counter, CollectorRegistry
import asyncio

GLASSNODE_KEY = os.getenv("GLASSNODE_API_KEY")
COVALENT_KEY  = os.getenv("COVALENT_API_KEY")

_METRICS_REGISTRY = CollectorRegistry()
GLASSNODE_CALLS = Counter('onchain_glassnode_requests_total', 'Total Glassnode requests', registry=_METRICS_REGISTRY)
COVALENT_CALLS = Counter('onchain_covalent_requests_total', 'Total Covalent requests', registry=_METRICS_REGISTRY)
ONCHAIN_PARSE_ERRORS = Counter('onchain_parse_errors_total', 'Total parse errors in on-chain adapter', registry=_METRICS_REGISTRY)
logger = logging.getLogger(__name__)

async def _retry_http(method, *args, retries: int = 3, backoff: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            try:
                logger.warning(f"On-chain call {method.__name__} failed attempt {attempt}/{retries}: {e}")
            except Exception:
                pass
            if attempt < retries:
                await asyncio.sleep(backoff * 2 ** (attempt - 1))
    return await method(*args, **kwargs)

async def fetch_glassnode(symbol: str, metric: str, days: int = 1) -> pd.DataFrame:
    """
    Pulls a timeseries from Glassnode for a given metric.
    """
    GLASSNODE_CALLS.inc()
    url = "https://api.glassnode.com/v1/metrics"
    endpoint = f"/{symbol.lower()}/{metric}"
    params = {
        "api_key": GLASSNODE_KEY,
        "a": symbol.upper(),
        "s": int((datetime.utcnow().timestamp() - days*86400)*1000),  # start ts
        "u": int(datetime.utcnow().timestamp()*1000),               # end ts
    }
    async with httpx.AsyncClient() as client:
        resp = await _retry_http(client.get, url + endpoint, params=params)
        resp.raise_for_status()
        try:
            data = resp.json()  # typically a list of [timestamp,value]
        except Exception as e:
            ONCHAIN_PARSE_ERRORS.inc()
            logger.error(f"JSON parse error in fetch_glassnode: {e}")
            return pd.DataFrame([])
        # Return empty DataFrame on no or invalid data
        if not data or not isinstance(data, list):
            ONCHAIN_PARSE_ERRORS.inc()
            return pd.DataFrame([])
    df = pd.DataFrame([{
        "source": "glassnode",
        "symbol": symbol,
        "metric": metric,
        "timestamp": point[0],
        "value": point[1]
    } for point in data])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

async def fetch_covalent(chain_id: int, address: str) -> pd.DataFrame:
    """
    Fetches address balances from Covalent.
    """
    COVALENT_CALLS.inc()
    url = f"https://api.covalenthq.com/v1/{chain_id}/address/{address}/balances_v2/"
    params = {"key": COVALENT_KEY}
    async with httpx.AsyncClient() as client:
        resp = await _retry_http(client.get, url, params=params)
        resp.raise_for_status()
        try:
            items = resp.json().get("data",{}).get("items",[])
        except Exception as e:
            ONCHAIN_PARSE_ERRORS.inc()
            logger.error(f"JSON parse error in fetch_covalent: {e}")
            return pd.DataFrame([])
        # Return empty DataFrame on no or invalid items
        if not items or not isinstance(items, list):
            ONCHAIN_PARSE_ERRORS.inc()
            return pd.DataFrame([])
    results = []
    for token in items:
        results.append({
            "source": "covalent",
            "symbol": token["contract_ticker_symbol"],
            "metric": "balance",
            "timestamp": int(datetime.utcnow().timestamp()*1000),
            "value": float(token["balance"]) / (10**token["contract_decimals"])
        })
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df