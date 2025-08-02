import os, httpx
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

GLASSNODE_KEY = os.getenv("GLASSNODE_API_KEY")
COVALENT_KEY  = os.getenv("COVALENT_API_KEY")

async def fetch_glassnode(symbol: str, metric: str, days: int = 1) -> pd.DataFrame:
    """
    Pulls a timeseries from Glassnode for a given metric.
    """
    url = "https://api.glassnode.com/v1/metrics"
    endpoint = f"/{symbol.lower()}/{metric}"
    params = {
        "api_key": GLASSNODE_KEY,
        "a": symbol.upper(),
        "s": int((datetime.utcnow().timestamp() - days*86400)*1000),  # start ts
        "u": int(datetime.utcnow().timestamp()*1000),               # end ts
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url + endpoint, params=params)
        resp.raise_for_status()
        data = resp.json()  # typically a list of [timestamp,value]
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
    url = f"https://api.covalenthq.com/v1/{chain_id}/address/{address}/balances_v2/"
    params = {"key": COVALENT_KEY}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        items = resp.json().get("data",{}).get("items",[])
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