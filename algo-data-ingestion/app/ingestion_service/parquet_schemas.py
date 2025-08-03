import pandas as pd
from typing import Dict

# Canonical Parquet schemas for each data domain

# Market (OHLCV)
MARKET_SCHEMA: Dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "symbol": "object",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

# On-chain (Glassnode / Covalent)
ONCHAIN_SCHEMA: Dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "source": "object",
    "asset": "object",  # symbol or address
    "metric": "object",
    "value": "float64",
}

# Social (Twitter / Reddit)
SOCIAL_SCHEMA: Dict[str, str] = {
    "ts": "datetime64[ns, UTC]",
    "source": "object",        # "twitter" or "reddit"
    "id": "object",
    "user": "object",
    "text": "object",
    "sentiment_score": "float64",
}

# News (API / RSS)
NEWS_SCHEMA: Dict[str, str] = {
    "published_at": "datetime64[ns, UTC]",
    "source_type": "object",   # "api" or "rss"
    "source_name": "object",
    "title": "object",
    "url": "object",
    "author": "object",
    "summary": "object",
}