import pandas as pd
from typing import Dict

# Canonical Parquet schemas for each data domain
# All timestamps are tz-aware UTC; textual fields use pandas 'string' dtype; integers use nullable 'Int64'.

# Market (OHLCV)
MARKET_SCHEMA: Dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "symbol": "string",
    "exchange": "string",
    "timeframe": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "dt": "string",
}

# On-chain (Glassnode / Covalent)
ONCHAIN_SCHEMA: Dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "source": "string",
    "symbol": "string",
    "metric": "string",
    "value": "float64",
    "contract_address": "string",
    "contract_name": "string",
    "dt": "string",
}

# Social (Twitter / Reddit)
SOCIAL_SCHEMA: Dict[str, str] = {
    "ts": "datetime64[ns, UTC]",
    "source": "string",          # e.g., "twitter" or "reddit"
    "id": "string",
    "author": "string",
    "text": "string",
    "title": "string",
    "selftext": "string",
    "likes": "Int64",
    "retweets": "Int64",
    "score": "Int64",
    "num_comments": "Int64",
    "sentiment_score": "float64",
    "subreddit": "string",
    "dt": "string",
}

# News (API / RSS)
NEWS_SCHEMA: Dict[str, str] = {
    "published_at": "datetime64[ns, UTC]",
    "id": "string",
    "title": "string",
    "url": "string",
    "source": "string",
    "author": "string",
    "description": "string",
    "dt": "string",
}