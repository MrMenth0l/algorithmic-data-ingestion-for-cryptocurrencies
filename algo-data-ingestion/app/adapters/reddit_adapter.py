import logging
import asyncio
from prometheus_client import Counter, CollectorRegistry
import os
import httpx
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from typing import Literal
from app.common.time_norm import standardize_time_column, add_dt_partition, coerce_schema

# Metrics Registry and Counters
_METRICS_REGISTRY = CollectorRegistry()
REDDIT_API_CALLS = Counter('reddit_api_calls_total', 'Total Reddit API calls', registry=_METRICS_REGISTRY)
PUSHSHIFT_CALLS = Counter('pushshift_calls_total', 'Total Pushshift API calls', registry=_METRICS_REGISTRY)
REDDIT_PARSE_ERRORS = Counter('reddit_parse_errors_total', 'Total parse errors in Reddit adapter', registry=_METRICS_REGISTRY)
logger = logging.getLogger(__name__)

# Retry helper
async def _retry_http(method, *args, retries: int = 3, backoff: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            try:
                logger.warning(f"{method.__name__} attempt {attempt}/{retries} failed: {e}")
            except Exception:
                pass
            if attempt < retries:
                await asyncio.sleep(backoff * 2 ** (attempt - 1))
    return await method(*args, **kwargs)

# Environment variables for Reddit OAuth
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "AlgoDataIngestion/0.1")

async def fetch_reddit_api(subreddit: str, since: datetime, until: datetime, limit: int) -> pd.DataFrame:
    # Parameter validation
    if since >= until:
        raise ValueError("`since` must be before `until`")
    if not (1 <= limit <= 1000):
        raise ValueError("`limit` must be between 1 and 1000")
    REDDIT_API_CALLS.inc()
    # 1. Obtain access token via client credentials
    async with httpx.AsyncClient(auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
                                 headers={"User-Agent": REDDIT_USER_AGENT}) as client:
        token_resp = await _retry_http(client.post,
                                       "https://www.reddit.com/api/v1/access_token",
                                       data={"grant_type": "client_credentials"})
        token_resp.raise_for_status()
        try:
            token = token_resp.json().get("access_token")
        except Exception as e:
            REDDIT_PARSE_ERRORS.inc()
            logger.error(f"Token JSON parse error: {e}")
            # Return schema-stable empty frame
            empty = coerce_schema(pd.DataFrame(), {
                "ts": "datetime64[ns, UTC]",
                "author": "string",
                "title": "string",
                "selftext": "string",
                "score": "Int64",
                "num_comments": "Int64",
                "id": "string",
                "subreddit": "string",
            })
            add_dt_partition(empty, ts_col="ts")
            return empty

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": REDDIT_USER_AGENT
    }
    url = f"https://oauth.reddit.com/r/{subreddit}/new"
    params = {"limit": limit}

    async with httpx.AsyncClient(headers=headers) as client:
        resp = await _retry_http(client.get, url, params=params)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except Exception as e:
            REDDIT_PARSE_ERRORS.inc()
            logger.error(f"JSON parse error in fetch_reddit_api: {e}")
            empty = coerce_schema(pd.DataFrame(), {
                "ts": "datetime64[ns, UTC]",
                "author": "string",
                "title": "string",
                "selftext": "string",
                "score": "Int64",
                "num_comments": "Int64",
                "id": "string",
                "subreddit": "string",
            })
            add_dt_partition(empty, ts_col="ts")
            return empty

        # Validate structure: expect dict with data.children list
        if not isinstance(payload, dict):
            REDDIT_PARSE_ERRORS.inc()
            empty = coerce_schema(pd.DataFrame(), {
                "ts": "datetime64[ns, UTC]",
                "author": "string",
                "title": "string",
                "selftext": "string",
                "score": "Int64",
                "num_comments": "Int64",
                "id": "string",
                "subreddit": "string",
            })
            add_dt_partition(empty, ts_col="ts")
            return empty
        data_obj = payload.get("data") or {}
        children = data_obj.get("children")
        if not isinstance(children, list):
            REDDIT_PARSE_ERRORS.inc()
            empty = coerce_schema(pd.DataFrame(), {
                "ts": "datetime64[ns, UTC]",
                "author": "string",
                "title": "string",
                "selftext": "string",
                "score": "Int64",
                "num_comments": "Int64",
                "id": "string",
                "subreddit": "string",
            })
            add_dt_partition(empty, ts_col="ts")
            return empty
        posts = children

    # Build normalized rows
    rows: List[Dict[str, Any]] = []
    for item in posts:
        data = item.get("data", {})
        created = data.get("created_utc")  # seconds epoch
        rows.append({
            "ts": pd.to_datetime(created, unit="s", utc=True),
            "author": data.get("author", ""),
            "title": data.get("title", ""),
            "selftext": data.get("selftext", ""),
            "score": data.get("score"),
            "num_comments": data.get("num_comments"),
            "id": data.get("id", ""),
            "subreddit": subreddit,
        })

    df = pd.DataFrame(rows)
    # Ensure time column normalized and schema coerced
    df = standardize_time_column(df, candidates=["ts", "created_utc", "created"], dest="ts")
    schema = {
        "ts": "datetime64[ns, UTC]",
        "author": "string",
        "title": "string",
        "selftext": "string",
        "score": "Int64",
        "num_comments": "Int64",
        "id": "string",
        "subreddit": "string",
    }
    df = coerce_schema(df, schema)
    add_dt_partition(df, ts_col="ts")
    return df

async def fetch_pushshift(subreddit: str, since: datetime, until: datetime, limit: int) -> pd.DataFrame:
    # Parameter validation
    if since >= until:
        raise ValueError("`since` must be before `until`")
    if not (1 <= limit <= 1000):
        raise ValueError("`limit` must be between 1 and 1000")
    PUSHSHIFT_CALLS.inc()
    url = "https://api.pushshift.io/reddit/search/submission"
    params = {
        "subreddit": subreddit,
        "size": limit,
        "after": int(since.timestamp()),
        "before": int(until.timestamp()),
    }
    async with httpx.AsyncClient() as client:
        resp = await _retry_http(client.get, url, params=params)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except Exception as e:
            REDDIT_PARSE_ERRORS.inc()
            logger.error(f"JSON parse error in fetch_pushshift: {e}")
            empty = coerce_schema(pd.DataFrame(), {
                "ts": "datetime64[ns, UTC]",
                "author": "string",
                "title": "string",
                "selftext": "string",
                "score": "Int64",
                "num_comments": "Int64",
                "id": "string",
                "subreddit": "string",
            })
            add_dt_partition(empty, ts_col="ts")
            return empty

        # Validate structure: expect dict with data list
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            REDDIT_PARSE_ERRORS.inc()
            empty = coerce_schema(pd.DataFrame(), {
                "ts": "datetime64[ns, UTC]",
                "author": "string",
                "title": "string",
                "selftext": "string",
                "score": "Int64",
                "num_comments": "Int64",
                "id": "string",
                "subreddit": "string",
            })
            add_dt_partition(empty, ts_col="ts")
            return empty
        data = payload.get("data")

    # Build normalized rows
    rows: List[Dict[str, Any]] = []
    for post in data:
        created = post.get("created_utc")  # seconds epoch
        rows.append({
            "ts": pd.to_datetime(created, unit="s", utc=True),
            "author": post.get("author", ""),
            "title": post.get("title", ""),
            "selftext": post.get("selftext", ""),
            "score": post.get("score"),
            "num_comments": post.get("num_comments"),
            "id": post.get("id", ""),
            "subreddit": subreddit,
        })

    df = pd.DataFrame(rows)
    df = standardize_time_column(df, candidates=["ts", "created_utc", "created"], dest="ts")
    schema = {
        "ts": "datetime64[ns, UTC]",
        "author": "string",
        "title": "string",
        "selftext": "string",
        "score": "Int64",
        "num_comments": "Int64",
        "id": "string",
        "subreddit": "string",
    }
    df = coerce_schema(df, schema)
    add_dt_partition(df, ts_col="ts")
    return df

async def fetch_reddit(
    subreddit: str,
    since: datetime,
    until: datetime,
    limit: int = 100,
    source: Literal["api", "pushshift"] = "pushshift"
) -> pd.DataFrame:
    """
    Unified interface: fetch from Reddit API or Pushshift.
    """
    if source == "api":
        return await fetch_reddit_api(subreddit, since, until, limit)
    if source == "pushshift":
        return await fetch_pushshift(subreddit, since, until, limit)
    raise ValueError(f"Unknown source: {source}")