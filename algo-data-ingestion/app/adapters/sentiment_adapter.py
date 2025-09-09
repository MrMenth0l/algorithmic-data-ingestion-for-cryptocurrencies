import os
import tweepy
import pandas as pd
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime
import logging
import asyncio
from prometheus_client import Counter, CollectorRegistry
from app.common.time_norm import standardize_time_column, add_dt_partition, coerce_schema

_METRICS_REGISTRY = CollectorRegistry()
TWITTER_CALLS = Counter('twitter_calls_total', 'Total Twitter API calls', registry=_METRICS_REGISTRY)
TWITTER_RATE_LIMITS = Counter('twitter_rate_limits_total', 'Twitter rate-limit hits', registry=_METRICS_REGISTRY)
TWITTER_PARSE_ERRORS = Counter('twitter_parse_errors_total', 'Total parse errors in Twitter adapter', registry=_METRICS_REGISTRY)
logger = logging.getLogger(__name__)
def _make_default_sentiment():
    return [{"score": 0.0}]

# Avoid heavy model downloads/initialization on import by default. Enable via env.
_ENABLE_PIPELINE = os.getenv("ENABLE_SENTIMENT_PIPELINE", "0") not in ("0", "false", "False", "")
if _ENABLE_PIPELINE:
    try:
        from transformers import pipeline  # import only when enabled
        sentiment_analyzer = pipeline("sentiment-analysis")
    except Exception as e:
        logger.warning(f"Sentiment pipeline load failed: {e}")
        def sentiment_analyzer(text):
            return _make_default_sentiment()
else:
    def sentiment_analyzer(text):
        return _make_default_sentiment()

# You’ll need to set these env vars: TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_BEARER
CLIENT = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER"),
    consumer_key=os.getenv("TWITTER_API_KEY"),
    consumer_secret=os.getenv("TWITTER_API_SECRET"),
)

async def _retry_http(method, *args, retries: int = 3, backoff: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            result = method(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except tweepy.TweepyException as e:
            try:
                logger.warning(f"{method.__name__} attempt {attempt}/{retries} failed: {e}")
                TWITTER_RATE_LIMITS.inc() if "Rate limit" in str(e) else TWITTER_PARSE_ERRORS.inc()
            except Exception:
                pass
            if attempt < retries:
                await asyncio.sleep(backoff * 2 ** (attempt - 1))
    # last attempt
    result = method(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


def _schema() -> Dict[str, str]:
    return {
        "ts": "datetime64[ns, UTC]",
        "author": "string",
        "text": "string",
        "likes": "Int64",
        "retweets": "Int64",
        "sentiment_score": "float64",
        "id": "string",
        "source": "string",
    }


def _empty_df() -> pd.DataFrame:
    df = coerce_schema(pd.DataFrame(), _schema())
    add_dt_partition(df, ts_col="ts")
    return df


async def fetch_twitter_sentiment(
    query: str,
    since: datetime,
    until: datetime,
    max_results: int = 10
) -> pd.DataFrame:
    """
    Fetch recent tweets and attach sentiment scores.
    Returns a normalized DataFrame with UTC timestamps and stable schema.
    """
    # Parameter validation
    if since > until:
        logger.warning("since must be before until")
        return _empty_df()
    if since == until:
        return _empty_df()
    if not (1 <= max_results <= 100):
        logger.warning("max_results must be between 1 and 100")
        return _empty_df()

    TWITTER_CALLS.inc()

    # Ensure UTC ISO-8601 for the API
    def _to_utc_iso(dt: datetime) -> str:
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()

    try:
        tweets = await _retry_http(
            CLIENT.search_recent_tweets,
            query,
            max_results=max_results,
            tweet_fields=["created_at", "author_id", "public_metrics"],
            start_time=_to_utc_iso(since),
            end_time=_to_utc_iso(until),
        )
    except Exception as e:
        logger.error(f"Twitter fetch error: {e}")
        TWITTER_PARSE_ERRORS.inc()
        return _empty_df()

    if not getattr(tweets, "data", None):
        return _empty_df()

    rows: List[Dict[str, Any]] = []
    for t in tweets.data:
        try:
            score = sentiment_analyzer(t.text)[0]["score"]
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            TWITTER_PARSE_ERRORS.inc()
            score = 0.0
        metrics = getattr(t, "public_metrics", None) or {}
        likes = metrics.get("like_count") if isinstance(metrics, dict) else None
        retweets = metrics.get("retweet_count") if isinstance(metrics, dict) else None
        rows.append({
            "ts": pd.to_datetime(getattr(t, "created_at", None), utc=True, errors="coerce"),
            "author": str(getattr(t, "author_id", "")),
            "text": getattr(t, "text", ""),
            "likes": likes,
            "retweets": retweets,
            "sentiment_score": float(score) if score is not None else None,
            "id": str(getattr(t, "id", "")),
            "source": "twitter",
        })

    df = pd.DataFrame(rows)
    df = standardize_time_column(df, candidates=["ts", "created_at"], dest="ts")
    df = coerce_schema(df, _schema())
    add_dt_partition(df, ts_col="ts")
    return df
