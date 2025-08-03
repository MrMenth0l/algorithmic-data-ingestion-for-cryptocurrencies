import os
import tweepy
import pandas as pd
from typing import Literal
from datetime import datetime
import logging
import asyncio
from prometheus_client import Counter, CollectorRegistry
from transformers import pipeline

_METRICS_REGISTRY = CollectorRegistry()
TWITTER_CALLS = Counter('twitter_calls_total', 'Total Twitter API calls', registry=_METRICS_REGISTRY)
TWITTER_RATE_LIMITS = Counter('twitter_rate_limits_total', 'Twitter rate-limit hits', registry=_METRICS_REGISTRY)
TWITTER_PARSE_ERRORS = Counter('twitter_parse_errors_total', 'Total parse errors in Twitter adapter', registry=_METRICS_REGISTRY)
logger = logging.getLogger(__name__)
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    logger.warning(f"Sentiment pipeline load failed: {e}")
    # Fallback analyzer returns neutral score
    def sentiment_analyzer(text):
        return [{"score": 0.0}]

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

async def fetch_twitter_sentiment(
    query: str,
    since: datetime,
    until: datetime,
    max_results: int = 10
) -> pd.DataFrame:
    # Parameter validation
    if since > until:
        raise ValueError("since must be before until")
    # If no time window, return empty DataFrame without error
    if since == until:
        return pd.DataFrame([], columns=["ts", "user", "text", "sentiment_score"])
    TWITTER_CALLS.inc()
    if not (1 <= max_results <= 100):
        raise ValueError("max_results must be between 1 and 100")
    try:
        tweets = await _retry_http(CLIENT.search_recent_tweets,
            query,
            max_results=max_results,
            tweet_fields=["created_at","author_id"],
            start_time=since.isoformat(),
            end_time=until.isoformat()
        )
    except Exception as e:
        logger.error(f"Twitter fetch error: {e}")
        TWITTER_PARSE_ERRORS.inc()
        return pd.DataFrame([], columns=["ts", "user", "text", "sentiment_score"])
    results = []
    for t in tweets.data:
        try:
            score = sentiment_analyzer(t.text)[0]['score']
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            TWITTER_PARSE_ERRORS.inc()
            score = 0.0
        results.append({
            "source": "twitter",
            "id": t.id,
            "text": t.text,
            "user": str(t.author_id),
            "created_at": t.created_at,
            "sentiment_score": score,
        })

    # Build DataFrame with required columns
    records = []
    for item in results:
        records.append({
            "ts": pd.to_datetime(item["created_at"], utc=True),
            "user": item["user"],
            "text": item["text"],
            "sentiment_score": item["sentiment_score"],
        })
    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame([], columns=["ts", "user", "text", "sentiment_score"])
    return df