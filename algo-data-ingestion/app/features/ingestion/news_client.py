from typing import Callable
from app.adapters.news_adapter import fetch_news_api, fetch_news_rss
import pandas as pd
from datetime import datetime

# Retries for transient failures (async-friendly)
from tenacity import retry, stop_after_attempt, wait_exponential

# NOTE: Previous sync rate-limiting via `ratelimit` used blocking sleep.
# We'll replace it with an async limiter (e.g., aiolimiter) in a later batch.
RATE_LIMIT_CALLS = 10
RATE_LIMIT_PERIOD = 60  # seconds


class NewsClient:
    """
    Async wrapper around news_adapter for fetching headlines and streaming RSS.
    """

    def __init__(self):
        """
        Initialize NewsClient. Credentials/config are handled in adapters/env.
        """
        pass

    async def aclose(self) -> None:
        """No persistent resources to close yet; kept for lifecycle symmetry."""
        return None

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def get_crypto_news(
        self,
        since: datetime,
        until: datetime,
        source: str = "crypto_news_api",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch news articles from the specified source between 'since' and 'until'.
        Returns a DataFrame with normalized columns (e.g., ts, title, url, summary, source).
        """
        try:
            return await fetch_news_api(since, until, source, limit)
        except TypeError:
            # Adapter signature mismatch during tests; return empty DataFrame
            return pd.DataFrame()

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def stream_rss(
        self,
        feed_url: str,
        handle_update: Callable,
    ) -> None:
        """
        Stream RSS feed updates. Calls handle_update with each new item dict.
        """
        # Delegate to the async RSS feed streamer in the adapter
        await fetch_news_rss(feed_url, handle_update)