

import asyncio
from app.adapters.news_adapter import fetch_news_api, fetch_news_rss
import pandas as pd
from datetime import datetime

class NewsClient:
    """
    Wrapper around news_adapter for fetching news headlines and streaming RSS feeds.
    """

    def __init__(self):
        """
        Initialize NewsClient. Currently, no credentials are required;
        adapters read from environment or config.
        """
        pass

    def get_crypto_news(
        self,
        since: datetime,
        until: datetime,
        source: str = "crypto_news_api",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch news articles from the specified source between 'since' and 'until'.
        Returns a DataFrame with normalized columns (e.g., ts, title, url, summary, source).
        """
        return asyncio.get_event_loop().run_until_complete(
            fetch_news_api(since, until, source, limit)
        )

    def stream_rss(
        self,
        feed_url: str,
        handle_update: callable
    ):
        """
        Stream RSS feed updates. Calls handle_update with each new item dict.
        """
        # Delegate to the async RSS feed streamer in the adapter
        asyncio.get_event_loop().run_until_complete(
            fetch_news_rss(feed_url, handle_update)
        )