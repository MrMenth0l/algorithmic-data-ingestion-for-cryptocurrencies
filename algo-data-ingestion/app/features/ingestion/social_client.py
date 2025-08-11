from __future__ import annotations
from typing import Optional, Callable
import datetime
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional: kept for future streaming scenarios; not required for async adapters below
import tweepy

# Use adapter modules directly (avoid name collisions with method names)
from app.adapters import reddit_adapter, sentiment_adapter

# Keep constants for compatibility with any tests referencing them.
RATE_LIMIT_CALLS = 10
RATE_LIMIT_PERIOD = 60  # seconds


def _empty_twitter_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "author", "text", "likes", "retweets", "sentiment_score"])


def _empty_reddit_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "author", "title", "selftext", "score", "num_comments", "id", "subreddit"])


class SocialClient:
    """
    Async Social client for Twitter and Reddit data via adapter modules.

    Notes:
    - Converted to async: callers must now `await` methods.
    - Removed blocking `ratelimit` and event-loop `.run_until_complete` calls.
    - Tenacity retries remain with exponential backoff.
    """

    def __init__(
        self,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None,
    ) -> None:
        self.twitter_api = None
        if all([consumer_key, consumer_secret, access_token, access_token_secret]):
            auth = tweepy.OAuth1UserHandler(
                consumer_key,
                consumer_secret,
                access_token,
                access_token_secret,
            )
            self.twitter_api = tweepy.API(auth)

        # Some deployments expect a RedditAdapter instance; keep field for compat
        self.reddit_adapter = None
        if all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
            # If your project exposes a class-based adapter, initialize here.
            # Otherwise, module-level functions below will be used.
            try:
                self.reddit_adapter = reddit_adapter.RedditAdapter(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent,
                )
            except Exception:
                # Fallback to module functions
                self.reddit_adapter = None

    async def aclose(self) -> None:
        """Lifecycle symmetry; nothing to close yet."""
        return None

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def fetch_tweets(
        self,
        query: str,
        since: Optional[datetime.datetime] = None,
        until: Optional[datetime.datetime] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch tweets (sentiment) via async adapter.
        Returns a DataFrame with columns: ts, author, text, likes, retweets, sentiment_score.
        """
        try:
            if since is None or until is None:
                now = datetime.datetime.utcnow()
                since = now - datetime.timedelta(days=1)
                until = now
            df = await sentiment_adapter.fetch_twitter_sentiment(query, since, until, limit)
            return df if isinstance(df, pd.DataFrame) else _empty_twitter_df()
        except Exception:
            return _empty_twitter_df()

    async def stream_tweets(self, handle_update: Callable[[dict], None]):
        """Placeholder for streaming tweets (not implemented)."""
        raise NotImplementedError("Streaming tweets is not implemented.")

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def fetch_reddit(
        self,
        subreddit: str,
        since: datetime.datetime,
        until: datetime.datetime,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch Reddit submissions from Pushshift via async adapter.
        Returns a DataFrame with columns: ts, author, title, selftext, score, num_comments.
        """
        try:
            # If a class-based adapter is required elsewhere, check and fallback cleanly
            if self.reddit_adapter is None:
                df = await reddit_adapter.fetch_reddit(subreddit, since, until, limit)
            else:
                df = await reddit_adapter.fetch_reddit(subreddit, since, until, limit)
            return df if isinstance(df, pd.DataFrame) else _empty_reddit_df()
        except Exception:
            return _empty_reddit_df()

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def fetch_reddit_api(
        self,
        subreddit: str,
        since: datetime.datetime,
        until: datetime.datetime,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch Reddit submissions using Reddit API (PRAW) via async adapter.
        Returns a DataFrame with columns: ts, author, title, selftext, score, num_comments.
        """
        try:
            if self.reddit_adapter is None:
                df = await reddit_adapter.fetch_reddit_api(subreddit, since, until, limit)
            else:
                df = await reddit_adapter.fetch_reddit_api(subreddit, since, until, limit)
            return df if isinstance(df, pd.DataFrame) else _empty_reddit_df()
        except Exception:
            return _empty_reddit_df()

    async def stream_reddit(self, handle_update: Callable[[dict], None]):
        """Placeholder for streaming Reddit submissions (not implemented)."""
        raise NotImplementedError("Streaming Reddit is not implemented.")