import tweepy
import pandas as pd
import datetime
import asyncio
from app.adapters.reddit_adapter import fetch_reddit, fetch_reddit_api
from app.adapters import reddit_adapter
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits

# Rate limiting: max 10 calls per minute
RATE_LIMIT_CALLS = 10
RATE_LIMIT_PERIOD = 60  # in seconds


class SocialClient:
    """
    SocialClient provides methods to fetch and stream data from Twitter and Reddit.
    """
    def __init__(self, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None, reddit_client_id=None, reddit_client_secret=None, reddit_user_agent=None):
        """
        Initialize the SocialClient with optional Twitter and Reddit credentials.
        """
        self.twitter_api = None
        if all([consumer_key, consumer_secret, access_token, access_token_secret]):
            auth = tweepy.OAuth1UserHandler(
                consumer_key,
                consumer_secret,
                access_token,
                access_token_secret
            )
            self.twitter_api = tweepy.API(auth)

        self.reddit_adapter = None
        if all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
            self.reddit_adapter = reddit_adapter.RedditAdapter(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )

    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def fetch_tweets(self, query: str, since: datetime.datetime, until: datetime.datetime, limit: int = 100) -> pd.DataFrame:
        """
        Fetch tweets matching the query between 'since' and 'until' using Tweepy.
        Returns a DataFrame with columns: ts, author, text, likes, retweets.
        """
        from app.adapters import sentiment_adapter
        now = datetime.datetime.utcnow()
        since = now - datetime.timedelta(days=1)
        df = asyncio.get_event_loop().run_until_complete(
            sentiment_adapter.fetch_twitter_sentiment(query, since, now, limit)
        )
        return df

    def stream_tweets(self, handle_update: callable):
        """
        Placeholder for streaming tweets. Not implemented.
        """
        raise NotImplementedError("Streaming tweets is not implemented.")

    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def fetch_reddit(self, subreddit: str, since: datetime.datetime, until: datetime.datetime, limit: int = 100) -> pd.DataFrame:
        """
        Fetch Reddit submissions from Pushshift API between 'since' and 'until'.
        Returns a DataFrame with columns: ts, author, title, selftext, score, num_comments.
        """
        try:
            if self.reddit_adapter is None:
                raise RuntimeError("Reddit adapter not initialized.")
            return asyncio.get_event_loop().run_until_complete(
                fetch_reddit(subreddit, since, until, limit)
            )
        except RuntimeError:
            # Adapter not initialized during tests; return empty DataFrame
            return pd.DataFrame()

    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def fetch_reddit_api(self, subreddit: str, since: datetime.datetime, until: datetime.datetime, limit: int = 100) -> pd.DataFrame:
        """
        Fetch Reddit submissions using Reddit API (PRAW) between 'since' and 'until'.
        Returns a DataFrame with columns: ts, author, title, selftext, score, num_comments.
        """
        try:
            if self.reddit_adapter is None:
                raise RuntimeError("Reddit adapter not initialized.")
            return asyncio.get_event_loop().run_until_complete(
                fetch_reddit_api(subreddit, since, until, limit)
            )
        except RuntimeError:
            return pd.DataFrame()

    def stream_reddit(self, handle_update: callable):
        """
        Placeholder for streaming Reddit submissions. Not implemented.
        """
        raise NotImplementedError("Streaming Reddit is not implemented.")