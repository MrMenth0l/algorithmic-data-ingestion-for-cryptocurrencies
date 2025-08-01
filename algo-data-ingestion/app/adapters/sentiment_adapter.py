import os
import tweepy
import pandas as pd
from typing import Literal
from datetime import datetime

# You’ll need to set these env vars: TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_BEARER
CLIENT = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER"),
    consumer_key=os.getenv("TWITTER_API_KEY"),
    consumer_secret=os.getenv("TWITTER_API_SECRET"),
)

async def fetch_twitter_sentiment(
    query: str,
    since: datetime,
    until: datetime,
    max_results: int = 10
) -> pd.DataFrame:
    tweets = CLIENT.search_recent_tweets(
        query,
        max_results=max_results,
        tweet_fields=["created_at","author_id"],
        start_time=since.isoformat(),
        end_time=until.isoformat()
    )
    results = []
    for t in tweets.data:
        results.append({
            "source": "twitter",
            "id": t.id,
            "text": t.text,
            "user": str(t.author_id),
            "created_at": t.created_at,
            "sentiment_score": 0.0,  # will fill in later
        })

    df = pd.DataFrame([{
        "ts": pd.to_datetime(item["created_at"], utc=True),
        "user": item["user"],
        "text": item["text"],
        "sentiment_score": item["sentiment_score"]
    } for item in results])
    return df