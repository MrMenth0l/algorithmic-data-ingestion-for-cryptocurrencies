# Status: ✅ Exists & handles both REST news and RSS feeds
import os
import asyncio
import httpx
import feedparser
import logging
from typing import List, Dict, Any
from datetime import datetime

async def _retry_http(method, *args, retries: int = 3, backoff_factor: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"HTTP call {method.__name__} failed attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                await asyncio.sleep(backoff_factor * 2 ** (attempt - 1))
    # Last attempt
    return await method(*args, **kwargs)

logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

async def fetch_crypto_news(query: str, limit: int) -> List[Dict[str, Any]]:
    url = "https://cryptonews-api.com/api/v1/category"
    params = {
        "section": query,
        "items": limit,
        "token": NEWS_API_KEY,
    }
    async with httpx.AsyncClient() as client:
        resp = await _retry_http(client.get, url, params=params)
        resp.raise_for_status()
        articles = resp.json().get("data", [])
    results: List[Dict[str, Any]] = []
    for art in articles:
        news_url = art.get("news_url", "")
        id = news_url.rstrip("/").split("/")[-1]
        results.append({
            "id": id,
            "title": art.get("title", ""),
            "url": news_url,
            "source": art.get("source_name", ""),
            "author": art.get("author", None),
            "description": art.get("text", None),
            "published_at": datetime.fromisoformat(art.get("date")) if art.get("date") else None,
        })
    return results

async def fetch_rss_feed(feed_url: str, handle_update: callable, poll_interval: int = 60):
    seen_ids = set()
    while True:
        async with httpx.AsyncClient() as client:
            resp = await _retry_http(client.get, feed_url)
            resp.raise_for_status()
            content = resp.text
        feed = feedparser.parse(content)
        for entry in feed.entries:
            entry_id = entry.get("id") or entry.get("link")
            if entry_id not in seen_ids:
                seen_ids.add(entry_id)
                item = {
                    "id": entry_id,
                    "title": entry.get("title"),
                    "url": entry.get("link"),
                    "summary": entry.get("summary"),
                    "published_at": entry.get("published"),
                }
                result = handle_update(item)
                if asyncio.iscoroutine(result):
                    await result
        await asyncio.sleep(poll_interval)