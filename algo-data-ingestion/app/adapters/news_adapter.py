# Status: ✅ Exists & handles both REST news and RSS feeds
import os
import asyncio
import httpx
import feedparser
import logging
from typing import List, Dict, Any
from datetime import datetime
from prometheus_client import Counter, CollectorRegistry

_METRICS_REGISTRY = CollectorRegistry()

NEWS_API_CALLS = Counter('news_api_calls_total', 'Total number of news API calls', registry=_METRICS_REGISTRY)
NEWS_RSS_POLLS = Counter('news_rss_polls_total', 'Total number of RSS polls', registry=_METRICS_REGISTRY)
NEWS_PARSE_ERRORS = Counter('news_parse_errors_total', 'Total number of parse errors (JSON/XML)', registry=_METRICS_REGISTRY)

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = int(os.getenv("RSS_POLL_INTERVAL", "60"))

async def _retry_http(method, *args, retries: int = 3, backoff_factor: float = 1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            NEWS_API_CALLS.inc()  # count each failed attempt
            try:
                logger.warning(f"HTTP call {method.__name__} failed attempt {attempt}/{retries}: {e}")
            except Exception:
                pass
            if attempt < retries:
                await asyncio.sleep(backoff_factor * 2 ** (attempt - 1))
    # Last attempt
    return await method(*args, **kwargs)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

async def fetch_news_api(query: str, limit: int) -> List[Dict[str, Any]]:
    NEWS_API_CALLS.inc()
    url = "https://cryptonews-api.com/api/v1/category"
    params = {
        "section": query,
        "items": limit,
        "token": NEWS_API_KEY,
    }
    async with httpx.AsyncClient() as client:
        resp = await _retry_http(client.get, url, params=params)
        resp.raise_for_status()
        try:
            data = resp.json().get("data", [])
        except Exception as e:
            NEWS_PARSE_ERRORS.inc()
            logger.error(f"JSON parse error in fetch_news_api: {e}")
            return []
    results: List[Dict[str, Any]] = []
    for art in data:
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

async def fetch_news_rss(feed_url: str, handle_update: callable, poll_interval: int = DEFAULT_POLL_INTERVAL):
    seen_ids = set()
    while True:
        NEWS_RSS_POLLS.inc()
        async with httpx.AsyncClient() as client:
            resp = await _retry_http(client.get, feed_url)
            resp.raise_for_status()
            content = resp.text
        try:
            feed = feedparser.parse(content)
        except Exception as e:
            NEWS_PARSE_ERRORS.inc()
            logger.error(f"RSS parse error: {e}")
            await asyncio.sleep(poll_interval)
            continue
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