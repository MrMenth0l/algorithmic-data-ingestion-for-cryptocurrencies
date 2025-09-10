#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import json
from typing import Optional, List
import pandas as pd
import httpx
import feedparser
from urllib.parse import urlparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.ingestion_service.config import settings
from app.ingestion_service.utils import write_to_parquet
from app.ingestion_service.parquet_schemas import NEWS_SCHEMA
from app.common.time_norm import standardize_time_column, add_dt_partition, coerce_schema


async def fetch_rss_once(feed_url: str, limit: int = 200) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        resp = await client.get(feed_url, timeout=15)
        resp.raise_for_status()
        content = resp.text
    feed = feedparser.parse(content)
    rows = []
    for i, entry in enumerate(feed.entries[:limit]):
        entry_id = entry.get("id") or entry.get("link") or f"idx-{i}"
        published_raw = entry.get("published") or entry.get("updated")
        rows.append({
            "id": entry_id,
            "title": entry.get("title"),
            "url": entry.get("link"),
            "source": urlparse(entry.get("link") or "").netloc,
            "author": None,
            "description": entry.get("summary"),
            "published_at": published_raw,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = standardize_time_column(df, candidates=["published_at", "date"], dest="published_at")
    df = coerce_schema(df, NEWS_SCHEMA)
    add_dt_partition(df, ts_col="published_at")
    return df


async def main_async(args) -> int:
    df = await fetch_rss_once(args.feed, args.limit)
    if df.empty:
        print("No entries fetched.")
        return 0
    base = settings.NEWS_PATH.rstrip("/") + "/rss"
    src = urlparse(args.feed).netloc.replace(":", "-")
    path = write_to_parquet(df, base, {"source": src})
    print(f"Wrote RSS parquet: {path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch RSS feed once and write to Parquet")
    ap.add_argument("--feed", required=True)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args(argv)
    import asyncio
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

