from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Query
from app.features.ingestion.ccxt_client import CCXTClient
from typing import Optional
from .adapters.ccxt_adapter import get_ticker_raw
from .models import Envelope, TickerData, SentimentEnvelope, SentimentItem
from .storage import persist_raw
from .adapters.sentiment_adapter import fetch_twitter_sentiment
from .adapters.reddit_adapter import fetch_reddit
from datetime import datetime, timedelta
from .adapters.news_adapter import fetch_news_api, fetch_news_rss
from .models import NewsEnvelope, NewsItem
from .adapters.onchain_adapter import fetch_glassnode, fetch_covalent
from .models import OnchainEnvelope, OnchainItem


app = FastAPI(
    title="Algo Data Ingestion",
    version="0.1.0",
    description="Unified API for market, on-chain, sentiment & news data"
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/v1/market/ticker", response_model=Envelope)
async def get_ticker(symbol: str = Query(..., description="Pair, e.g. BTC/USDT")):
    try:
        raw = await get_ticker_raw(symbol)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    ticker = TickerData(
        symbol=raw["symbol"],
        bid=raw["bid"],
        ask=raw["ask"],
        last=raw["last"],
        timestamp=raw["timestamp"],
    )

    try:
        await persist_raw("market", symbol, raw)
    except Exception as e:
        print(f"[WARN] failed to persist raw data: {e}")
    return Envelope(data=ticker, metadata={"fetched_at": ticker.timestamp})

@app.get("/v1/sentiment/twitter", response_model=SentimentEnvelope)
async def get_twitter_sentiment(
    q: str = Query(..., description="Search query, e.g. BTC OR Ethereum"),
    limit: int = Query(10, ge=1, le=100),
):
    try:
        now = datetime.utcnow()
        since = now - timedelta(days=1)
        df = await fetch_twitter_sentiment(q, since, now, limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # (Optional) You can run an NLP sentiment model here to fill sentiment_score
    items = [
        SentimentItem(
            source="twitter",
            id=rec.get("id", ""),
            text=rec["text"],
            user=rec["user"],
            created_at=rec["ts"],
            sentiment_score=rec["sentiment_score"]
        )
        for rec in df.to_dict(orient="records")
    ]
    return SentimentEnvelope(data=items)

@app.get("/v1/sentiment/reddit", response_model=SentimentEnvelope)
async def get_reddit_sentiment(
    q: str = Query(..., description="Subreddit name, e.g. cryptocurrency"),
    limit: int = Query(10, ge=1, le=100),
    mode: str = Query("api", regex="^(api|pushshift)$", description="Mode switch: 'api' or 'pushshift'")
):
    now = datetime.utcnow()
    since = now - timedelta(days=1)
    try:
        df = await fetch_reddit(q, since, now, limit, source=mode)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    items = [
        SentimentItem(
            source="reddit",
            id=rec["id"],
            text=rec.get("title","") or rec.get("selftext",""),
            user=rec["author"],
            created_at=rec["ts"],
            sentiment_score=0.0
        )
        for rec in df.to_dict(orient="records")
    ]

    # Optionally persist raw payload
    try:
        await persist_raw("sentiment", f"reddit_{mode}", df.to_dict(orient="records"))
    except Exception as e:
        print(f"[WARN] failed to persist raw data: {e}")
    return SentimentEnvelope(data=items, metadata={"mode": mode, "fetched_at": datetime.utcnow()})

@app.get("/v1/news/crypto", response_model=NewsEnvelope)
async def get_crypto_news(
    section: str = Query("general", description="News section, e.g. general, exchange"),
    limit: int = Query(10, ge=1, le=50)
):
    try:
        raw = await fetch_news_api(section, limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    items = [
        NewsItem(
            id=art["id"],
            title=art["title"],
            url=art["url"],
            source=art["source"],
            author=art.get("author"),
            description=art.get("description"),
            published_at=art["published_at"],
        )
        for art in raw
    ]

    try:
        await persist_raw("news", f"crypto_{section}", raw)
    except Exception as e:
        print(f"[WARN] failed to persist raw data: {e}")
    return NewsEnvelope(data=items, metadata={"section": section, "fetched_at": datetime.utcnow()})

@app.get("/v1/news/rss", response_model=NewsEnvelope)
async def get_news_rss(
    feed_url: str = Query(..., description="RSS feed URL, e.g. https://example.com/rss")
):
    try:
        raw = await fetch_news_rss(feed_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    items = [
        NewsItem(
            id=art.get("id", ""),
            title=art.get("title", ""),
            url=art.get("url", ""),
            source=art.get("source", ""),
            author=art.get("author"),
            description=art.get("description"),
            published_at=art.get("published_at"),
        )
        for art in raw
    ]

    try:
        await persist_raw("news", f"rss_{feed_url}", raw)
    except Exception as e:
        print(f"[WARN] failed to persist raw RSS data: {e}")

    return NewsEnvelope(data=items, metadata={"feed_url": feed_url, "fetched_at": datetime.utcnow()})

@app.get("/v1/onchain", response_model=OnchainEnvelope)
async def get_onchain(
    source: str = Query("glassnode", regex="^(glassnode|covalent)$"),
    symbol: str = Query("BTC", description="Asset symbol or chain ID/address"),
    metric: str = Query("exchange_netflow", description="Glassnode metric name"),
    days: int = Query(1, ge=1, le=30),
    chain_id: int = Query(1, description="Covalent chain ID (e.g. 1 for Ethereum)"),
    address: str = Query(None, description="Covalent address (for covalent mode)")
):
    try:
        if source == "glassnode":
            raw = await fetch_glassnode(symbol, metric, days)
        else:
            raw = await fetch_covalent(chain_id, address)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    items = [OnchainItem(**pt) for pt in raw]
    try:
        await persist_raw("onchain", source, raw)
    except Exception as e:
        print(f"[WARN] failed to persist raw data: {e}")
    return OnchainEnvelope(data=items, metadata={"source": source, "fetched_at": datetime.utcnow()})