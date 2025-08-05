from fastapi import APIRouter, HTTPException
from app.ingestion_service.config import settings
from .schemas import MarketIngestRequest, OnchainIngestRequest, SocialIngestRequest, NewsIngestRequest
from app.adapters.ccxt_adapter import CCXTAdapter
from app.adapters.onchain_adapter import fetch_glassnode, fetch_covalent
from app.adapters.reddit_adapter import fetch_reddit_api, fetch_pushshift
from app.adapters.news_adapter import fetch_news_api, fetch_news_rss
from app.adapters.sentiment_adapter import fetch_twitter_sentiment
from .utils import write_to_parquet
from typing import Optional
from app.features.ingestion.ccxt_client import CCXTClient

router = APIRouter()


# Historical Market Data via CCXT client
@router.get("/ccxt/historical")
def get_ccxt_historical(symbol: str, since: Optional[int] = None, limit: int = 100):
    """
    Fetch historical market data via CCXT client.
    """
    client = CCXTClient(exchange_name="")  # uses default exchange if none specified
    return client.fetch_historical(symbol, since=since, limit=limit)


@router.post("/market/{exchange}")
async def ingest_market(exchange: str, body: MarketIngestRequest):
    adapter = CCXTAdapter(exchange)
    df = await adapter.fetch_ohlcv(body.symbol, body.granularity)
    base = settings.MARKET_PATH
    partitions = {
        "exchange": exchange,
        "symbol": body.symbol,
        "year": df["timestamp"].dt.year.iloc[0] if not df.empty and "timestamp" in df.columns else None,
        "month": df["timestamp"].dt.month.iloc[0] if not df.empty and "timestamp" in df.columns else None,
        "day": df["timestamp"].dt.day.iloc[0] if not df.empty and "timestamp" in df.columns else None,
    }
    try:
        path = write_to_parquet(df, base, partitions)
    except ValueError as ve:
        # Schema validation failure
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")
    if path is None:
        return {"status": "no_data", "path": None}
    return {"status": "ok", "path": str(path)}


# On-Chain Ingestion Endpoint
@router.post("/onchain/{source}")
async def ingest_onchain(source: str, body: OnchainIngestRequest):
    # Determine adapter based on source
    if source.lower() == "glassnode":
        df = await fetch_glassnode(body.symbol, body.metric, days=body.days)
        base = settings.ONCHAIN_PATH + "/glassnode"
        partitions = {
            "symbol": body.symbol or "",
            "metric": body.metric or "",
            "year": df["timestamp"].dt.year.iloc[0] if not df.empty and "timestamp" in df.columns else None,
            "month": df["timestamp"].dt.month.iloc[0] if not df.empty and "timestamp" in df.columns else None,
            "day": df["timestamp"].dt.day.iloc[0] if not df.empty and "timestamp" in df.columns else None,
        }
    elif source.lower() == "covalent":
        df = await fetch_covalent(chain_id=body.chain_id, address=body.address)
        base = settings.ONCHAIN_PATH + "/covalent"
        partitions = {
            "chain_id": body.chain_id,
            "address": body.address or "",
            "year": df["timestamp"].dt.year.iloc[0] if not df.empty and "timestamp" in df.columns else None,
            "month": df["timestamp"].dt.month.iloc[0] if not df.empty and "timestamp" in df.columns else None,
            "day": df["timestamp"].dt.day.iloc[0] if not df.empty and "timestamp" in df.columns else None,
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown onchain source: {source}")
    try:
        path = write_to_parquet(df, base, partitions)
    except ValueError as ve:
        # Schema validation failure
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")
    if path is None:
        return {"status": "no_data", "path": None}
    return {"status": "ok", "path": str(path)}


# Social Ingestion Endpoint
@router.post("/social/{platform}")
async def ingest_social(platform: str, body: SocialIngestRequest):
    platform_lower = platform.lower()
    if platform_lower == "twitter":
        df = await fetch_twitter_sentiment(body.query, body.since, body.until, body.max_results)
        base = settings.SOCIAL_PATH + "/twitter"
    elif platform_lower == "reddit":
        df = await fetch_reddit_api(body.query, body.since, body.until, body.max_results)
        base = settings.SOCIAL_PATH + "/reddit"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown social platform: {platform}")
    partitions = {
        "query": body.query.replace(" ", "_"),
        "year": df["ts"].dt.year.iloc[0] if not df.empty and "ts" in df.columns else None,
        "month": df["ts"].dt.month.iloc[0] if not df.empty and "ts" in df.columns else None,
        "day": df["ts"].dt.day.iloc[0] if not df.empty and "ts" in df.columns else None,
    }
    try:
        path = write_to_parquet(df, base, partitions)
    except ValueError as ve:
        # Schema validation failure
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")
    if path is None:
        return {"status": "no_data", "path": None}
    return {"status": "ok", "path": str(path)}


# News Ingestion Endpoint
@router.post("/news")
async def ingest_news(body: NewsIngestRequest):
    if body.source_type.lower() == "api":
        df = await fetch_news_api(category=body.category)
        base = settings.NEWS_PATH + "/api"
    elif body.source_type.lower() == "rss":
        df = await fetch_news_rss(body.feed_url)
        base = settings.NEWS_PATH + "/rss"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown news source_type: {body.source_type}")
    partitions = {
        "source": body.feed_url or body.category or "",
        "year": df["published_at"].dt.year.iloc[0] if not df.empty and "published_at" in df.columns else None,
        "month": df["published_at"].dt.month.iloc[0] if not df.empty and "published_at" in df.columns else None,
        "day": df["published_at"].dt.day.iloc[0] if not df.empty and "published_at" in df.columns else None,
    }
    try:
        path = write_to_parquet(df, base, partitions)
    except ValueError as ve:
        # Schema validation failure
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")
    if path is None:
        return {"status": "no_data", "path": None}
    return {"status": "ok", "path": str(path)}


# Legacy synchronous News fetch via NewsClient
from app.features.ingestion.news_client import NewsClient

@router.get("/news")
def get_news_articles(
    source: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100
):
    """
    Fetch news articles via NewsClient.
    """
    client = NewsClient()
    return client.get_crypto_news(since=since, until=until, source=source, limit=limit)