from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from datetime import datetime, timezone
import redis
from app.ingestion_service.config import settings
from .schemas import MarketIngestRequest, OnchainIngestRequest, SocialIngestRequest, NewsIngestRequest
from .schemas import FeatureVector
from fastapi import Body, Query
from app.adapters.ccxt_adapter import CCXTAdapter
from app.adapters.onchain_adapter import fetch_glassnode, fetch_covalent
from app.adapters.reddit_adapter import fetch_reddit_api, fetch_pushshift
from app.adapters.news_adapter import fetch_news_api, fetch_news_rss
from app.adapters.sentiment_adapter import fetch_twitter_sentiment
from .utils import write_to_parquet
from typing import Optional
from app.features.ingestion.ccxt_client import CCXTClient
from app.features.ingestion.news_client import NewsClient
from app.ingestion_service.main import get_news
from app.features.ingestion.onchain_client import OnchainClient
from app.features.ingestion.social_client import SocialClient



router = APIRouter()

# Redis client for feature store health checks
_redis = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password or None,
    decode_responses=True,
)




# Historical Market Data via CCXT client for a specific exchange
@router.get("/ccxt/{exchange}/historical")
async def get_ccxt_historical(
    exchange: str,
    symbol: str,
    limit: int = 100
):
    """
    Fetch historical market data via CCXT client for a specific exchange.
    """
    client = CCXTClient(exchange_name=exchange)
    try:
        return await client.fetch_historical(symbol=symbol, limit=limit)
    finally:
        await client.aclose()


@router.post("/market/{exchange}")
async def ingest_market(exchange: str, body: MarketIngestRequest):
    adapter = CCXTAdapter(exchange)
    df = await adapter.fetch_ohlcv(
        body.symbol,
        body.granularity,
        since=None,
        limit=body.limit
    )
    # Ensure partition columns exist for parquet write
    if not df.empty:
        df['symbol'] = body.symbol
        df['exchange'] = exchange
    base = settings.MARKET_PATH
    partitions = {
        "exchange": exchange,
        "symbol": body.symbol,
        "year": df["timestamp"].dt.year.iloc[0] if not df.empty and "timestamp" in df.columns else None,
        "month": df["timestamp"].dt.month.iloc[0] if not df.empty and "timestamp" in df.columns else None,
        "day": df["timestamp"].dt.day.iloc[0] if not df.empty and "timestamp" in df.columns else None,
    }
    if not df.empty:
        df['symbol'] = body.symbol
        df['exchange'] = exchange
        # Localize timestamp to UTC for Parquet schema compliance
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
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
    # Normalize columns for social data
    if not df.empty:
        # Source platform
        df['source'] = platform_lower
        # User and text fields
        if 'author' in df.columns:
            df['user'] = df['author']
        if 'content' in df.columns:
            df['text'] = df['content']
        elif 'selftext' in df.columns:
            df['text'] = df['selftext']
        # Sentiment score default (if missing)
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = None
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


def _parse_epoch_sec(ts: Optional[int]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)

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


# News search (async) via NewsClient
@router.get("/news/search")
async def get_news_articles(
    source: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
    news: NewsClient = Depends(get_news),
):
    """
    Fetch news articles via NewsClient (one-shot search). Returns records JSON.
    - `since`/`until` are epoch seconds (UTC).
    - `source` selects provider/channel handled by the adapter layer.
    """
    df = await news.get_crypto_news(
        since=_parse_epoch_sec(since),
        until=_parse_epoch_sec(until),
        source=source,
        limit=limit,
    )
    return {"rows": 0 if df is None else len(df),
            "data": [] if df is None else df.to_dict(orient="records")}

# Optional: WebSocket stream of RSS items via NewsClient
@router.websocket("/ws/news/rss")
async def ws_news_rss(websocket: WebSocket, feed_url: str):
    """
    Live-stream RSS items to clients over WebSocket.
    Sends each item as JSON as it arrives.
    """
    await websocket.accept()
    news = NewsClient()  # local instance because DI isn't used in WS context
    try:
        async def _on_item(item: dict):
            await websocket.send_json(item)
        await news.stream_rss(feed_url, handle_update=_on_item)
    except WebSocketDisconnect:
        # client disconnected
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close()
    finally:
        await news.aclose()


# OnchainClient GET endpoints
@router.get("/onchain/glassnode")
async def get_glassnode_data(
    symbol: str,
    metric: str,
    days: int = 1
):
    """
    Fetch Glassnode metric data via OnchainClient (async) and return JSON.
    """
    client = OnchainClient()
    try:
        df = await client.get_glassnode_metric(symbol=symbol, metric=metric, days=days)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()



@router.get("/onchain/covalent")
async def get_covalent_balances(
    chain_id: int,
    address: str
):
    """
    Fetch Covalent token balances via OnchainClient (async) and return JSON.
    """
    client = OnchainClient()
    try:
        df = await client.get_covalent_balances(chain_id=chain_id, address=address)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

# SocialClient GET endpoints
@router.get("/social/twitter")
async def get_twitter_data(
    query: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100
):
    """
    Fetch tweets via SocialClient (async) and return JSON.
    since/until are epoch seconds (UTC).
    """
    client = SocialClient()
    try:
        df = await client.fetch_tweets(
            query,
            since=_parse_epoch_sec(since) if since is not None else None,
            until=_parse_epoch_sec(until) if until is not None else None,
            limit=limit,
        )
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

@router.get("/social/reddit")
async def get_reddit_data(
    subreddit: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100
):
    """
    Fetch Reddit posts via SocialClient (async) and return JSON.
    since/until are epoch seconds (UTC).
    """
    client = SocialClient()
    try:
        df = await client.fetch_reddit_api(
            subreddit,
            since=_parse_epoch_sec(since) if since is not None else None,
            until=_parse_epoch_sec(until) if until is not None else None,
            limit=limit,
        )
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()


# Redis health-check endpoint
@router.get("/health/redis")
def redis_health_check():
    """
    Check connectivity to Redis.
    """
    try:
        if _redis.ping():
            return {"redis": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    raise HTTPException(status_code=503, detail="Redis ping failed")


# Feature store write endpoint
@router.post("/features/write")
def write_features(vec: FeatureVector = Body(...)):
    """
    Write a feature vector to Redis with TTL.
    """
    key = f"features:{vec.symbol}:{vec.timestamp}"
    payload = vec.json()
    _redis.set(key, payload, ex=settings.feature_ttl_seconds)
    return {"status": "ok", "key": key}

# Feature store read endpoint
@router.get("/features/read")
def read_features(
    symbol: str = Query(...),
    timestamp: int = Query(...)
):
    """
    Read a feature vector from Redis.
    """
    key = f"features:{symbol}:{timestamp}"
    raw = _redis.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="Feature vector not found")
    vec = FeatureVector.parse_raw(raw)
    return vec