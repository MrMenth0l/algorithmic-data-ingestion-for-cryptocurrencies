from __future__ import annotations

import json
import math
import logging
from datetime import datetime, timezone
from typing import Optional, List, Any
from app.features.jobs.backfill import backfill_market_once, ttl_sweep_once
from fastapi import BackgroundTasks
from fastapi import Security
from fastapi.security import APIKeyHeader

import pandas as pd
import redis
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    Body,
    Query,
)
from fastapi.encoders import jsonable_encoder

from app.ingestion_service.config import settings
from .schemas import (
    MarketIngestRequest,
    OnchainIngestRequest,
    SocialIngestRequest,
    NewsIngestRequest,
    FeatureVector,
)
from app.adapters.ccxt_adapter import CCXTAdapter
from app.adapters.onchain_adapter import fetch_glassnode, fetch_covalent
from app.adapters.reddit_adapter import fetch_reddit_api, fetch_pushshift
from app.adapters.news_adapter import fetch_news_api, fetch_news_rss
from app.adapters.sentiment_adapter import fetch_twitter_sentiment
from .utils import write_to_parquet
from app.features.ingestion.ccxt_client import CCXTClient
from app.features.ingestion.news_client import NewsClient
from app.features.ingestion.onchain_client import OnchainClient
from app.features.ingestion.social_client import SocialClient
from app.features.store.redis_store import get_store, RedisFeatureStore
from app.features.factory.market_factory import build_market_features

logger = logging.getLogger(__name__)

router = APIRouter()

# Sync Redis client for simple health/demo endpoints (feature store is async)
_redis = redis.Redis(
    host=getattr(settings, "redis_host", "redis"),
    port=getattr(settings, "redis_port", 6379),
    db=getattr(settings, "redis_db", 0),
    password=getattr(settings, "redis_password", None) or None,
    decode_responses=True,
)

# ---------------------------
# Helpers
# ---------------------------

def _parse_epoch_sec(ts: Optional[int]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)

def _clean_numbers(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python, replace NaN/Inf with None."""
    try:
        import numpy as np
        np_f = (np.floating, np.integer)
    except Exception:
        np_f = tuple()

    if isinstance(obj, dict):
        return {k: _clean_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_numbers(v) for v in obj]
    if np_f and isinstance(obj, np_f):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def provide_news() -> NewsClient:
    """Local provider to avoid circular import from main."""
    return NewsClient()

# ---------------------------
# CCXT (GET) Historical
# ---------------------------

@router.get("/ccxt/{exchange}/historical")
async def get_ccxt_historical(
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100,
):
    """
    Fetch historical market data via CCXT client (async). Returns {rows, data}.
    """
    client = CCXTClient(exchange_name=exchange)
    try:
        df = await client.fetch_historical(symbol=symbol, timeframe=timeframe, limit=limit)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

# ---------------------------
# Market Ingest (POST)
# ---------------------------

@router.post("/market/{exchange}")
async def ingest_market(exchange: str, body: MarketIngestRequest):
    from ccxt.base.errors import BadSymbol  # local import to avoid hard dep at module import time
    adapter = CCXTAdapter(exchange)

    # Handle invalid market symbols explicitly (e.g., "BTC-USDT" vs "BTC/USDT")
    try:
        df = await adapter.fetch_ohlcv(
            body.symbol,
            body.granularity,
            since=None,
            limit=body.limit,
        )
    except BadSymbol as e:
        # Map CCXT BadSymbol -> 400 so clients can correct the request
        raise HTTPException(
            status_code=400,
            detail=f"{exchange} does not have market symbol {body.symbol}",
        ) from e

    # Ensure required/partition columns, and make timestamp UTC safely
    try:
        if not df.empty:
            df["symbol"] = body.symbol
            df["exchange"] = exchange

            if "timestamp" not in df.columns:
                raise ValueError("Missing columns: ['timestamp']")

            s = df["timestamp"]

            if pd.api.types.is_datetime64_any_dtype(s):
                # tz-aware -> convert; tz-naive -> localize
                if getattr(s.dt, "tz", None) is not None:
                    df["timestamp"] = s.dt.tz_convert("UTC")
                else:
                    df["timestamp"] = s.dt.tz_localize("UTC")
            else:
                # strings/ints/objects -> coerce to UTC-aware
                df["timestamp"] = pd.to_datetime(s, utc=True)

        base = settings.MARKET_PATH
        partitions = {
            "exchange": exchange,
            "symbol": body.symbol,
            "year": df["timestamp"].dt.year.iloc[0] if not df.empty else None,
            "month": df["timestamp"].dt.month.iloc[0] if not df.empty else None,
            "day": df["timestamp"].dt.day.iloc[0] if not df.empty else None,
        }

        path = write_to_parquet(df, base, partitions)

    except ValueError as ve:
        # Schema/normalization failure (Batch 1.2 enforcement)
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Anything else -> 500 with explicit message
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")

    # Write engineered features to Redis (Batch 2.2)
    features_written = 0
    if not df.empty:
        try:
            features_written = await _write_market_features_to_store(df)
        except Exception as e:
            logger.warning("Feature write failed", exc_info=e)
            features_written = 0

    if path is None:
        return {"status": "no_data", "path": None, "features_written": features_written}
    return {"status": "ok", "path": str(path), "features_written": features_written}
# ---------------------------
# Onchain Ingest (POST)
# ---------------------------

@router.post("/onchain/{source}")
async def ingest_onchain(source: str, body: OnchainIngestRequest):
    source_l = source.lower()
    if source_l == "glassnode":
        df = await fetch_glassnode(body.symbol, body.metric, days=body.days)
        base = settings.ONCHAIN_PATH + "/glassnode"
        partitions = {
            "symbol": body.symbol or "",
            "metric": body.metric or "",
            "year": df["timestamp"].dt.year.iloc[0] if not df.empty and "timestamp" in df.columns else None,
            "month": df["timestamp"].dt.month.iloc[0] if not df.empty and "timestamp" in df.columns else None,
            "day": df["timestamp"].dt.day.iloc[0] if not df.empty and "timestamp" in df.columns else None,
        }
    elif source_l == "covalent":
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
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")

    if path is None:
        return {"status": "no_data", "path": None}
    return {"status": "ok", "path": str(path)}

# ---------------------------
# Social Ingest (POST)
# ---------------------------

@router.post("/social/{platform}")
async def ingest_social(platform: str, body: SocialIngestRequest):
    platform_l = platform.lower()
    if platform_l == "twitter":
        df = await fetch_twitter_sentiment(body.query, body.since, body.until, body.max_results)
        base = settings.SOCIAL_PATH + "/twitter"
    elif platform_l == "reddit":
        df = await fetch_reddit_api(body.query, body.since, body.until, body.max_results)
        base = settings.SOCIAL_PATH + "/reddit"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown social platform: {platform}")

    # Light normalization passthrough (adapters mostly normalized already)
    if not df.empty:
        df["source"] = platform_l
        if "author" in df.columns and "user" not in df.columns:
            df["user"] = df["author"]
        if "content" in df.columns and "text" not in df.columns:
            df["text"] = df["content"]
        elif "selftext" in df.columns and "text" not in df.columns:
            df["text"] = df["selftext"]
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = None

    partitions = {
        "query": body.query.replace(" ", "_"),
        "year": df["ts"].dt.year.iloc[0] if not df.empty and "ts" in df.columns else None,
        "month": df["ts"].dt.month.iloc[0] if not df.empty and "ts" in df.columns else None,
        "day": df["ts"].dt.day.iloc[0] if not df.empty and "ts" in df.columns else None,
    }

    try:
        path = write_to_parquet(df, base, partitions)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")

    if path is None:
        return {"status": "no_data", "path": None}
    return {"status": "ok", "path": str(path)}

# ---------------------------
# News Ingest (POST)
# ---------------------------

@router.post("/news")
async def ingest_news(body: NewsIngestRequest):
    st = body.source_type.lower()
    if st == "api":
        df = await fetch_news_api(category=body.category)
        base = settings.NEWS_PATH + "/api"
        src = body.category or ""
    elif st == "rss":
        df = await fetch_news_rss(body.feed_url)
        base = settings.NEWS_PATH + "/rss"
        src = body.feed_url or ""
    else:
        raise HTTPException(status_code=400, detail=f"Unknown news source_type: {body.source_type}")

    partitions = {
        "source": src,
        "year": df["published_at"].dt.year.iloc[0] if not df.empty and "published_at" in df.columns else None,
        "month": df["published_at"].dt.month.iloc[0] if not df.empty and "published_at" in df.columns else None,
        "day": df["published_at"].dt.day.iloc[0] if not df.empty and "published_at" in df.columns else None,
    }

    try:
        path = write_to_parquet(df, base, partitions)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")

    if path is None:
        return {"status": "no_data", "path": None}
    return {"status": "ok", "path": str(path)}

# ---------------------------
# News Search (GET via NewsClient)
# ---------------------------

@router.get("/news/search")
async def get_news_articles(
    source: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
    news: NewsClient = Depends(provide_news),
):
    """
    Fetch news articles via NewsClient (one-shot search). Returns {rows, data}.
    since/until are epoch seconds (UTC).
    """
    df = await news.get_crypto_news(
        since=_parse_epoch_sec(since),
        until=_parse_epoch_sec(until),
        source=source,
        limit=limit,
    )
    rows = 0 if df is None else len(df)
    data = [] if df is None or df.empty else df.to_dict(orient="records")
    return {"rows": rows, "data": data}

# ---------------------------
# News RSS WebSocket stream
# ---------------------------

@router.websocket("/ws/news/rss")
async def ws_news_rss(websocket: WebSocket, feed_url: str):
    await websocket.accept()
    news = NewsClient()
    try:
        async def _on_item(item: dict):
            await websocket.send_json(item)
        await news.stream_rss(feed_url, handle_update=_on_item)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close()
    finally:
        await news.aclose()

# ---------------------------
# Onchain GET (clients)
# ---------------------------

@router.get("/onchain/glassnode")
async def get_glassnode_data(symbol: str, metric: str, days: int = 1):
    client = OnchainClient()
    try:
        df = await client.get_glassnode_metric(symbol=symbol, metric=metric, days=days)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

@router.get("/onchain/covalent")
async def get_covalent_balances(chain_id: int, address: str):
    client = OnchainClient()
    try:
        df = await client.get_covalent_balances(chain_id=chain_id, address=address)
        rows = 0 if df is None else len(df)
        data = [] if df is None or df.empty else df.to_dict(orient="records")
        return {"rows": rows, "data": data}
    finally:
        await client.aclose()

# ---------------------------
# Social GET (clients)
# ---------------------------

@router.get("/social/twitter")
async def get_twitter_data(
    query: str,
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 100,
):
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
    limit: int = 100,
):
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

# ---------------------------
# Redis health + simple demo feature endpoints (sync)
# ---------------------------

@router.get("/health/redis")
def redis_health_check():
    try:
        if _redis.ping():
            return {"redis": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    raise HTTPException(status_code=503, detail="Redis ping failed")

@router.post("/features/write")
def write_features(vec: FeatureVector = Body(...)):
    key = f"features:{vec.symbol}:{vec.timestamp}"
    payload = vec.json()
    _redis.set(key, payload, ex=getattr(settings, "feature_ttl_seconds", 3600))
    return {"status": "ok", "key": key}

@router.get("/features/read")
def read_features(symbol: str = Query(...), timestamp: int = Query(...)):
    key = f"features:{symbol}:{timestamp}"
    raw = _redis.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="Feature vector not found")
    vec = FeatureVector.parse_raw(raw)
    return vec

# ---------------------------
# Feature Retrieval (async, via RedisFeatureStore)
# ---------------------------

@router.get("/features/market")
async def get_market_features(
    symbol: str,
    timeframe: str,
    ts: List[int] = Query(..., description="Repeat per epoch-second"),
    store: RedisFeatureStore = Depends(get_store),
):
    """
    Retrieve market features from Redis by (symbol, timeframe, ts...).
    Returns JSON-safe payload with {rows, data}, where data = [{timestamp, ...payload}]
    """
    items = [("market", symbol, timeframe, t) for t in ts]
    vals = await store.batch_read(items)

    data: List[dict] = []
    for t, payload in zip(ts, vals):
        if payload is None:
            continue
        safe_payload = _clean_numbers(payload)
        data.append({"timestamp": int(t), **safe_payload})

    return {"rows": len(data), "data": data}

# ---------------------------
# Internal: build & write features
# ---------------------------

async def _write_market_features_to_store(df: pd.DataFrame) -> int:
    """
    Build features from normalized OHLCV and write them to Redis.
    Returns number of feature rows written.
    """
    if df.empty:
        return 0

    feats = build_market_features(df)
    if feats.empty:
        return 0

    # choose compact payload set; extend as needed
    payload_cols = [c for c in ["ret1", "rsi_14", "hl_spread", "oi_obv"] if c in feats.columns]

    items = []
    for _, r in feats.iterrows():
        payload = {c: r[c] for c in payload_cols if pd.notna(r[c])}
        if not payload:
            continue
        items.append({
            "domain": "market",
            "symbol": str(r["symbol"]),
            "timeframe": str(r["timeframe"]),
            "ts": r["timestamp"],  # tz-aware UTC
            "payload": payload,
        })

    if not items:
        return 0

    store = get_store()
    await store.batch_write(items)
    return len(items)

# --- Admin guard via header token ---
_admin_hdr = APIKeyHeader(name="X-Admin-Token", auto_error=False)

def require_admin(token: Optional[str] = Security(_admin_hdr)):
    expected = getattr(settings, "ADMIN_TOKEN", None)
    # Dev-friendly default: if no ADMIN_TOKEN is set, allow (but warn).
    # Set ADMIN_TOKEN in prod to enforce.
    if not expected:
        logger.warning("ADMIN_TOKEN not set; admin endpoints are not protected")
        return True
    if token == expected:
        return True
    raise HTTPException(status_code=401, detail="Admin token required")

@router.post("/admin/backfill/market", dependencies=[Depends(require_admin)])
async def admin_backfill_market(
    exchange: str = "binance",
    symbol: str = Query(..., description="e.g. BTC/USDT"),
    timeframe: str = Query(..., description="e.g. 1m"),
    lookback_minutes: int = Query(120, ge=1, le=7*24*60),
):
    """
    One-shot backfill for (exchange, symbol, timeframe) over the last N minutes.
    """
    result = await backfill_market_once(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        lookback_minutes=lookback_minutes,
    )
    return result


@router.post("/admin/features/ttl-sweep", dependencies=[Depends(require_admin)])
async def admin_ttl_sweep(
    pattern: str = Query("features:market:*"),
    ttl_default: Optional[int] = Query(None, description="Seconds to apply when no TTL is set"),
    max_keys: Optional[int] = Query(None, description="Stop after scanning this many keys"),
):
    """
    One-shot TTL sweep: ensures keys under pattern have expirations.
    """
    result = await ttl_sweep_once(pattern=pattern, ttl_default=ttl_default, max_keys=max_keys)
    return result