from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.features.factory.market_factory import build_market_features, FEATURE_VERSION
from app.features.store.redis_store import RedisFeatureStore, get_store

# Minimal payload fields to cache (expand as needed)
_PAYLOAD_KEEP = {
    "feature_version",
    "symbol", "exchange", "timeframe",
    "ret_1", "logret_1", "rvol_5", "rvol_20",
    "ema_12", "ema_26", "macd", "macd_signal_9",
    "rsi_14", "hl_spread", "oi_obv",
}

async def build_and_store_market_features(
    ohlcv: pd.DataFrame,
    *,
    store: Optional[RedisFeatureStore] = None,
    ttl: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build market features from normalized OHLCV and cache each row in Redis.
    Returns (features_df, list_of_keys_written).
    """
    feats = build_market_features(ohlcv)
    if feats.empty:
        return feats, []

    rs = store or get_store()

    # Prepare batch items for Redis
    items: List[Dict] = []
    for _, row in feats.iterrows():
        ts = row["timestamp"]                # tz-aware UTC
        symbol = str(row["symbol"])
        tf = str(row["timeframe"])

        payload = {k: row[k] for k in feats.columns if k in _PAYLOAD_KEEP}
        # Ensure version tag present
        payload.setdefault("feature_version", FEATURE_VERSION)

        items.append({
            "domain": "market",
            "symbol": symbol,
            "timeframe": tf,
            "ts": ts,
            "payload": {k: (v.item() if hasattr(v, "item") else v) for k, v in payload.items()},
            "ttl": ttl,
        })

    keys = await rs.batch_write(items)
    return feats, keys