from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence

from app.common.time_norm import standardize_time_column, add_dt_partition, coerce_schema

FEATURE_VERSION = "market.v1"

# Expected input (already enforced upstream, but we re-assert defensively)
REQUIRED_COLS: Sequence[str] = (
    "timestamp", "open", "high", "low", "close", "volume",
    "symbol", "exchange", "timeframe"
)

# Output schema (types are indicative; writer/validate_schema will enforce)
FEATURE_SCHEMA: Dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "dt": "string",
    "symbol": "string",
    "exchange": "string",
    "timeframe": "string",
    "feature_version": "string",

    # basic returns
    "ret_1": "float64",
    "logret_1": "float64",

    # rolling volatility (log-return std)
    "rvol_5": "float64",
    "rvol_20": "float64",

    # momentum
    "ema_12": "float64",
    "ema_26": "float64",
    "macd": "float64",
    "macd_signal_9": "float64",

    # RSI
    "rsi_14": "float64",

    # microstructure-ish
    "hl_spread": "float64",       # (high - low) / close
    "oi_obv": "float64",          # On-Balance Volume
}

# ---------------------------
# helpers (pure pandas / numpy)
# ---------------------------

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(period).mean()
    roll_down = pd.Series(down, index=close.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    # sign of price change times volume, cumulative
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume.fillna(0.0)).cumsum()

# ---------------------------
# public API
# ---------------------------

def build_market_features(
    df: pd.DataFrame,
    *,
    expect: Sequence[str] = REQUIRED_COLS,
    dropna_final: bool = True,
) -> pd.DataFrame:
    """
    Input: normalized OHLCV DataFrame (UTC, with symbol/exchange/timeframe).
    Output: feature rows aligned to 'timestamp', with 'dt' and 'feature_version'.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=list(FEATURE_SCHEMA.keys()))

    # Defensive: ensure we have timestamp and it's tz-aware UTC
    if "timestamp" not in df.columns:
        df = standardize_time_column(df, candidates=["ts", "time", "date"], dest="timestamp")
    # If timestamp exists but dtype is naive, leave to writer? Safer to coerce here:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Basic column presence
    missing = [c for c in expect if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by time and ensure one symbol/exchange/timeframe per frame (or per-row carry)
    df = df.sort_values("timestamp").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --- features ---
    ret_1 = close.pct_change()
    logret_1 = np.log(close.replace(0, np.nan)).diff()

    rvol_5 = logret_1.rolling(5).std()
    rvol_20 = logret_1.rolling(20).std()

    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    macd = ema_12 - ema_26
    macd_signal_9 = _ema(macd, 9)

    rsi_14 = _rsi(close, 14)

    hl_spread = (high - low) / close.replace(0, np.nan)
    oi_obv = _obv(close, volume)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "symbol": df["symbol"].astype("string"),
        "exchange": df["exchange"].astype("string"),
        "timeframe": df["timeframe"].astype("string"),
        "feature_version": FEATURE_VERSION,

        "ret_1": ret_1,
        "logret_1": logret_1,
        "rvol_5": rvol_5,
        "rvol_20": rvol_20,

        "ema_12": ema_12,
        "ema_26": ema_26,
        "macd": macd,
        "macd_signal_9": macd_signal_9,

        "rsi_14": rsi_14,
        "hl_spread": hl_spread,
        "oi_obv": oi_obv,
    })

    # Add dt partition (YYYY-MM-DD) from timestamp
    add_dt_partition(out, ts_col="timestamp")

    # Enforce output schema ordering/types (string Dtypes and tz-aware timestamp)
    out = out[list(FEATURE_SCHEMA.keys())]
    out = coerce_schema(out, FEATURE_SCHEMA)

    if dropna_final:
        # Drop rows that are all-NaN on features but keep rows where at least one feature exists
        feature_cols = [c for c in out.columns if c not in ("timestamp", "dt", "symbol", "exchange", "timeframe", "feature_version")]
        mask = out[feature_cols].notna().any(axis=1)
        out = out[mask].reset_index(drop=True)

    return out