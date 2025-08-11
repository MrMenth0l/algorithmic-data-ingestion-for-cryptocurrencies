from __future__ import annotations
"""Time & DataFrame normalization helpers for UTC + schema coercion.

Usage patterns in adapters:

    from time_norm import (
        standardize_time_column,
        to_utc_dt,
        add_dt_partition,
        coerce_schema,
    )

    df = standardize_time_column(df, candidates=["timestamp", "ts", "published_at"], dest="timestamp")
    df = coerce_schema(df, {
        "timestamp": "datetime64[ns, UTC]",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "symbol": "string",
        "exchange": "string",
    })
    add_dt_partition(df, ts_col="timestamp")

Notes:
- All datetime outputs are tz-aware UTC.
- Coercion prefers Pandas nullable dtypes where appropriate.
"""

from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_integer_dtype,
)

# ------------------------------------------------------------
# Datetime helpers
# ------------------------------------------------------------

UTC_DTYPE = pd.DatetimeTZDtype(tz="UTC")


def _detect_epoch_unit(values: pd.Series) -> str:
    """Heuristically determine epoch unit from magnitude.
    Returns 'ms' if values look like milliseconds, else 's'.
    """
    s = pd.to_numeric(values, errors="coerce")
    if s.empty:
        return "s"
    # use finite values only
    s = s[np.isfinite(s)]
    if s.empty:
        return "s"
    q = s.astype("float64").quantile(0.5)
    # thresholds: ~2001-09-09 in seconds (1e9) / in ms (1e12)
    return "ms" if q >= 1e12 else "s"


def to_utc_dt(values: pd.Series, *, unit: Optional[str] = None) -> pd.Series:
    """Coerce a Series to tz-aware UTC datetimes.

    - Datetime w/ tz: converted to UTC.
    - Naive datetime: localized to UTC (assumes values are already UTC).
    - Integers: parsed as epoch (auto-detects seconds vs ms unless unit provided).
    - Strings/objects: parsed with `utc=True`.
    """
    if is_datetime64tz_dtype(values):
        return values.dt.tz_convert("UTC")
    if is_datetime64_any_dtype(values) and not is_datetime64tz_dtype(values):
        return values.dt.tz_localize("UTC")
    if is_integer_dtype(values):
        u = unit or _detect_epoch_unit(values)
        return pd.to_datetime(values, unit=u, utc=True)
    # strings / objects
    s = pd.to_datetime(values, utc=True, errors="coerce")
    return s


def standardize_time_column(df: pd.DataFrame, *, candidates: Iterable[str], dest: str = "timestamp", unit: Optional[str] = None) -> pd.DataFrame:
    """Find first present time column from `candidates`, make it tz-aware UTC, rename to `dest`.
    If none found, creates an empty dest column with UTC dtype.
    """
    for c in candidates:
        if c in df.columns:
            df = df.copy()
            df[dest] = to_utc_dt(df[c], unit=unit)
            if c != dest:
                df.drop(columns=[c], inplace=True)
            return df
    # none found → create empty column
    df = df.copy()
    df[dest] = pd.Series(pd.NaT, index=df.index, dtype=UTC_DTYPE)
    return df


def add_dt_partition(df: pd.DataFrame, *, ts_col: str = "timestamp", out_col: str = "dt") -> None:
    """Add `dt=YYYY-MM-DD` partition column in-place from a tz-aware timestamp column."""
    if ts_col not in df.columns:
        df[out_col] = pd.Series(dtype="string")
        return
    ts = df[ts_col]
    if not is_datetime64tz_dtype(ts):
        ts = to_utc_dt(ts)
    df[out_col] = ts.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")


# ------------------------------------------------------------
# Schema coercion
# ------------------------------------------------------------

NULLABLE_INT = "Int64"  # pandas nullable int
STRING_DTYPE = "string"  # use pandas string; backends can map to pyarrow later


def _coerce_one(series: pd.Series, dtype: str) -> pd.Series:
    if dtype.startswith("datetime64"):
        # Expect tz-aware UTC
        return to_utc_dt(series)
    if dtype in ("float64", "float32"):
        return pd.to_numeric(series, errors="coerce").astype(dtype)
    if dtype in ("int64", "int32"):
        # Prefer nullable ints to preserve NaNs
        return pd.to_numeric(series, errors="coerce").astype(NULLABLE_INT)
    if dtype.lower().startswith("int") or dtype.startswith("Int"):
        return pd.to_numeric(series, errors="coerce").astype(NULLABLE_INT)
    if dtype.startswith("string"):
        return series.astype(STRING_DTYPE)
    # Fallback: try astype, allow failure to surface
    return series.astype(dtype)


def coerce_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Return a new DataFrame with columns coerced to the provided schema.

    - Missing columns are created with NA values of the right dtype.
    - Extra columns are preserved (reordering at the end keeps schema columns first).
    - Datetime columns become tz-aware UTC.
    """
    df = df.copy()
    for col, dtype in schema.items():
        if col in df.columns:
            df[col] = _coerce_one(df[col], dtype)
        else:
            # Create empty column with the desired dtype
            if dtype.startswith("datetime64"):
                df[col] = pd.Series(pd.NaT, index=df.index, dtype=UTC_DTYPE)
            elif dtype in ("float64", "float32"):
                df[col] = pd.Series(np.nan, index=df.index, dtype=dtype)
            elif dtype.lower().startswith("int") or dtype.startswith("Int"):
                df[col] = pd.Series(pd.NA, index=df.index, dtype=NULLABLE_INT)
            elif dtype.startswith("string"):
                df[col] = pd.Series(pd.NA, index=df.index, dtype=STRING_DTYPE)
            else:
                df[col] = pd.Series(pd.NA, index=df.index)
    # Order: schema columns first, keep any extras afterward
    ordered = [c for c in schema.keys() if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


__all__ = [
    "to_utc_dt",
    "standardize_time_column",
    "add_dt_partition",
    "coerce_schema",
]
