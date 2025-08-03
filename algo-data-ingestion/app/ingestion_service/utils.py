import logging
import os
import time
import tempfile
import fsspec
import pandas as pd
from typing import Dict
from prometheus_client import Counter, Histogram, CollectorRegistry
from app.ingestion_service.parquet_schemas import MARKET_SCHEMA, ONCHAIN_SCHEMA, SOCIAL_SCHEMA, NEWS_SCHEMA

_METRICS_REGISTRY = CollectorRegistry()
PARQUET_WRITES_TOTAL = Counter(
    'parquet_writes_total',
    'Total successful Parquet writes',
    registry=_METRICS_REGISTRY
)
PARQUET_WRITE_ERRORS = Counter(
    'parquet_write_errors_total',
    'Total failed Parquet writes',
    registry=_METRICS_REGISTRY
)
PARQUET_WRITE_LATENCY = Histogram(
    'parquet_write_latency_seconds',
    'Parquet write latency in seconds',
    registry=_METRICS_REGISTRY
)
logger = logging.getLogger(__name__)


# Inline schema validator
def validate_schema(
    df: pd.DataFrame,
    schema: Dict[str, str],
    coerce: bool = False
) -> None:
    """
    Ensures df contains the specified columns with the exact dtypes.
    If coerce=True, attempts to cast columns to the expected dtype.
    Raises ValueError on missing columns or irreconcilable dtypes.
    """
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col, expected_dtype in schema.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            if coerce:
                try:
                    df[col] = df[col].astype(expected_dtype)
                except Exception:
                    raise ValueError(
                        f"Failed to coerce column '{col}' from {actual_dtype} to {expected_dtype}"
                    )
            else:
                raise ValueError(
                    f"Wrong dtype for column '{col}': actual={actual_dtype}, expected={expected_dtype}"
                )


def write_to_parquet(df, base_path, partitions, filename=None):
    if df.empty:
        logger.warning("Empty DataFrame, skipping Parquet write")
        return None
    # Schema validation before write
    if "market" in base_path:
        validate_schema(df, MARKET_SCHEMA, coerce=True)
    elif "onchain" in base_path:
        validate_schema(df, ONCHAIN_SCHEMA, coerce=True)
    elif "social" in base_path:
        validate_schema(df, SOCIAL_SCHEMA, coerce=True)
    elif "news" in base_path:
        validate_schema(df, NEWS_SCHEMA, coerce=True)
    # Resolve filesystem and root path
    fs, root = fsspec.core.url_to_fs(base_path)
    # Ensure fs is an instance (tests may return a class)
    if isinstance(fs, type):
        fs = fs(root)
    # Build directory path
    parts = [f"{k}={v}" for k, v in partitions.items()]
    dir_path = os.path.join(root, *parts)
    fs.makedirs(dir_path, exist_ok=True)
    # Generate filename
    fname = filename or f"part-{int(time.time() * 1000)}.parquet"
    full_path = os.path.join(dir_path, fname)
    temp_path = full_path + ".tmp"
    start = time.time()
    try:
        # Atomic write to temp file
        with fs.open(temp_path, 'wb') as f:
            df.to_parquet(f, compression="snappy", index=False)
        # Move temp to final path
        fs.mv(temp_path, full_path, rename_if_exists=True)
        duration = time.time() - start
        PARQUET_WRITES_TOTAL.inc()
        PARQUET_WRITE_LATENCY.observe(duration)
        return full_path
    except Exception as e:
        PARQUET_WRITE_ERRORS.inc()
        logger.error(f"Failed writing Parquet to {full_path}: {e}")
        raise

