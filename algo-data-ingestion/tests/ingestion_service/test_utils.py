import pandas as pd
import os
from types import SimpleNamespace
import pytest
from app.ingestion_service.utils import write_to_parquet, PARQUET_WRITES_TOTAL, PARQUET_WRITE_ERRORS, PARQUET_WRITE_LATENCY
from app.ingestion_service.parquet_schemas import MARKET_SCHEMA
from app.ingestion_service.utils import validate_schema


def test_write_to_parquet_skips_empty(tmp_path, caplog, monkeypatch):
    # Empty DataFrame should return None and log info
    df = pd.DataFrame()
    result = write_to_parquet(df, str(tmp_path), {"year": 2025})
    assert result is None
    assert "Empty DataFrame, skipping Parquet write" in caplog.text


def test_write_to_parquet_success(tmp_path, monkeypatch):
    # Prepare DataFrame
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    # Monkeypatch metrics to count calls
    writes = {"n": 0}
    latency = {"obs": []}
    monkeypatch.setattr('app.ingestion_service.utils.PARQUET_WRITES_TOTAL', SimpleNamespace(inc=lambda: writes.__setitem__('n', writes.get('n', 0) + 1)))
    monkeypatch.setattr('app.ingestion_service.utils.PARQUET_WRITE_LATENCY', SimpleNamespace(observe=lambda x: latency['obs'].append(x)))
    # Invoke write
    base = str(tmp_path / "base")
    partitions = {"year": 2025, "month": 8}
    path = write_to_parquet(df, base, partitions)
    # File should exist
    assert os.path.exists(path)
    # Path includes partitions
    assert "year=2025" in path and "month=8" in path
    # Metrics should have been called
    assert writes["n"] == 1
    assert len(latency["obs"]) == 1


def test_write_to_parquet_error(tmp_path, monkeypatch):
    # Prepare DataFrame
    df = pd.DataFrame({"a": [1], "b": ["z"]})
    # Make fs.open raise
    import fsspec
    class BadFS:
        def __init__(self, root): pass
        def makedirs(self, path, exist_ok): pass
        def open(self, path, mode):
            raise IOError("write error")
        def mv(self, src, dst, rename_if_exists=True): pass
    monkeypatch.setattr('fsspec.core.url_to_fs', lambda url: (BadFS, url))
    # Monkeypatch error counter
    errors = {"n": 0}
    monkeypatch.setattr('app.ingestion_service.utils.PARQUET_WRITE_ERRORS', SimpleNamespace(inc=lambda: errors.__setitem__('n', errors.get('n', 0) + 1)))
    # Expect exception
    with pytest.raises(IOError):
        write_to_parquet(df, str(tmp_path), {"year": 2025})
    assert errors["n"] == 1


# Schema validation tests
def test_write_to_parquet_schema_missing_columns(tmp_path):
    df = pd.DataFrame({"symbol": ["BTC"], "open": [1.0], "high": [2.0]})
    with pytest.raises(ValueError) as exc:
        validate_schema(df, MARKET_SCHEMA, coerce=False)
    msg = str(exc.value)
    assert "Missing columns" in msg
    assert "timestamp" in msg


def test_write_to_parquet_schema_wrong_dtype(tmp_path):
    df = pd.DataFrame({
        "timestamp": ["2025-08-01T00:00:00Z"],
        "symbol": ["BTC-USDT"],
        "open": [1.0],
        "high": [2.0],
        "low": [0.5],
        "close": [1.5],
        "volume": [100.0]
    })
    with pytest.raises(ValueError) as exc:
        validate_schema(df, MARKET_SCHEMA, coerce=False)
    assert "Wrong dtype for column 'timestamp'" in str(exc.value)


def test_write_to_parquet_schema_coerce_int_to_float(tmp_path):
    # DataFrame with volume as int (should coerce to float64)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-08-01T00:00:00Z"], utc=True),
        "symbol": ["BTC-USDT"],
        "open": [1],   # int, will coerce to float
        "high": [2],   # int
        "low": [1],    # int
        "close": [1],  # int
        "volume": [100]# int
    })
    # Should succeed and write file
    path = write_to_parquet(df, str(tmp_path), {"exchange": "binance", "symbol": "BTC-USDT"})
    assert os.path.exists(path)