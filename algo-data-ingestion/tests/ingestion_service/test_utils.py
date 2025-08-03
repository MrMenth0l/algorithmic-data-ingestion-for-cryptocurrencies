import pandas as pd
import os
from types import SimpleNamespace
import pytest
from app.ingestion_service.utils import write_to_parquet, PARQUET_WRITES_TOTAL, PARQUET_WRITE_ERRORS, PARQUET_WRITE_LATENCY


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