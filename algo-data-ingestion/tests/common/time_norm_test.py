import pandas as pd
import pytest
from app.common.time_norm import to_utc_dt, standardize_time_column, add_dt_partition, coerce_schema


def test_to_utc_dt_epoch_ms_vs_s():
    ms = pd.Series([1_700_000_000_000])  # milliseconds
    s = pd.Series([1_700_000_000])       # seconds
    dt_ms = to_utc_dt(ms)
    dt_s = to_utc_dt(s)
    assert str(dt_ms.dtype).endswith("UTC]")
    assert str(dt_s.dtype).endswith("UTC]")
    assert dt_ms.iloc[0] == dt_s.iloc[0]


def test_standardize_and_dt_partition():
    # two points within the same day
    df = pd.DataFrame({"ts": [1_700_000_000_000, 1_700_000_060_000]})
    df = standardize_time_column(df, candidates=["timestamp", "ts"], dest="timestamp")
    assert "timestamp" in df
    assert str(df["timestamp"].dtype).endswith("UTC]")
    add_dt_partition(df, ts_col="timestamp")
    assert "dt" in df
    # single-day enforcement
    assert df["dt"].nunique() == 1
    # dt matches date portion of timestamp
    assert df["dt"].iloc[0] == df["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]


def test_coerce_schema_adds_missing_columns_and_types():
    df = pd.DataFrame({"a": [1]})
    schema = {
        "a": "float64",
        "b": "string",
        "c": "Int64",
        "t": "datetime64[ns, UTC]",
    }
    out = coerce_schema(df, schema)
    # schema columns come first and exist
    assert list(out.columns)[:4] == ["a", "b", "c", "t"]
    assert str(out["b"].dtype) == "string"
    assert str(out["c"].dtype) == "Int64"
    assert str(out["t"].dtype).endswith("UTC]")


def test_to_utc_dt_parses_strings_to_utc():
    s = pd.Series(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"])  # ISO strings
    dt = to_utc_dt(s)
    assert str(dt.dtype).endswith("UTC]")
    # round-trip check: still UTC after tz_convert
    assert (dt.dt.tz_convert("UTC") == dt).all()