import pandas as pd
import pytest
from pathlib import Path

from app.features.processors.chunk_loader import iter_parquet_partitions

@pytest.fixture
def tmp_parquet_files(tmp_path):
    # Create two small DataFrames and write them as Parquet files in tmp_path
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"a": [4, 5],    "b": ["u", "v"]})
    file1 = tmp_path / "part-000.parquet"
    file2 = tmp_path / "part-001.parquet"
    df1.to_parquet(file1)
    df2.to_parquet(file2)
    return df1, df2, tmp_path

def test_iter_parquet_partitions_yields_dataframes_in_order(tmp_parquet_files):
    df1, df2, tmp_path = tmp_parquet_files
    pattern = str(tmp_path / "part-*.parquet")
    # Collect the yielded DataFrames
    dfs = list(iter_parquet_partitions(pattern))
    # We expect two DataFrames in order
    assert len(dfs) == 2
    pd.testing.assert_frame_equal(dfs[0].reset_index(drop=True), df1.reset_index(drop=True))
    pd.testing.assert_frame_equal(dfs[1].reset_index(drop=True), df2.reset_index(drop=True))

def test_iter_parquet_partitions_empty_pattern(tmp_path):
    # No files match the pattern
    pattern = str(tmp_path / "nomatch-*.parquet")
    dfs = list(iter_parquet_partitions(pattern))
    assert dfs == []