import glob
import pyarrow.parquet as pq
import pandas as pd
from typing import Iterator

def iter_parquet_partitions(path_pattern: str) -> Iterator[pd.DataFrame]:
    """
    Yield each Parquet partition as a Pandas DataFrame, ordered by filename.

    Args:
        path_pattern: Glob pattern to match Parquet files (e.g.,
                      "data_lake/market/**/part-*.parquet").

    Yields:
        DataFrame for each partition file.
    """
    # Find all matching files, sorted for deterministic order
    files = sorted(glob.glob(path_pattern, recursive=True))
    for fn in files:
        # Read the Parquet file into a Pandas DataFrame
        table = pq.read_table(fn)
        df = table.to_pandas()
        yield df