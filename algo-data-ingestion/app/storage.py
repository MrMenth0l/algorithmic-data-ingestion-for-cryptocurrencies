#Legacy file, not in actual implementation.

import os, json
import pandas as pd
from datetime import datetime

BASE_PATH = os.getenv("DATA_LAKE_PATH", "./data_lake")

async def persist_raw(source: str, symbol: str, raw: dict):
    # partition by date and source
    date = datetime.utcfromtimestamp(raw["timestamp"] / 1000).strftime("%Y-%m-%d")
    path = os.path.join(BASE_PATH, source, symbol.replace("/", "_"), f"date={date}")
    os.makedirs(path, exist_ok=True)

    # write one-row Parquet
    df = pd.json_normalize(raw)
    file_path = os.path.join(path, f"{int(raw['timestamp'])}.parquet")
    df.to_parquet(file_path, index=False)