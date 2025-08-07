import pandas as pd
from typing import Optional
from app.features.processors.ta_indicators import compute_batch_indicators

class StatefulTAProcessor:
    """
    Stateful processor for computing batch TA indicators across Parquet partitions.
    Carries over the last (window - 1) rows to maintain rolling-window continuity.
    """

    def __init__(self, window: int = 20, constant: float = 0.015):
        self.window = window
        self.constant = constant
        # Buffer for carry-over rows: DataFrame of high, low, close
        self._carry: Optional[pd.DataFrame] = None

    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single chunk of raw data (with columns high, low, close).
        Returns a DataFrame of indicators (cci, roc) for the chunk, preserving original index.
        """
        # Prepare DataFrame for processing: prepend carry-over rows if any
        if self._carry is not None and not self._carry.empty:
            df = pd.concat([self._carry, chunk], axis=0)
        else:
            df = chunk

        # Compute batch indicators on the combined DataFrame
        feats = compute_batch_indicators(df, window=self.window, constant=self.constant)

        # Update carry-over buffer: keep last window raw rows
        self._carry = df[["high", "low", "close"]].iloc[-self.window:].copy()

        # Return only the rows corresponding to the current chunk
        return feats.loc[chunk.index]