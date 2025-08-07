import pandas as pd
import numpy as np
import pytest

from app.features.processors.stateful_processor import StatefulTAProcessor
from app.features.processors.ta_indicators import compute_batch_indicators

@pytest.fixture
def synthetic_df():
    # Create a synthetic time series of 30 days
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    price = np.arange(1, 31, dtype=float)
    high = price + 1
    low = price - 1
    close = price
    df = pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)
    return df

def test_process_chunk_full_match(synthetic_df):
    # Test that processing in one chunk equals processing in two sequential chunks
    window = 5
    constant = 0.015

    # Full processing
    full_feats = compute_batch_indicators(synthetic_df, window=window, constant=constant)

    # Split into two chunks
    chunk1 = synthetic_df.iloc[:20]
    chunk2 = synthetic_df.iloc[20:]

    processor = StatefulTAProcessor(window=window, constant=constant)
    out1 = processor.process_chunk(chunk1)
    out2 = processor.process_chunk(chunk2)

    # Combine outputs and compare to full
    combined = pd.concat([out1, out2])
    pd.testing.assert_frame_equal(combined, full_feats)

def test_carry_over_usage(synthetic_df):
    # Ensure carry-over buffer grows to window-1 rows and is used
    window = 7
    constant = 0.015

    processor = StatefulTAProcessor(window=window, constant=constant)
    # Before any processing, _carry should be None
    assert processor._carry is None

    # Process first chunk
    chunk1 = synthetic_df.iloc[:10]
    _ = processor.process_chunk(chunk1)
    # After first, carry should be last (window-1) rows of chunk1
    expected_carry = chunk1.iloc[-window:][["high","low","close"]]
    pd.testing.assert_frame_equal(processor._carry, expected_carry)

    # Process second chunk and ensure no error
    chunk2 = synthetic_df.iloc[10:15]
    out2 = processor.process_chunk(chunk2)
    # Output should have same index as chunk2
    assert list(out2.index) == list(chunk2.index)