import pandas as pd
import numpy as np


def compute_orderbook_imbalance(snapshot: pd.DataFrame) -> float:
    """
    Compute order book imbalance for a single snapshot.

    Parameters:
        snapshot (pd.DataFrame): DataFrame with columns ['price', 'amount', 'side'] for one timestamp.

    Returns:
        float: Imbalance ratio = (bid_volume - ask_volume) / (bid_volume + ask_volume).
    """
    bids = snapshot[snapshot['side'] == 'bid']
    asks = snapshot[snapshot['side'] == 'ask']
    bid_vol = bids['amount'].sum()
    ask_vol = asks['amount'].sum()
    total = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total if total != 0 else 0.0


def compute_orderbook_spread(snapshot: pd.DataFrame) -> float:
    """
    Compute spread for a single order book snapshot.

    Parameters:
        snapshot (pd.DataFrame): DataFrame with columns ['price', 'amount', 'side'].

    Returns:
        float: Spread = best ask price - best bid price.
    """
    bids = snapshot[snapshot['side'] == 'bid']
    asks = snapshot[snapshot['side'] == 'ask']
    if bids.empty or asks.empty:
        return 0.0
    best_bid = bids['price'].max()
    best_ask = asks['price'].min()
    return best_ask - best_bid


def compute_orderbook_depth(snapshot: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    Compute depth of order book for top N levels.

    Parameters:
        snapshot (pd.DataFrame): DataFrame with ['price','amount','side'].
        n_levels (int): Number of price levels to include on each side.

    Returns:
        pd.Series: Series with bid_depth_i and ask_depth_i keys for i=1..n_levels.
    """
    bids = snapshot[snapshot['side'] == 'bid'].sort_values('price', ascending=False).head(n_levels)
    asks = snapshot[snapshot['side'] == 'ask'].sort_values('price', ascending=True).head(n_levels)
    data = {}
    for i, vol in enumerate(bids['amount'].values, start=1):
        data[f'bid_depth_{i}'] = vol
    for i, vol in enumerate(asks['amount'].values, start=1):
        data[f'ask_depth_{i}'] = vol
    return pd.Series(data)


def compute_imbalance_series(orderbook_df: pd.DataFrame) -> pd.Series:
    """
    Compute time series of order book imbalance across snapshots.

    Parameters:
        orderbook_df (pd.DataFrame): DataFrame with columns ['ts', 'price', 'amount', 'side'].

    Returns:
        pd.Series: Imbalance ratio per timestamp = (bid_vol - ask_vol) / (bid_vol + ask_vol).
    """
    pivot = orderbook_df.pivot_table(
        index='ts', columns='side', values='amount', aggfunc='sum', fill_value=0
    )
    # Ensure both 'bid' and 'ask' columns exist
    if 'bid' not in pivot.columns:
        pivot['bid'] = 0
    if 'ask' not in pivot.columns:
        pivot['ask'] = 0
    bid_vol = pivot['bid']
    ask_vol = pivot['ask']
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total.replace(0, np.nan)
    return imbalance.fillna(0)


def compute_spread_series(orderbook_df: pd.DataFrame) -> pd.Series:
    """
    Compute time series of order book spread across snapshots.

    Parameters:
        orderbook_df (pd.DataFrame): DataFrame with columns ['ts', 'price', 'amount', 'side'].

    Returns:
        pd.Series: Spread per timestamp = best ask price - best bid price.
    """
    bids = orderbook_df[orderbook_df['side'] == 'bid']
    asks = orderbook_df[orderbook_df['side'] == 'ask']
    best_bid = bids.groupby('ts')['price'].max()
    best_ask = asks.groupby('ts')['price'].min()
    # Align indices
    spread = best_ask.reindex(best_bid.index).fillna(method='ffill') - best_bid
    return spread.fillna(0)


def compute_depth_series(orderbook_df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """
    Compute depth time series of order book for top N levels.

    Parameters:
        orderbook_df (pd.DataFrame): DataFrame with columns ['ts','price','amount','side'].
        n_levels (int): Number of price levels to include on each side.

    Returns:
        pd.DataFrame: Multi-index DataFrame indexed by 'ts' with columns
                      bid_depth_1..n and ask_depth_1..n.
    """
    # Group by timestamp and apply depth calculation
    depth_series = orderbook_df.groupby('ts').apply(
        lambda df: compute_orderbook_depth(df, n_levels)
    )
    depth_series.index = depth_series.index.droplevel(1)  # drop inner index
    return depth_series

