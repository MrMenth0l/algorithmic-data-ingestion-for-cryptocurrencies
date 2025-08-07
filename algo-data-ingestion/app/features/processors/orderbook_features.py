import pandas as pd
import numpy as np
from numba import njit


# Numba-accelerated imbalance kernel
@njit
def _imbalance_nb(bid_array: np.ndarray, ask_array: np.ndarray) -> np.ndarray:
    n = bid_array.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        total = bid_array[i] + ask_array[i]
        out[i] = (bid_array[i] - ask_array[i]) / total if total != 0 else 0.0
    return out


# Numba-accelerated batch orderbook kernel
@njit
def _batch_orderbook_nb(
    bid_vol: np.ndarray,
    ask_vol: np.ndarray,
    bid_price: np.ndarray,
    ask_price: np.ndarray
) -> np.ndarray:
    n = bid_vol.shape[0]
    imbalance = np.empty(n, dtype=np.float64)
    spread = np.empty(n, dtype=np.float64)
    for i in range(n):
        total = bid_vol[i] + ask_vol[i]
        imbalance[i] = (bid_vol[i] - ask_vol[i]) / total if total != 0 else 0.0
        spread[i] = ask_price[i] - bid_price[i]
    # Stack to shape (n, 2)
    return np.vstack((imbalance, spread)).T


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
    # Use Numba-accelerated kernel for imbalance
    bid_array = bid_vol.values.astype(np.float64)
    ask_array = ask_vol.values.astype(np.float64)
    imbalance_array = _imbalance_nb(bid_array, ask_array)
    return pd.Series(imbalance_array, index=pivot.index)


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




# Batch computation of orderbook features (imbalance and spread)
def compute_batch_orderbook(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute batch orderbook features (imbalance and spread) in one pass.
    Returns a DataFrame with columns 'imbalance' and 'spread', indexed by 'ts'.
    """
    # Pivot to per-timestamp bid/ask volumes and prices
    pivot_vol = orderbook_df.pivot_table(
        index='ts', columns='side', values='amount', aggfunc='sum', fill_value=0
    )
    pivot_price = orderbook_df.pivot_table(
        index='ts', columns='side', values='price', aggfunc='first'
    ).fillna(method='ffill')
    # Ensure both columns exist
    for col in ('bid', 'ask'):
        if col not in pivot_vol.columns:
            pivot_vol[col] = 0
        if col not in pivot_price.columns:
            pivot_price[col] = pivot_price[col].ffill().fillna(0)
    bid_vol = pivot_vol['bid'].values.astype(np.float64)
    ask_vol = pivot_vol['ask'].values.astype(np.float64)
    bid_price = pivot_price['bid'].values.astype(np.float64)
    ask_price = pivot_price['ask'].values.astype(np.float64)
    batch = _batch_orderbook_nb(bid_vol, ask_vol, bid_price, ask_price)
    return pd.DataFrame(batch, columns=['imbalance', 'spread'], index=pivot_vol.index)