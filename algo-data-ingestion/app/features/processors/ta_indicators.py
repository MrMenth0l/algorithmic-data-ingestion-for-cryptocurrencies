import pandas as pd
import numpy as np
from numba import jit

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a given price series.

    Parameters:
    prices (pd.Series): Series of prices.
    window (int): The lookback period for RSI calculation (default is 14).

    Returns:
    pd.Series: RSI values.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })

def compute_bollinger(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle_band = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = middle_band + num_std * std
    lower_band = middle_band - num_std * std
    return pd.DataFrame({
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band
    })

def compute_vwap(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
    """
    Compute the Volume Weighted Average Price (VWAP) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing price and volume columns.
    price_col (str): Name of the price column (default 'close').
    volume_col (str): Name of the volume column (default 'volume').

    Returns:
    pd.Series: VWAP values.
    """
    pv = df[price_col] * df[volume_col]
    vwap = pv.cumsum() / df[volume_col].cumsum()
    return vwap


# Additional technical indicators
def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute Simple Moving Average (SMA) for a given price series and window.
    """
    return prices.rolling(window).mean()

def compute_ema(prices: pd.Series, span: int) -> pd.Series:
    """
    Compute Exponential Moving Average (EMA) for a given price series and span.
    """
    return prices.ewm(span=span, adjust=False).mean()

def compute_atr(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close', window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_obv(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
    """
    Compute On-Balance Volume (OBV).
    """
    close = df[price_col]
    volume = df[volume_col]
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def compute_cci(df: pd.DataFrame, window: int = 20, constant: float = 0.015) -> pd.Series:
    """
    Compute Commodity Channel Index (CCI).
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window).mean()
    md = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (constant * md)

def compute_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator %K and %D.
    """
    low_min = df['low'].rolling(k_window).min()
    high_max = df['high'].rolling(k_window).max()
    percent_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(d_window).mean()
    return pd.DataFrame({'%K': percent_k, '%D': percent_d})

def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX).
    """
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window).mean()
    return adx

def compute_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Money Flow Index (MFI).
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0.0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0.0)
    positive_mf = positive_flow.rolling(window).sum()
    negative_mf = negative_flow.rolling(window).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    return mfi

def compute_roc(prices: pd.Series, window: int = 12) -> pd.Series:
    """
    Compute Rate of Change (ROC).
    """
    return 100 * (prices.diff(window) / prices.shift(window))

