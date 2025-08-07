import pandas as pd
import numpy as np
from numba import jit
from numba import njit
@njit
def _cci_nb(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int, constant: float) -> np.ndarray:
    n = high.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            # Compute typical prices for the window
            tp_sum = 0.0
            for j in range(i - window + 1, i + 1):
                tp_sum += (high[j] + low[j] + close[j]) / 3.0
            ma = tp_sum / window
            # Mean deviation
            md_sum = 0.0
            for j in range(i - window + 1, i + 1):
                tp = (high[j] + low[j] + close[j]) / 3.0
                md_sum += abs(tp - ma)
            md = md_sum / window
            tp_current = (high[i] + low[i] + close[i]) / 3.0
            out[i] = (tp_current - ma) / (constant * md) if md != 0 else 0.0
    return out

# Batch kernel for CCI and ROC using Numba
@njit
def _batch_indicators_nb(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int, constant: float) -> np.ndarray:
    n = high.shape[0]
    cci_out = np.empty(n, dtype=np.float64)
    roc_out = np.empty(n, dtype=np.float64)
    for i in range(n):
        # CCI
        if i < window - 1:
            cci_out[i] = np.nan
        else:
            tp_sum = 0.0
            for j in range(i - window + 1, i + 1):
                tp_sum += (high[j] + low[j] + close[j]) / 3.0
            ma = tp_sum / window
            md_sum = 0.0
            for j in range(i - window + 1, i + 1):
                tp = (high[j] + low[j] + close[j]) / 3.0
                md_sum += abs(tp - ma)
            md = md_sum / window
            tp_current = (high[i] + low[i] + close[i]) / 3.0
            cci_out[i] = (tp_current - ma) / (constant * md) if md != 0 else 0.0
        # ROC
        if i < window:
            roc_out[i] = np.nan
        else:
            roc_out[i] = 100.0 * ((close[i] - close[i - window]) / close[i - window])
    # Combine outputs: stack cci then roc
    result = np.vstack((cci_out, roc_out)).T
    return result

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

def compute_vwap(df: pd.DataFrame, window: int, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
    """
    Compute the rolling Volume Weighted Average Price (VWAP) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing price and volume columns.
    window (int): The rolling window size.
    price_col (str): Name of the price column (default 'close').
    volume_col (str): Name of the volume column (default 'volume').

    Returns:
    pd.Series: Rolling VWAP values.
    """
    pv = df[price_col] * df[volume_col]
    vwap = pv.rolling(window).sum() / df[volume_col].rolling(window).sum()
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
    Compute Commodity Channel Index (CCI) using a Numba-optimized kernel.
    """
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    close_arr = df['close'].values.astype(np.float64)
    out = _cci_nb(high_arr, low_arr, close_arr, window, constant)
    return pd.Series(out, index=df.index)

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


# Batch compute CCI and ROC in one Numba pass
def compute_batch_indicators(df: pd.DataFrame, window: int = 20, constant: float = 0.015) -> pd.DataFrame:
    """
    Compute batch indicators (CCI and ROC) in one pass using Numba.
    Returns a DataFrame with columns 'cci' and 'roc'.
    """
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    close_arr = df['close'].values.astype(np.float64)
    batch = _batch_indicators_nb(high_arr, low_arr, close_arr, window, constant)
    cci_series = pd.Series(batch[:, 0], index=df.index)
    roc_series = pd.Series(batch[:, 1], index=df.index)
    return pd.DataFrame({'cci': cci_series, 'roc': roc_series})

