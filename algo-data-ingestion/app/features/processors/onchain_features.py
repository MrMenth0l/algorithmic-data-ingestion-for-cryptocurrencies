import pandas as pd
import numpy as np

def compute_diff(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Compute difference of 'value' column over a specified period.
    """
    s = df.set_index('timestamp')['value']
    return s.diff(period).rename(f'diff_{period}')


def compute_pct_change(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Compute percentage change of 'value' column over a specified period.
    """
    s = df.set_index('timestamp')['value']
    return s.pct_change(period).fillna(0).rename(f'pct_change_{period}')


def compute_rolling_mean(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute rolling mean of 'value' over a window of periods.
    """
    s = df.set_index('timestamp')['value']
    return s.rolling(window).mean().rename(f'rolling_mean_{window}')


def compute_rolling_std(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute rolling standard deviation of 'value' over a window of periods.
    """
    s = df.set_index('timestamp')['value']
    return s.rolling(window).std().rename(f'rolling_std_{window}')


def compute_drawdown(df: pd.DataFrame) -> pd.Series:
    """
    Compute drawdown series for 'value'.
    """
    s = df.set_index('timestamp')['value']
    running_max = s.cummax()
    drawdown = (s - running_max) / running_max
    return drawdown.rename('drawdown')


def compute_whale_flow(df: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Count instances where 'value' exceeds a threshold in each timestamp.
    """
    s = df.set_index('timestamp')['value']
    return (s.abs() > threshold).astype(int).rename(f'whale_flow_{threshold}')


def compute_rolling_median(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute rolling median of 'value' over a window of periods.
    """
    s = df.set_index('timestamp')['value']
    return s.rolling(window).median().rename(f'rolling_median_{window}')


def compute_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute rolling volatility (std of returns) annualized over a window.
    """
    s = df.set_index('timestamp')['value']
    returns = s.pct_change().dropna()
    vol = returns.rolling(window).std()
    annualized_vol = vol * np.sqrt(365 * (24*60*60) / window)  # assuming seconds timestamp
    return annualized_vol.rename(f'volatility_{window}')


def compute_zscore(df: pd.DataFrame) -> pd.Series:
    """
    Compute z-score of 'value' over the entire series.
    """
    s = df.set_index('timestamp')['value']
    return ((s - s.mean()) / s.std()).rename('zscore')


def compute_rolling_quantile(df: pd.DataFrame, window: int, quantile: float = 0.5) -> pd.Series:
    """
    Compute rolling quantile of 'value' over a window.
    """
    s = df.set_index('timestamp')['value']
    return s.rolling(window).quantile(quantile).rename(f'rolling_q{quantile}_{window}')


def compute_cumulative(df: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative sum of 'value'.
    """
    s = df.set_index('timestamp')['value']
    return s.cumsum().rename('cumulative_value')