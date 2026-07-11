"""
This will include functions for computing fixed periods indicators for describing a stock as input to the NN.
These are not the chromosome genes for the GA, this is date used for the NN training and prediction.

Every function here given an OHCLV DataFrame from data/fetcher.py and returns a new DataFrame with computed values
"""

from __future__ import annotations
import pandas as pd
from config import CONFIG
 
 
def wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    """
    Computes Wilder's smoothing for a given series and period.
    Used for both RSI and ATR per their standard definitions. This is an exponential moving average with alpha = 1/period.
    """
    return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def compute_sma(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.Series:
    """
    Computes the Simple Moving Average (SMA) for a given DataFrame and window size.
    """
    return df[price_col].rolling(window=window, min_periods=window).mean().rename(f"sma_{window}")


def compute_rsi(df: pd.DataFrame, period: int, price_col: str = "close") -> pd.Series:
    """
    Computes the Relative Strength Index (RSI) for a given DataFrame and period,
    using Wilder's smoothing method of average gains and losses.
    """
    delta = df[price_col].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = wilder_smoothing(gain, period)
    avg_loss = wilder_smoothing(loss, period)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Flat-price edge case where avg_gain and avg_loss are both zero, resulting in NaN. In this case, we define RSI to be 50 (neutral).
    flat_price_mask = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.mask(flat_price_mask, 50)

    return rsi.astype(float).rename(f"rsi_{period}")


def compute_macd( df: pd.DataFrame, fast: int, slow: int, signal: int, price_col: str = "close") -> pd.DataFrame:
    """
    MACD line (fast EMA - slow EMA), signal line (EMA of MACD line), and
    histogram (MACD line - signal line).
    """
    fast_ema = df[price_col].ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = df[price_col].ewm(span=slow, adjust=False, min_periods=slow).mean()
 
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
 
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })
 
 
def compute_bollinger_bands(df: pd.DataFrame, window: int, num_std: float, price_col: str = "close") -> pd.DataFrame:
    """
    Bollinger Bands: rolling mean, plus/minus num_std rolling std devs.
    """
    mid = df[price_col].rolling(window=window, min_periods=window).mean()
    std = df[price_col].rolling(window=window, min_periods=window).std()
 
    upper = mid + num_std * std
    lower = mid - num_std * std
 
    return pd.DataFrame({"bb_upper": upper, "bb_mid": mid, "bb_lower": lower})
 
 
def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Average True Range using Wilder's smoothing of the true range series.
    """
    prev_close = df["close"].shift(1)
 
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
 
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
 
    atr = wilder_smoothing(true_range, period)
    return atr.rename(f"atr_{period}")
 
 
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes all the indicators defined above and adds them to the original DataFrame.
    It uses the fixed periods from CONFIG.indicators, and appends them as new columns to the original DataFrame (input OHCLV DataFrame).
    """
    ind_cfg = CONFIG.indicators
    out = df.copy()
 
    out[f"sma_{ind_cfg.sma_short_period}"] = compute_sma(df, ind_cfg.sma_short_period)
    out[f"sma_{ind_cfg.sma_long_period}"] = compute_sma(df, ind_cfg.sma_long_period)
 
    out[f"rsi_{ind_cfg.rsi_period}"] = compute_rsi(df, ind_cfg.rsi_period)
 
    macd_df = compute_macd(
        df,
        fast=ind_cfg.macd_fast_period,
        slow=ind_cfg.macd_slow_period,
        signal=ind_cfg.macd_signal_period,
    )
    out = pd.concat([out, macd_df], axis=1)
 
    bb_df = compute_bollinger_bands(
        df,
        window=ind_cfg.bbands_period,
        num_std=ind_cfg.bbands_num_std,
    )
    out = pd.concat([out, bb_df], axis=1)
 
    out[f"atr_{ind_cfg.atr_period}"] = compute_atr(df, ind_cfg.atr_period)
 
    return out