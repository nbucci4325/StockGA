"""
Transforming the raw OHCLV data into ML-ready training data for the volatile-prediction NN.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from config import CONFIG
from data.fetcher import fetch_multi_ticker
from features.indicators import add_all_indicators

_TRADING_DAYS_PER_YEAR = 252  # ~1 year of trading days
_VOLUME_ZSCORE_WINDOW = 20 # rolling window for normalizing volume


def compute_forward_volatility(df: pd.DataFrame, horizon: int, price_col: str = "close") -> pd.Series:
    """
    Computes the forward volatility for each day in the DataFrame, over the specified horizon.
    Forward volatility is defined as the standard deviation of daily returns over the next `horizon` trading days, annualized.

    The last `horizon` days in the DataFrame will have NaN forward volatility, since there are not enough future days to compute it.
    This is expected and will be dropped before training.
    """
    log_returns = np.log(df[price_col] / df[price_col].shift(1))

    trailing_vol = log_returns.rolling(window=horizon, min_periods=horizon).std()
    forward_vol = trailing_vol.shift(-horizon) * np.sqrt(_TRADING_DAYS_PER_YEAR)

    return forward_vol.rename(f"forward_vol_{horizon}d")



def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the NN input feature matrix from an OHLCV+indicator DataFrame
    (i.e. the output of features.indicators.add_all_indicators).
 
    Every feature here is a ratio, %, or z-score rather than a
    raw price/volume level, so it can be comarable across all tickers with 
    varying prices.
    """
    ind_cfg = CONFIG.indicators
    feat = pd.DataFrame(index=df.index)
 
    sma_short_col = f"sma_{ind_cfg.sma_short_period}"
    sma_long_col = f"sma_{ind_cfg.sma_long_period}"
    rsi_col = f"rsi_{ind_cfg.rsi_period}"
    atr_col = f"atr_{ind_cfg.atr_period}"
 
    # Trend features: price position relative to moving averages.
    feat["price_to_sma_short"] = df["close"] / df[sma_short_col]
    feat["price_to_sma_long"] = df["close"] / df[sma_long_col]
    feat["sma_short_to_long"] = df[sma_short_col] / df[sma_long_col]
 
    # Momentum: RSI is already bounded [0, 100], comparable across
    # tickers as-is -- no normalization needed.
    feat["rsi"] = df[rsi_col]
 
    # MACD: normalize by price so units are comparable across tickers
    # (raw MACD is in price-difference units, which scale with the stock's
    # absolute price level).
    feat["macd_line_norm"] = df["macd_line"] / df["close"]
    feat["macd_signal_norm"] = df["macd_signal"] / df["close"]
    feat["macd_hist_norm"] = df["macd_hist"] / df["close"]
 
    # Bollinger Bands: use the standard normalized forms (%B and
    # bandwidth) instead of the raw upper/mid/lower price levels.
    band_range = df["bb_upper"] - df["bb_lower"]
    feat["bb_percent_b"] = (df["close"] - df["bb_lower"]) / band_range
    feat["bb_bandwidth"] = band_range / df["bb_mid"]
 
    # ATR normalized as a percentage of price (a raw ATR of $2 means very
    # different things for a $20 stock vs. a $200 stock).
    feat["atr_pct"] = df[atr_col] / df["close"]
 
    # Volume: z-score relative to its own recent rolling stats, so scale
    # differences across tickers' typical volume are normalized away.
    vol_mean = df["volume"].rolling(window=_VOLUME_ZSCORE_WINDOW, min_periods=_VOLUME_ZSCORE_WINDOW).mean()
    vol_std = df["volume"].rolling(window=_VOLUME_ZSCORE_WINDOW, min_periods=_VOLUME_ZSCORE_WINDOW).std()
    feat["volume_zscore"] = (df["volume"] - vol_mean) / vol_std
 
    return feat


def build_training_dataset(tickers: list[str], start_date, end_date) -> tuple:
    """
    This builds the full (X, y) training dataset for the NN, across multiple clickers. It fetcher OHCLV data, computes indicators,
    computes forward volatility labels, and buiulds the feature matrix.

    Returns (X, y) where X is the feature matrix and y is the forward volatility labels.
    """
    horizon = CONFIG.labels.forward_volatility_horizon_days
    # Fetch OHCLV data for all tickers
    data_dict = fetch_multi_ticker(tickers, start_date, end_date)

    X_parts = []
    y_parts = []

    for ticker, df in data_dict.items():
        if df.empty:
            print(f"Warning: No data found for ticker {ticker} in the range {start_date} to {end_date}. Skipping.")
            continue

        # Compute indicators
        df_ind = add_all_indicators(df)

        # Compute forward volatility labels
        forward_vol = compute_forward_volatility(df_ind, horizon)

        feat = build_feature_matrix(df_ind)
        feat['forward_vol'] = forward_vol

        feat = feat.dropna()  # Drop null rows
        if feat.empty:
            print(f"Warning: No valid data after computing features and forward volatility for ticker {ticker}. Skipping.")
            continue

        feat.index = pd.MultiIndex.from_arrays([[ticker] * len(feat), feat.index], names=["ticker", "date"])

        X_parts.append(feat.drop(columns=["forward_vol"]))
        y_parts.append(feat["forward_vol"])

    if not X_parts:
        raise ValueError("No usable data was produced for any ticker in the training universe.")
 
    X = pd.concat(X_parts, axis=0)
    y = pd.concat(y_parts, axis=0)
 
    return X, y

def train_val_split(X: pd.DataFrame, y: pd.Series, random_seed: int = 42) -> tuple:
    """
    Splits (X, y) into train/validation sets using the method configured
    in CONFIG.data.train_val_split_method:
    
    "ticker": holds out entire tickers for validation (no ticker appears in both train and val). 
    Requires X to have a MultiIndex with a "ticker" level (as produced by build_training_dataset).
 
    "time": holds out a trailing time slice across all tickers combined (validates on the 
    most recent period). Requires X to have a MultiIndex with a "date" level.
 
    Returns (X_train, X_val, y_train, y_val).
    """
    method = CONFIG.data.train_val_split_method
    frac = CONFIG.data.train_val_split_fraction
 
    if method == "ticker":
        tickers = X.index.get_level_values("ticker")
        unique_tickers = sorted(tickers.unique())
 
        rng = np.random.default_rng(seed=random_seed)
        shuffled = list(unique_tickers)
        rng.shuffle(shuffled)
 
        n_val = max(1, int(len(shuffled) * frac))
        val_tickers = set(shuffled[:n_val])
 
        val_mask = tickers.isin(val_tickers)
 
    elif method == "time":
        dates = X.index.get_level_values("date")
        sorted_dates = dates.sort_values()
        cutoff_idx = int(len(sorted_dates) * (1 - frac))
        cutoff_date = sorted_dates[cutoff_idx]
 
        val_mask = dates >= cutoff_date
 
    else:
        raise ValueError(f"Unknown train_val_split_method: {method!r}")
 
    X_train, X_val = X[~val_mask], X[val_mask]
    y_train, y_val = y[~val_mask], y[val_mask]
 
    return X_train, X_val, y_train, y_val