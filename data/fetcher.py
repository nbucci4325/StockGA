"""
This is a wrapper around yfinance so it's not called anywhere else directly.

It will be responsible for the following:
- Normalize column names/index across inconsistent return shapes
- Handle the necessary "buffer logic" so rolling indicators that require a lookback period can be calculated without NaN's
- Fetch multiple tickers
"""

from __future__ import annotations
import pandas as pd 
import yfinance as yf
from config import CONFIG
from datetime import datetime, timedelta, date
from typing import Optional

# Calculating and storing the ratio of calendar days to trading days
# This is done to convert a desired number of trading-days buffer into a calendar window for the yfinance query, since yfinance queries by calendar days.
# Some extra days will be added to this to account for holidays.
_TRADING_TO_CALENDAR_RATIO = 365 / 252  # ~1.45
_CALENDAR_BUFFER_DAYS = 10  # extra days to account for

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the column names of the DataFrame returned by yfinance to a consistent format, 
    and flatten yfinance's multi index columns down to a single-level.
    """
    if df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    })
    df.index.name = 'date'

    return df


def fetch_ohclv(ticker: str, start_date, end_date) -> pd.DataFrame:
    """
    Fetches OHCLV data for a given ticker from yfinance, between start and end dates. (start inclusive, end exclusive)
    If no dates are provided, it fetches the maximum available data.
    End date defaulted to today, and start date defaults to a year ago (with a buffer for rolling indicators).

    Returns a DataFrame with standardized column names and a datetime index, with the following columns: ['open', 'high', 'low', 'close', 'adj_close', 'volume'].

    DataFrame returned is empty if there's no data for the given ticker (invalid ticker, no data in the given date range, etc.).
    """

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

    return standardize_columns(df)




def fetch_ohclv_with_buffer(ticker: str, window_start, window_end, buffer_trading_days: int = None) -> pd.DataFrame:
    """
    Fetches OHCLV data for a given ticker from yfinance, over the specified range PLUS extra data from before window_start for rolling indicators
    to be calculated.
    End date defaulted to today, and start date defaults to a year ago (with a buffer for rolling indicators).

    buffer_trading_days: will default to CONFIG.data.warmup_buffer_days if not provided.

    This will return a DataFrame with the full fetched range, buffer included.
    """

    if buffer_trading_days is None:
        buffer_trading_days = CONFIG.indicators.warmup_buffer_days


    # Calculate the number of calendar days before window_start to fetch based on the trading days and the ratio
    calendar_days_to_fetch = int(buffer_trading_days * _TRADING_TO_CALENDAR_RATIO) + _CALENDAR_BUFFER_DAYS


    if isinstance(window_start, str):
        window_start = datetime.strptime(window_start, "%Y-%m-%d").date()

    # Adjust the start date to fetch more data for rolling indicators
    adjusted_start_date = window_start - timedelta(days=calendar_days_to_fetch)

    df = yf.download(ticker, start=adjusted_start_date, end=window_end, auto_adjust=False, progress=False)

    return standardize_columns(df)


def fetch_multi_ticker(tickers: list[str], start_date, end_date) -> dict:
    """
    Fetches OHCLV data for multiple tickers instead of one, so a single bad ticker doesn't break the whole process. 
    
    Returns a dict of DataFrames, keyed by ticker.
    """

    data_dict = {}
    for ticker in tickers:
        try:
            df = fetch_ohclv(ticker, start_date, end_date)
            if df.empty:
                print(f"Warning: No data found for ticker {ticker} in the range {start_date} to {end_date}.")
                continue

            data_dict[ticker] = df
        except Exception as e:
            print(f"Failed to fetch ticker: {ticker} ({e})")

    return data_dict


def get_experiment_date_range(as_of: Optional[date] = None) -> tuple:
    """
    Returns (start_date, end_date) for the experiment-time window: the
    past CONFIG.data.experiment_history_days days, ending today (or
    `as_of` if provided -- useful for reproducible, dated experiment runs).
    """
    end_date = as_of or date.today()
    start_date = end_date - timedelta(days=CONFIG.data.experiment_history_days)
    return start_date, end_date
 
 
def get_training_date_range(as_of: Optional[date] = None) -> tuple:
    """
    Returns (start_date, end_date) for the NN training window: the past
    CONFIG.data.training_history_years years, ending today (or `as_of`).
    """
    end_date = as_of or date.today()
    start_date = end_date - timedelta(days=CONFIG.data.training_history_years * 365)
    return start_date, end_date