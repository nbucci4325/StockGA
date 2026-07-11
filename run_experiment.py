"""
The experiment-time entry point: given a single stock ticker, runs the
full pre-GA pipeline for it:

    fetch (with buffer) -> compute indicators -> compute features
    -> load trained NN + scaler -> predict crossover rate

and returns the processed OHLCV+indicator data for that stock's
~1-year window, plus the crossover rate derived from it based on the forward volatility. 
This then leads next to the population initialization of the GA.
"""

from __future__ import annotations
from datetime import date
import pandas as pd
from config import CONFIG
from data.fetcher import fetch_ohclv_with_buffer, get_experiment_date_range
from features.indicators import add_all_indicators
from features.dataset_builder import build_feature_matrix
from model.predict import load_trained_artifacts, predict_crossover_rate


def run_experiment(ticker: str, as_of: date = None) -> dict:
    """
    Runs the full pre-GA pipeline for a single stock.

    Args:
        ticker: the stock ticker to run the experiment for.
        as_of: optional reference date for the experiment window (defaults
               to today). Passing a fixed date makes experiments
               reproducible across separate runs.

    Returns a dict with:
        ticker                -- the ticker requested
        window_start/window_end -- the experiment date window used
        ohlcv_with_indicators  -- DataFrame of OHLCV + all indicator
                                   columns for the window of interest
                                   (buffer rows already excluded)
        predicted_volatility   -- the NN's predicted forward volatility
                                   for this stock (most recent snapshot)
        crossover_rate         -- the crossover rate derived from that
                                   prediction, ready to hand off to the
                                   GA
    """
    window_start, window_end = get_experiment_date_range(as_of)

    print(f"[run_experiment] Fetching data for {ticker}: {window_start} to "
          f"{window_end} (plus indicator warmup buffer)...")
    buffered_df = fetch_ohclv_with_buffer(ticker, window_start, window_end)

    if buffered_df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' -- check that the "
            f"ticker symbol is valid and has trading history in this range."
        )

    # Compute indicators and features on the FULL buffered range (see
    # module docstring for why this ordering matters).
    indicator_df = add_all_indicators(buffered_df)
    feature_df = build_feature_matrix(indicator_df)

    # Now slice both down to the window actually being analyzed, dropping
    # the buffer rows that existed only to warm up rolling calculations.
    window_indicator_df = indicator_df.loc[str(window_start):]
    window_feature_df = feature_df.loc[str(window_start):]

    if window_feature_df.isna().any().any():
        n_nan_rows = window_feature_df.isna().any(axis=1).sum()
        print(f"[run_experiment] WARNING: {n_nan_rows} row(s) in the "
              f"analysis window still contain NaN feature values -- the "
              f"indicator warmup buffer may be insufficient for this "
              f"ticker's data. Dropping those rows.")
        window_feature_df = window_feature_df.dropna()

    if window_feature_df.empty:
        raise ValueError(
            f"No valid feature rows remain for '{ticker}' after processing "
            f"-- the requested window may be too short relative to the "
            f"indicator warmup requirements."
        )

    # Align the returned OHLCV+indicator data to the same valid rows as
    # the feature matrix, so the two pieces of returned data are always
    # consistent with each other -- callers should never see a row in
    # ohlcv_with_indicators that wasn't actually usable for prediction.
    window_indicator_df = window_indicator_df.loc[window_feature_df.index]

    model, scaler = load_trained_artifacts()
    prediction = predict_crossover_rate(model, scaler, window_feature_df)

    print(f"[run_experiment] {ticker}: predicted forward volatility = "
          f"{prediction['predicted_volatility']:.4f}, crossover rate = "
          f"{prediction['crossover_rate']:.4f}")

    return {
        "ticker": ticker,
        "window_start": window_start,
        "window_end": window_end,
        "ohlcv_with_indicators": window_indicator_df,
        "predicted_volatility": prediction["predicted_volatility"],
        "crossover_rate": prediction["crossover_rate"],
    }


if __name__ == "__main__":
    import sys

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = run_experiment(ticker_arg)
    print(f"\nDone. Crossover rate for {result['ticker']}: "
          f"{result['crossover_rate']:.4f}")