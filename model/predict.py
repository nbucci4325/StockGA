"""
This is experiment-time logic for a single, specific stock.

2 distinctive steps are involved:
1. run the alrady trained NN on one stocks feature window to predict a real, learnable quality. In this case, being forward volatility.
2. map_to_crossover_rate is a function used to map the predicted volatility to a crossover rate to be used by the GA.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from config import CONFIG
from model.nn_model import load_model, load_scaler


def predict_characteristic( model: keras.Model, scaler: StandardScaler, stock_features: pd.DataFrame) -> float:
    """
    predicts forward volatility for a single stock, using the most recent row of stock_features. This would be
    the latest trading day in the experimet window.

    By this point, stock_features should already have NaN rows dropped.
    """
    if stock_features.empty:
        raise ValueError("stock_features is empty, cannot predict on no data.")

    latest_row = stock_features.iloc[[-1]]  # keep as a 1-row DataFrame, preserves column order/names
    scaled = scaler.transform(latest_row.values)

    raw_prediction = float(model.predict(scaled, verbose=0).flatten()[0])

    return max(raw_prediction, 0.0)


def map_to_crossover_rate(predicted_volatility: float) -> float:
    """
    Hardcoded function responsible for mapping the predicted forward volatility to a crossover rate,
    ensuring it stays within the specified range defined under config.py.
    """
    cfg = CONFIG.crossover_mapping

    clipped = min(
        max(predicted_volatility, cfg.volatility_reference_min),
        cfg.volatility_reference_max,
    )

    vol_range = cfg.volatility_reference_max - cfg.volatility_reference_min
    fraction = (clipped - cfg.volatility_reference_min) / vol_range

    rate_range = cfg.crossover_rate_max - cfg.crossover_rate_min
    crossover_rate = cfg.crossover_rate_min + fraction * rate_range

    return crossover_rate


def predict_crossover_rate(model: keras.Model, scaler: StandardScaler, stock_features: pd.DataFrame) -> dict:
    """
    wrapper combining the steps for a single stock, predicting the forward volatility and mapping that to a crossover rate.
    """
    predicted_volatility = predict_characteristic(model, scaler, stock_features)
    crossover_rate = map_to_crossover_rate(predicted_volatility)

    return {
        "predicted_volatility": predicted_volatility,
        "crossover_rate": crossover_rate,
    }


def load_trained_artifacts() -> tuple:
    """
    Loads the trained model and its paired scalar from the paths configured in CONFIG.model.
    """
    model = load_model()
    scaler = load_scaler()
    return model, scaler