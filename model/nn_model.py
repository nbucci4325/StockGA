"""
Here the NN's architecture is defined to be used to predict a stock's forward
volatility from its normalized feature vector (built under features/dataset_builder.py)

This is a standard imported Kera model, sequential of dense layers.
"""

from __future__ import annotations
from tensorflow import keras
from tensorflow.keras import layers
from config import CONFIG
import os


def build_model(input_dim: int) -> keras.Model:
    """
    Builds and compiles feed-forward NN for regression
    single output: predicted forward volatility

    Architecture and parameters are read from config.py instead of hardcoded here, so adjustmments are made there.
    """
    model_cfg = CONFIG.model

    model = keras.Sequential(name="volatility_predictor")
    model.add(keras.Input(shape=(input_dim)))

    for units in model_cfg.hidden_layer_sizes:
        model.add(layers.Dense(units, activation=model_cfg.activation))

    # single output of predicted forward volatility
    model.add(layers.Dense(1, activation=None))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=model_cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model

def save_model(model: keras.Model, path: str = None) -> None:
    """
    Saves a trained model, creating parent directories if they don't already exist.
    Save apth wil default to CONFIG.model.save_path if none specified
    """
    if path is None:
        path = CONFIG.model.save_path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_model(path: str = None) -> keras.Model:
    """
    loads a model previously saved using save_model
    path will default to CONFIG.model.save_path if none specified
    """
    if path is None:
        path = CONFIG.model.save_path

    return keras.models.load_model(path)