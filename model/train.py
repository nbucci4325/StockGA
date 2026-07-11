"""
Fits and evaluates the volatility-prediction NN

Feature Sclaing:
the columns int he feature matrix are in ratios.%'s/z-scores, but are not in a uniform format directly comparable to each other.
To solve this, a StandardScaler is used here on the training data only, then applied to train both train and validation sets.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from config import CONFIG
from model.nn_model import build_model

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame) -> tuple:
    """
    Fits the volatility prediction NN

    Starts with fitting the StandardSclaer on X_train only, then to transform X_train and X_val with that fitted scalar.
    The model is then built and fit with early stopping on validation loss, restoring the best spoch's weights.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
 
    model = build_model(input_dim=X_train.shape[1])
 
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CONFIG.model.early_stopping_patience,
        restore_best_weights=True,
    )
 
    history = model.fit(
        X_train_scaled,
        y_train.values,
        validation_data=(X_val_scaled, y_val.values),
        epochs=CONFIG.model.epochs,
        batch_size=CONFIG.model.batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
 
    return model, scaler, history


def evaluate_model(model: keras.Model, scaler: StandardScaler, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Evaluates a trained model on a validation set, applying the same
    fitted scaler used during training. Returns a dict of standard
    regression metrics so you can sanity-check the NN isn't garbage
    before trusting its output at inference time.
 
    Includes a baseline comparison (predicting the mean of y_val for
    every row) so the metrics have context.
    """
    X_val_scaled = scaler.transform(X_val.values)
    preds = model.predict(X_val_scaled, verbose=0).flatten()
 
    mse = mean_squared_error(y_val.values, preds)
    mae = mean_absolute_error(y_val.values, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val.values, preds)
 
    baseline_preds = np.full_like(y_val.values, fill_value=y_val.values.mean())
    baseline_mae = mean_absolute_error(y_val.values, baseline_preds)
 
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "baseline_mae_predicting_mean": baseline_mae,
        "improvement_over_baseline_mae_pct": (
            100 * (baseline_mae - mae) / baseline_mae if baseline_mae > 0 else float("nan")
        ),
    }