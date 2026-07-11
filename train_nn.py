"""
Builds the NN's training dataset across the full list of training tickers, 
trains the volatility-prediction model, evaluates it on the
held-out validation set, and saves the trained model + scaler.
"""

from __future__ import annotations
from config import CONFIG
from data.fetcher import get_training_date_range
from features.dataset_builder import build_training_dataset, train_val_split
from model.train import train_model, evaluate_model
from model.nn_model import save_model, save_scaler


def main(tickers: list = None, start_date=None, end_date=None) -> tuple:
    """
    Runs the full training pipeline end-to-end. Arguments default to
    CONFIG values but can be overwritten.

    Returns (model, scaler, metrics).
    """
    tickers = tickers if tickers is not None else list(CONFIG.data.training_tickers)

    if start_date is None or end_date is None:
        start_date, end_date = get_training_date_range()

    print(f"[train_nn] Building training dataset: {len(tickers)} tickers, "
          f"{start_date} to {end_date}...")
    X, y = build_training_dataset(tickers, start_date, end_date)

    n_tickers_used = X.index.get_level_values("ticker").nunique()
    print(f"[train_nn] Dataset built: {X.shape[0]} rows, {X.shape[1]} features, "
          f"from {n_tickers_used}/{len(tickers)} tickers "
          f"({len(tickers) - n_tickers_used} skipped due to fetch/data issues).")

    X_train, X_val, y_train, y_val = train_val_split(X, y)
    print(f"[train_nn] Split (method={CONFIG.data.train_val_split_method}): "
          f"train={X_train.shape[0]} rows, val={X_val.shape[0]} rows.")

    print("[train_nn] Training model...")
    model, scaler, history = train_model(X_train, y_train, X_val, y_val)
    n_epochs_run = len(history.history["loss"])
    print(f"[train_nn] Training complete after {n_epochs_run} epochs "
          f"(configured max was {CONFIG.model.epochs}, "
          f"early stopping patience={CONFIG.model.early_stopping_patience}).")

    metrics = evaluate_model(model, scaler, X_val, y_val)
    print("[train_nn] Validation metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")

    if metrics["improvement_over_baseline_mae_pct"] <= 0:
        print("[train_nn] WARNING: model does not improve on the "
              "predict-the-mean baseline. Review features/training data "
              "before trusting this model's predictions downstream.")

    save_model(model)
    save_scaler(scaler)
    print(f"[train_nn] Model saved to {CONFIG.model.save_path}")
    print(f"[train_nn] Scaler saved to {CONFIG.model.scaler_save_path}")

    return model, scaler, metrics


if __name__ == "__main__":
    main()