"""
Callbacks - training.callbacks.early_stopping
Reusable Keras callback factories for LSTM and other model training.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tensorflow.keras import callbacks
from config import RESULTS_DIR, NBM_EARLY_STOPPING_PATIENCE


def get_lstm_callbacks(model_path: str, log_dir: str = None) -> list:
    """Standard callback set for LSTM training."""
    if log_dir is None:
        log_dir = os.path.join(RESULTS_DIR, "lstm_7day_logs")
    return [
        callbacks.ModelCheckpoint(filepath=model_path, monitor="val_loss",
                                  save_best_only=True, mode="min", verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=NBM_EARLY_STOPPING_PATIENCE,
                                mode="min", verbose=1, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                    mode="min", min_lr=1e-7, verbose=1),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
    ]


def get_generic_callbacks(model_path: str, monitor: str = "val_loss",
                          patience: int = 10, log_dir: str = None) -> list:
    """Generic callback set reusable for any Keras model."""
    cb_list = [
        callbacks.ModelCheckpoint(filepath=model_path, monitor=monitor,
                                  save_best_only=True, mode="min", verbose=1),
        callbacks.EarlyStopping(monitor=monitor, patience=patience,
                                mode="min", verbose=1, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                    patience=max(1, patience // 2),
                                    mode="min", min_lr=1e-7, verbose=1),
    ]
    if log_dir:
        cb_list.append(callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0))
    return cb_list