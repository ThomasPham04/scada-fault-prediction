"""
Train LSTM — training.scripts.train_lstm
Entry-point script for training the LSTM prediction model.

Usage:
    python -m src.training.scripts.train_lstm
    # or
    python src/training/scripts/train_lstm.py
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from config import (
    WIND_FARM_A_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
    NBM_BATCH_SIZE,
    NBM_EPOCHS,
    RANDOM_SEED,
    ensure_dirs,
)
from models.architectures.lstm import build_lstm_model
from training.callbacks.early_stopping import get_lstm_callbacks


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Ensure deterministic training runs."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    """Load preprocessed LSTM training data from disk."""
    import joblib
    data_dir = os.path.join(WIND_FARM_A_PROCESSED, '7day')

    print("Loading 7-day preprocessed data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val   = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val   = np.load(os.path.join(data_dir, 'y_val.npy'))
    metadata = joblib.load(os.path.join(data_dir, 'metadata_7day.pkl'))

    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  Features: {metadata['n_features']}, Window: {metadata['window_size']}")

    return {
        'X_train': X_train, 'X_val': X_val,
        'y_train': y_train, 'y_val': y_val,
        'metadata': metadata,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_val, y_val, model_path: str):
    """Run model.fit with the standard LSTM callback set."""
    print(f"\nTraining LSTM (epochs={NBM_EPOCHS}, batch={NBM_BATCH_SIZE})...")
    print(f"  Train samples: {len(X_train)} | Val samples: {len(X_val)}")
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NBM_EPOCHS,
        batch_size=NBM_BATCH_SIZE,
        callbacks=get_lstm_callbacks(model_path),
        verbose=1,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_history(history, save_path: str) -> None:
    """Save loss and MAE training curves."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, metric, ylabel in zip(
        axes,
        [('loss', 'val_loss'), ('mae', 'val_mae')],
        ['Loss (MSE)', 'MAE'],
    ):
        ax.plot(history.history[metric[0]], label='Train', linewidth=2)
        ax.plot(history.history[metric[1]], label='Val', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Training and Validation {ylabel}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("LSTM Prediction Model — Training Pipeline")
    print("=" * 70)

    set_seeds()
    ensure_dirs()

    results_dir = os.path.join(RESULTS_DIR, '7day')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    data = load_data()
    window_size = data['X_train'].shape[1]
    n_features  = data['X_train'].shape[2]

    # Build model
    print(f"\nBuilding LSTM (window={window_size}, features={n_features})...")
    model = build_lstm_model(input_shape=(window_size, n_features), output_dim=n_features)

    # Train
    model_path = os.path.join(MODELS_DIR, 'lstm_7day.keras')
    history = train(model, data['X_train'], data['y_train'], data['X_val'], data['y_val'], model_path)

    # Plot history
    plot_training_history(history, os.path.join(results_dir, 'training_history.png'))

    # Reload best weights
    print(f"\nLoading best model from: {model_path}")
    model = keras.models.load_model(model_path)

    # Quick evaluation
    from evaluation.evaluate_lstm import evaluate_train_val, plot_error_distributions, save_results
    errors = evaluate_train_val(model, data['X_train'], data['y_train'], data['X_val'], data['y_val'])
    plot_error_distributions(errors, os.path.join(results_dir, 'error_distributions.png'))
    save_results(errors, data['metadata'], results_dir)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Model:   {model_path}")
    print(f"  Results: {results_dir}")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run threshold tuning: src/training/experiments/threshold_tuning.py")
    print("  2. Evaluate on test set: src/evaluation/evaluate_lstm.py")


if __name__ == "__main__":
    main()
