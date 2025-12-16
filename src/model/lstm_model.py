"""
LSTM Model for SCADA Fault Prediction.

This module implements an LSTM neural network for binary classification
of wind turbine faults using TensorFlow/Keras.
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs,
    LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE, DENSE_UNITS,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, RANDOM_SEED
)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    
    X_train = np.load(os.path.join(WIND_FARM_A_PROCESSED, "X_train.npy"))
    X_val = np.load(os.path.join(WIND_FARM_A_PROCESSED, "X_val.npy"))
    X_test = np.load(os.path.join(WIND_FARM_A_PROCESSED, "X_test.npy"))
    
    y_train = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_train.npy"))
    y_val = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_val.npy"))
    y_test = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_test.npy"))
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary with class weights
    """
    n_samples = len(y)
    n_pos = y.sum()
    n_neg = n_samples - n_pos
    
    # Compute weights inversely proportional to class frequencies
    weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
    weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
    
    class_weight = {0: weight_neg, 1: weight_pos}
    print(f"\nClass weights: 0={weight_neg:.4f}, 1={weight_pos:.4f}")
    
    return class_weight


def build_lstm_model(input_shape: tuple, 
                     lstm_units_1: int = LSTM_UNITS_1,
                     lstm_units_2: int = LSTM_UNITS_2,
                     dropout_rate: float = DROPOUT_RATE,
                     dense_units: int = DENSE_UNITS,
                     learning_rate: float = LEARNING_RATE) -> keras.Model:
    """
    Build LSTM model for binary classification.
    
    Architecture:
        Input -> LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Dense(1)
    
    Args:
        input_shape: (sequence_length, n_features)
        lstm_units_1: Units in first LSTM layer
        lstm_units_2: Units in second LSTM layer
        dropout_rate: Dropout rate
        dense_units: Units in dense layer
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        LSTM(lstm_units_2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_callbacks(model_path: str, patience: int = EARLY_STOPPING_PATIENCE) -> list:
    """
    Get training callbacks.
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def train_model(model: keras.Model, 
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                class_weight: dict,
                model_path: str,
                epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE) -> keras.callbacks.History:
    """
    Train the LSTM model.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        class_weight: Class weights for imbalanced data
        model_path: Path to save best model
        epochs: Maximum number of epochs
        batch_size: Batch size
        
    Returns:
        Training history
    """
    callbacks = get_callbacks(model_path)
    
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def plot_training_history(history: keras.callbacks.History, save_path: str):
    """
    Plot and save training history.
    
    Args:
        history: Training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train')
    axes[0, 0].plot(history.history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def evaluate_model(model: keras.Model, X: np.ndarray, y: np.ndarray,
                   dataset_name: str = "Test") -> dict:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X: Feature sequences
        y: True labels
        dataset_name: Name for printing
        
    Returns:
        Dictionary of metrics
    """
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else 0.0
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix ({dataset_name}):")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    print(f"\nClassification Report ({dataset_name}):")
    print(classification_report(y, y_pred, target_names=['Normal', 'Fault']))
    
    return metrics


def main():
    """Main training and evaluation pipeline."""
    print("=" * 60)
    print("SCADA Fault Prediction - LSTM Model")
    print("=" * 60)
    
    # Set seeds
    set_seeds(RANDOM_SEED)
    
    # Ensure directories exist
    ensure_dirs()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Compute class weights
    class_weight = compute_class_weights(y_train)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, n_features)
    print(f"\nInput shape: {input_shape}")
    
    model = build_lstm_model(input_shape)
    model.summary()
    
    # Train model
    model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        class_weight, model_path
    )
    
    # Plot training history
    history_plot_path = os.path.join(RESULTS_DIR, "lstm_training_history.png")
    plot_training_history(history, history_plot_path)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Save results
    results = {
        'model': 'LSTM',
        'architecture': {
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'dropout_rate': DROPOUT_RATE,
            'dense_units': DENSE_UNITS
        },
        'training': {
            'epochs_trained': len(history.history['loss']),
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        },
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(RESULTS_DIR, "lstm_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("LSTM training complete!")
    print("=" * 60)
    
    return model, test_metrics


if __name__ == "__main__":
    main()
