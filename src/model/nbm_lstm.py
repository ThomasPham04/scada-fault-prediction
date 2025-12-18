"""
Normal Behavior Model (NBM) - LSTM Prediction Model

This module implements LSTM model for predicting next timestep from 14-day history.
Model trains ONLY on normal operation data and uses prediction error for anomaly detection.
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs,
    NBM_LSTM_UNITS_1, NBM_LSTM_UNITS_2, NBM_DROPOUT_RATE, NBM_DENSE_UNITS,
    NBM_LEARNING_RATE, NBM_BATCH_SIZE, NBM_EPOCHS, NBM_EARLY_STOPPING_PATIENCE,
    RANDOM_SEED
)


def set_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_nbm_data():
    """
    Load NBM preprocessed data.
    
    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test, metadata
    """
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM')
    
    print("Loading NBM preprocessed data...")
    X_train = np.load(os.path.join(nbm_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(nbm_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(nbm_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(nbm_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(nbm_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(nbm_dir, 'y_test.npy'))
    
    # Load metadata
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata.pkl'))
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Window size: {metadata['window_size']} timesteps")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'metadata': metadata
    }


def build_nbm_lstm_model(input_shape: tuple,
                         output_dim: int,
                         lstm_units_1: int = NBM_LSTM_UNITS_1,
                         lstm_units_2: int = NBM_LSTM_UNITS_2,
                         dropout_rate: float = NBM_DROPOUT_RATE,
                         dense_units: int = NBM_DENSE_UNITS,
                         learning_rate: float = NBM_LEARNING_RATE) -> Model:
    """
    Build LSTM Prediction Model for NBM.
    
    Architecture:
        Input (window_size, n_features)
        → LSTM(128, return_sequences=True)
        → Dropout(0.2)
        → LSTM(64, return_sequences=False)
        → Dropout(0.2)
        → Dense(64, ReLU)
        → Dense(n_features, Linear) - Predict next timestep
    
    Args:
        input_shape: (window_size, n_features)
        output_dim: Number of features to predict
        lstm_units_1, lstm_units_2: LSTM layer sizes
        dropout_rate: Dropout rate
        dense_units: Dense layer size
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input_sequence')
    
    # First LSTM layer with return_sequences=True
    x = layers.LSTM(lstm_units_1, return_sequences=True, name='lstm_1')(inputs)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Second LSTM layer, return last output only
    x = layers.LSTM(lstm_units_2, return_sequences=False, name='lstm_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    # Dense layers for prediction
    x = layers.Dense(dense_units, activation='relu', name='dense_intermediate')(x)
    outputs = layers.Dense(output_dim, activation='linear', name='output_prediction')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='NBM_LSTM_Prediction')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    model.summary()
    return model


def get_callbacks(model_path: str, patience: int = NBM_EARLY_STOPPING_PATIENCE) -> list:
    """
    Get training callbacks for NBM.
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    callback_list = [
        # Save best model based on validation loss
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        
        # Learning rate reduction on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            mode='min',
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=os.path.join(RESULTS_DIR, 'nbm_logs'),
            histogram_freq=0
        )
    ]
    
    return callback_list


def train_nbm_model(model: Model,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    model_path: str,
                    epochs: int = NBM_EPOCHS,
                    batch_size: int = NBM_BATCH_SIZE) -> keras.callbacks.History:
    """
    Train NBM LSTM Prediction model.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_path: Path to save model
        epochs: Maximum number of epochs
        batch_size: Batch size
        
    Returns:
        Training history
    """
    print("\\nTraining NBM LSTM Prediction Model...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_path),
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss (MSE)', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss (MSE)', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def compute_prediction_errors(model: Model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute prediction errors for NBM.
    
    Args:
        model: Trained model
        X: Input sequences
        y: True next timesteps
        
    Returns:
        Dictionary with prediction errors and statistics
    """
    # Predict
    y_pred = model.predict(X, verbose=0)
    
    # Compute errors
    squared_errors = (y - y_pred) ** 2
    mse_per_sample = np.mean(squared_errors, axis=1)  # Average over features
    mae_per_sample = np.mean(np.abs(y - y_pred), axis=1)
    
    return {
        'y_pred': y_pred,
        'mse_per_sample': mse_per_sample,
        'mae_per_sample': mae_per_sample,
        'mean_mse': np.mean(mse_per_sample),
        'std_mse': np.std(mse_per_sample),
        'mean_mae': np.mean(mae_per_sample),
        'std_mae': np.std(mae_per_sample),
    }


def evaluate_nbm_model(model: Model, 
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate NBM model and compute error distributions.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with errors for each split
    """
    print("\\nEvaluating NBM model...")
    
    errors = {}
    for split_name, X, y in [('train', X_train, y_train), 
                              ('val', X_val, y_val), 
                              ('test', X_test, y_test)]:
        print(f"  {split_name.capitalize()}...")
        errors[split_name] = compute_prediction_errors(model, X, y)
        print(f"    Mean MSE: {errors[split_name]['mean_mse']:.6f} ± {errors[split_name]['std_mse']:.6f}")
        print(f"    Mean MAE: {errors[split_name]['mean_mae']:.6f} ± {errors[split_name]['std_mae']:.6f}")
    
    return errors


def plot_error_distributions(errors: dict, save_path: str):
    """
    Plot error distributions for train/val/test.
    
    Args:
        errors: Dictionary with error statistics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # MSE distribution
    for split_name in ['train', 'val', 'test']:
        mse = errors[split_name]['mse_per_sample']
        axes[0].hist(mse, bins=50, alpha=0.6, label=f'{split_name.capitalize()} (μ={mse.mean():.4f})')
    axes[0].set_xlabel('MSE', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Prediction Error Distribution (MSE)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE distribution
    for split_name in ['train', 'val', 'test']:
        mae = errors[split_name]['mae_per_sample']
        axes[1].hist(mae, bins=50, alpha=0.6, label=f'{split_name.capitalize()} (μ={mae.mean():.4f})')
    axes[1].set_xlabel('MAE', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Prediction Error Distribution (MAE)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error distribution plot saved to: {save_path}")
    plt.close()


def save_results(errors: dict, metadata: dict, output_dir: str):
    """
    Save NBM results and statistics.
    
    Args:
        errors: Dictionary with error statistics
        metadata: NBM metadata
        output_dir: Directory to save results
    """
    results = {
        'model_type': 'NBM_LSTM_Prediction',
        'window_size': metadata['window_size'],
        'n_features': metadata['n_features'],
        'train_events': metadata['train_events'],
        'val_events': metadata['val_events'],
        'test_events': metadata['test_events'],
        'filtering_criteria': metadata['filtering_criteria'],
        'train_error_stats': {
            'mean_mse': float(errors['train']['mean_mse']),
            'std_mse': float(errors['train']['std_mse']),
            'mean_mae': float(errors['train']['mean_mae']),
            'std_mae': float(errors['train']['std_mae']),
        },
        'val_error_stats': {
            'mean_mse': float(errors['val']['mean_mse']),
            'std_mse': float(errors['val']['std_mse']),
            'mean_mae': float(errors['val']['mean_mae']),
            'std_mae': float(errors['val']['std_mae']),
        },
        'test_error_stats': {
            'mean_mse': float(errors['test']['mean_mse']),
            'std_mse': float(errors['test']['std_mse']),
            'mean_mae': float(errors['test']['mean_mae']),
            'std_mae': float(errors['test']['std_mae']),
        },
    }
    
    results_path = os.path.join(output_dir, 'nbm_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Save error arrays for threshold tuning
    errors_path = os.path.join(output_dir, 'nbm_errors.npz')
    np.savez(
        errors_path,
        train_mse=errors['train']['mse_per_sample'],
        val_mse=errors['val']['mse_per_sample'],
        test_mse=errors['test']['mse_per_sample'],
        train_mae=errors['train']['mae_per_sample'],
        val_mae=errors['val']['mae_per_sample'],
        test_mae=errors['test']['mae_per_sample'],
    )
    print(f"Error arrays saved to: {errors_path}")


def main():
    """Main NBM training and evaluation pipeline."""
    print("=" * 70)
    print("NBM LSTM Prediction Model - Training Pipeline")
    print("=" * 70)
    
    # Set seeds for reproducibility
    set_seeds()
    
    # Ensure directories exist
    ensure_dirs()
    nbm_results_dir = os.path.join(RESULTS_DIR, 'NBM')
    os.makedirs(nbm_results_dir, exist_ok=True)
    
    # Load data
    data = load_nbm_data()
    
    # Build model
    window_size = data['X_train'].shape[1]
    n_features = data['X_train'].shape[2]
    
    print(f"\\nBuilding NBM LSTM Prediction Model...")
    model = build_nbm_lstm_model(
        input_shape=(window_size, n_features),
        output_dim=n_features
    )
    
    # Train model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_prediction.keras')
    history = train_nbm_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        model_path
    )
    
    # Plot training history
    history_plot_path = os.path.join(nbm_results_dir, 'nbm_training_history.png')
    plot_training_history(history, history_plot_path)
    
    # Load best model
    print(f"\\nLoading best model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Evaluate model
    errors = evaluate_nbm_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test']
    )
    
    # Plot error distributions
    error_dist_path = os.path.join(nbm_results_dir, 'nbm_error_distributions.png')
    plot_error_distributions(errors, error_dist_path)
    
    # Save results
    save_results(errors, data['metadata'], nbm_results_dir)
    
    print("\\n" + "=" * 70)
    print("NBM Training Complete!")
    print("=" * 70)
    print(f"\\nModel saved to: {model_path}")
    print(f"Results saved to: {nbm_results_dir}")
    print("\\nNext steps:")
    print("  1. Use validation error distribution to set anomaly threshold")
    print("  2. Test on anomaly events to evaluate detection performance")
    print("  3. Tune threshold for optimal precision/recall tradeoff")


if __name__ == "__main__":
    main()
