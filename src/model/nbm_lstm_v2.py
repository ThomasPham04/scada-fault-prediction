"""
Normal Behavior Model (NBM) - LSTM Prediction Model V2

Train on NBM_v2 data (from all 22 events)
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

# Add parent directory to path
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


def load_nbm_v2_data():
    """Load NBM V2 preprocessed data."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    
    print("Loading NBM V2 preprocessed data...")
    X_train = np.load(os.path.join(nbm_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(nbm_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(nbm_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(nbm_dir, 'y_val.npy'))
    
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_v2.pkl'))
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Window: {metadata['window_size']} timesteps")
    
    return {
        'X_train': X_train, 'X_val': X_val,
        'y_train': y_train, 'y_val': y_val,
        'metadata': metadata
    }


def build_nbm_lstm_model(input_shape: tuple, output_dim: int) -> Model:
    """Build LSTM Prediction Model for NBM."""
    inputs = layers.Input(shape=input_shape, name='input_sequence')
    
    x = layers.LSTM(NBM_LSTM_UNITS_1, return_sequences=True, name='lstm_1')(inputs)
    x = layers.Dropout(NBM_DROPOUT_RATE, name='dropout_1')(x)
    
    x = layers.LSTM(NBM_LSTM_UNITS_2, return_sequences=False, name='lstm_2')(x)
    x = layers.Dropout(NBM_DROPOUT_RATE, name='dropout_2')(x)
    
    x = layers.Dense(NBM_DENSE_UNITS, activation='relu', name='dense_intermediate')(x)
    outputs = layers.Dense(output_dim, activation='linear', name='output_prediction')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='NBM_LSTM_Prediction_V2')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NBM_LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    return model


def get_callbacks(model_path: str) -> list:
    """Get training callbacks."""
    return [
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=NBM_EARLY_STOPPING_PATIENCE,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            mode='min',
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(RESULTS_DIR, 'nbm_v2_logs'),
            histogram_freq=0
        )
    ]


def train_nbm_model(model: Model, X_train, y_train, X_val, y_val, model_path: str):
    """Train NBM model."""
    print("\nTraining NBM LSTM Prediction Model V2...")
    print(f"  Epochs: {NBM_EPOCHS}")
    print(f"  Batch size: {NBM_BATCH_SIZE}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NBM_EPOCHS,
        batch_size=NBM_BATCH_SIZE,
        callbacks=get_callbacks(model_path),
        verbose=1
    )
    
    return history


def plot_training_history(history, save_path: str):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    plt.close()


def evaluate_model(model: Model, X_train, y_train, X_val, y_val):
    """Evaluate model and compute errors."""
    print("\nEvaluating model...")
    
    errors = {}
    for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val)]:
        y_pred = model.predict(X, verbose=0)
        mse_per_sample = np.mean((y - y_pred) ** 2, axis=1)
        mae_per_sample = np.mean(np.abs(y - y_pred), axis=1)
        
        errors[name] = {
            'mse_per_sample': mse_per_sample,
            'mae_per_sample': mae_per_sample,
            'mean_mse': np.mean(mse_per_sample),
            'std_mse': np.std(mse_per_sample),
            'mean_mae': np.mean(mae_per_sample),
            'std_mae': np.std(mae_per_sample),
        }
        
        print(f"  {name.capitalize()}:")
        print(f"    MSE: {errors[name]['mean_mse']:.6f} ± {errors[name]['std_mse']:.6f}")
        print(f"    MAE: {errors[name]['mean_mae']:.6f} ± {errors[name]['std_mae']:.6f}")
    
    return errors


def plot_error_distributions(errors: dict, save_path: str):
    """Plot error distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for split in ['train', 'val']:
        mse = errors[split]['mse_per_sample']
        axes[0].hist(mse, bins=50, alpha=0.6, label=f'{split.capitalize()} (μ={mse.mean():.4f})')
    axes[0].set_xlabel('MSE', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Prediction Error Distribution (MSE)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for split in ['train', 'val']:
        mae = errors[split]['mae_per_sample']
        axes[1].hist(mae, bins=50, alpha=0.6, label=f'{split.capitalize()} (μ={mae.mean():.4f})')
    axes[1].set_xlabel('MAE', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Prediction Error Distribution (MAE)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error distributions saved to: {save_path}")
    plt.close()


def save_results(errors: dict, metadata: dict, output_dir: str):
    """Save results."""
    results = {
        'version': 2,
        'model_type': 'NBM_LSTM_Prediction_V2',
        'window_size': metadata['window_size'],
        'n_features': metadata['n_features'],
        'split_strategy': metadata['split_strategy'],
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
    }
    
    results_path = os.path.join(output_dir, 'nbm_v2_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Save errors for threshold tuning
    errors_path = os.path.join(output_dir, 'nbm_v2_errors.npz')
    np.savez(
        errors_path,
        train_mse=errors['train']['mse_per_sample'],
        val_mse=errors['val']['mse_per_sample'],
        train_mae=errors['train']['mae_per_sample'],
        val_mae=errors['val']['mae_per_sample'],
    )
    print(f"Error arrays saved to: {errors_path}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("NBM LSTM Prediction Model V2 - Training Pipeline")
    print("=" * 70)
    
    set_seeds()
    ensure_dirs()
    
    results_dir = os.path.join(RESULTS_DIR, 'NBM_v2')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data = load_nbm_v2_data()
    
    # Build model
    window_size = data['X_train'].shape[1]
    n_features = data['X_train'].shape[2]
    
    print(f"\nBuilding NBM LSTM Model...")
    model = build_nbm_lstm_model(
        input_shape=(window_size, n_features),
        output_dim=n_features
    )
    
    # Train
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_v2.keras')
    history = train_nbm_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        model_path
    )
    
    # Plot history
    history_plot = os.path.join(results_dir, 'training_history.png')
    plot_training_history(history, history_plot)
    
    # Load best model
    print(f"\nLoading best model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Evaluate
    errors = evaluate_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Plot errors
    error_plot = os.path.join(results_dir, 'error_distributions.png')
    plot_error_distributions(errors, error_plot)
    
    # Save results
    save_results(errors, data['metadata'], results_dir)
    
    print("\n" + "=" * 70)
    print("NBM V2 Training Complete!")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Results: {results_dir}")
    print("\nNext steps:")
    print("  1. Set anomaly threshold from validation errors")
    print("  2. Test on 13 prediction events (4 anomaly, 9 normal)")
    print("  3. Evaluate anomaly detection performance")


if __name__ == "__main__":
    main()
