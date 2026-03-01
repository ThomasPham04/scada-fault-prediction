"""
Normalizer — data_pipeline.preprocessing.normalizer
StandardScaler normalization fitted exclusively on training data.
"""

import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from sklearn.preprocessing import StandardScaler
from config import NBM_STRIDE
from data_pipeline.loaders.sequence_maker import create_sequences


def normalize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    test_data_dict: dict,
    output_dir: str,
) -> tuple:
    """
    Normalize all data splits using a StandardScaler fit ONLY on training data.

    CRITICAL: The scaler must NOT see any test/anomaly data. This ensures
    anomalous samples produce naturally higher reconstruction errors.

    Args:
        X_train: Train sequences (n_seq, window, features)
        X_val: Val sequences (n_seq, window, features)
        y_train: Train targets (n_seq, features)
        y_val: Val targets (n_seq, features)
        test_data_dict: Raw feature arrays per event from process_all_events_test()
        output_dir: Directory to save the fitted scaler

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled,
                  test_data_scaled_dict)
    """
    n_train, seq_len, n_features = X_train.shape
    n_val = X_val.shape[0]

    # Flatten for sklearn scaler
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)

    # Fit ONLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)

    y_train_scaled = scaler.transform(y_train)
    y_val_scaled = scaler.transform(y_val)

    # Reshape back to sequences
    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)

    # Scale test events (sequence them first, then transform)
    test_data_scaled = {}
    for event_id, data in test_data_dict.items():
        features = data['features']
        X_test, y_test = create_sequences(features, seq_len, NBM_STRIDE)

        if len(X_test) == 0:
            continue

        X_test_flat = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.transform(X_test_flat).reshape(-1, seq_len, n_features)
        y_test_scaled = scaler.transform(y_test)

        test_data_scaled[event_id] = {
            'X': X_test_scaled,
            'y': y_test_scaled,
            'label': data['label'],
            'event_start': data['event_start'],
            'event_end': data['event_end'],
        }

    # Save scaler for inference reuse
    scaler_dir = os.path.join(output_dir, '7day')
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, 'scaler_7day.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\nNBM Scaler saved to: {scaler_path}")

    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, test_data_scaled
