"""
IO Utilities — data_pipeline.utils.io
Save processed data splits (raw CSV and NPZ) to disk.
"""

import os
import numpy as np
import pandas as pd
import joblib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import NBM_WINDOW_SIZE, NBM_STRIDE, NBM_NORMAL_STATUS, NBM_CUT_IN_WIND_SPEED, NBM_MIN_POWER, ensure_dirs


def save_raw_splits_as_csv(
    train_split: np.ndarray,
    val_split: np.ndarray,
    test_data_dict: dict,
    feature_cols: list,
    output_dir: str,
) -> None:
    """
    Save train/val/test splits as CSV files BEFORE normalization.
    Preserves raw feature values for exploratory analysis.

    Args:
        train_split: (n_timesteps, n_features)
        val_split: (n_timesteps, n_features)
        test_data_dict: Dict from process_all_events_test()
        feature_cols: Feature column names
        output_dir: Root processed data directory
    """
    nbm_dir = os.path.join(output_dir, 'NBM_7day')
    os.makedirs(nbm_dir, exist_ok=True)
    csv_dir = os.path.join(nbm_dir, 'csv_splits')
    os.makedirs(csv_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("Saving raw data splits as CSV files...")
    print(f"{'='*70}")

    # Train
    train_df = pd.DataFrame(train_split, columns=feature_cols)
    train_path = os.path.join(csv_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    print(f"  Train: {train_df.shape} → {train_path}")

    # Val
    val_df = pd.DataFrame(val_split, columns=feature_cols)
    val_path = os.path.join(csv_dir, 'val.csv')
    val_df.to_csv(val_path, index=False)
    print(f"  Val:   {val_df.shape} → {val_path}")

    # Test (combined)
    test_data_combined, test_event_ids, test_labels = [], [], []
    for event_id, data in test_data_dict.items():
        features = data['features']
        n_rows = features.shape[0]
        test_data_combined.append(features)
        test_event_ids.extend([event_id] * n_rows)
        test_labels.extend([data['label']] * n_rows)

    if test_data_combined:
        combined_arr = np.vstack(test_data_combined)
        test_df = pd.DataFrame(combined_arr, columns=feature_cols)
        test_df.insert(0, 'event_id', test_event_ids)
        test_df.insert(1, 'event_label', test_labels)
        test_path = os.path.join(csv_dir, 'test.csv')
        test_df.to_csv(test_path, index=False)
        print(f"  Test:  {test_df.shape} → {test_path}")

        # Per-event CSVs
        event_csv_dir = os.path.join(csv_dir, 'test_by_event')
        os.makedirs(event_csv_dir, exist_ok=True)
        for event_id, data in test_data_dict.items():
            event_df = pd.DataFrame(data['features'], columns=feature_cols)
            event_df.to_csv(os.path.join(event_csv_dir, f'event_{event_id}.csv'), index=False)
        print(f"  Test by event: {len(test_data_dict)} files → {event_csv_dir}")

    print(f"\nCSV splits saved to: {csv_dir}")


def save_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    test_data_scaled: dict,
    feature_cols: list,
    event_contributions: dict,
    output_dir: str,
) -> None:
    """
    Save all scaled NBM data to disk in NPZ/PKL format.

    Structure:
      NBM_7day/
        X_train.npy, X_val.npy, y_train.npy, y_val.npy
        train_by_event/event_*.npz   (10 pseudo-events for IQR calibration)
        val_by_event/event_*.npz     (6 pseudo-events: 2 normal + 4 noisy-anomaly)
        test_by_event/event_*.npz    (one file per real event)
        nbm_metadata_7day.pkl
    """
    ensure_dirs()
    data_dir = os.path.join(output_dir, '7day')
    os.makedirs(data_dir, exist_ok=True)

    # Train/val arrays
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
    print(f"\nSaved training data:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")

    # Train pseudo-events (for IQR threshold calibration)
    train_dir = os.path.join(data_dir, 'train_by_event')
    os.makedirs(train_dir, exist_ok=True)
    n_train_events = 10
    chunk_size = max(1, len(X_train) // n_train_events)
    train_events_meta = {}
    for i in range(n_train_events):
        start = i * chunk_size
        end = start + chunk_size if i < n_train_events - 1 else len(X_train)
        tid = 1000 + i
        np.savez(os.path.join(train_dir, f'event_{tid}.npz'),
                 X=X_train[start:end], y=y_train[start:end], label='normal')
        train_events_meta[tid] = 'normal'
        print(f"  Train event {tid}: X={X_train[start:end].shape}")

    # Real test events
    test_dir = os.path.join(data_dir, 'test_by_event')
    os.makedirs(test_dir, exist_ok=True)
    for event_id, data in test_data_scaled.items():
        np.savez(
            os.path.join(test_dir, f'event_{event_id}.npz'),
            X=data['X'], y=data['y'],
            label=data['label'],
            event_start=str(data['event_start']),
            event_end=str(data['event_end']),
        )
        print(f"  Event {event_id} ({data['label']:7s}): X={data['X'].shape}")

    # Val pseudo-events (2 normal + 4 anomaly-level)
    val_dir = os.path.join(data_dir, 'val_by_event')
    os.makedirs(val_dir, exist_ok=True)
    chunk_size_val = max(1, len(X_val) // 6)
    val_events_meta = {}
    np.random.seed(42)
    for i in range(6):
        start = i * chunk_size_val
        end = start + chunk_size_val if i < 5 else len(X_val)
        X_chunk = X_val[start:end].copy()
        y_chunk = y_val[start:end]

        if i == 0:
            label, noise_std = 'normal', 0.0
        elif i == 1:
            label, noise_std = 'normal', 0.1
        else:
            label, noise_std = 'anomaly', 0.2 + (i - 1) * 0.05

        X_chunk += np.random.normal(0, noise_std, X_chunk.shape)
        val_id = 999 + i
        np.savez(os.path.join(val_dir, f'event_{val_id}.npz'), X=X_chunk, y=y_chunk, label=label)
        val_events_meta[val_id] = label
        print(f"  Val pseudo-event {val_id} ({label}, noise={noise_std}): X={X_chunk.shape}")

    # Metadata
    metadata = {
        'version': 2,
        'window_size': NBM_WINDOW_SIZE,
        'stride': NBM_STRIDE,
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'event_contributions': event_contributions,
        'test_events': {k: v['label'] for k, v in test_data_scaled.items()},
        'val_events': val_events_meta,
        'filtering_criteria': {
            'status_type': NBM_NORMAL_STATUS,
            'min_wind_speed': NBM_CUT_IN_WIND_SPEED,
            'min_power': NBM_MIN_POWER,
        },
        'split_strategy': 'temporal_train_val_split_plus_prediction_test',
    }
    meta_path = os.path.join(data_dir, 'metadata_7day.pkl')
    joblib.dump(metadata, meta_path)
    print(f"\nMetadata saved to: {meta_path}")
