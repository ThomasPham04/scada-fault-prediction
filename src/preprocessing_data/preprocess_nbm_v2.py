"""
Normal Behavior Model (NBM) - Data Preprocessing Pipeline V2

CORRECT APPROACH per CARE paper methodology:
- Train: train_test=='train' from ALL 22 events (filtered status=0, wind>4, power>0)
- Val: Last 10-20% of train data (temporal split)
- Test: train_test=='prediction' from ALL 22 events (KEEP ALL STATUS - for anomaly evaluation)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WIND_FARM_A_DIR, WIND_FARM_A_DATASETS, WIND_FARM_A_PROCESSED,
    TIME_RESOLUTION, RANDOM_SEED,
    NBM_WINDOW_SIZE, NBM_WINDOW_DAYS, NBM_STRIDE, NBM_CUT_IN_WIND_SPEED, NBM_MIN_POWER,
    NBM_NORMAL_STATUS, NBM_FEATURE_COLUMNS, NBM_ANGLE_FEATURES, 
    NBM_COUNTER_FEATURES, EXCLUDE_COLUMNS,
    ensure_dirs
)


def load_event_info(farm_dir: str) -> pd.DataFrame:
    """Load event information."""
    event_info_path = os.path.join(farm_dir, "event_info.csv")
    df = pd.read_csv(event_info_path, sep=';')
    df['event_start'] = pd.to_datetime(df['event_start'])
    df['event_end'] = pd.to_datetime(df['event_end'])
    
    print(f"Loaded event info: {len(df)} events")
    print(f"  - Anomalies: {(df['event_label'] == 'anomaly').sum()}")
    print(f"  - Normal: {(df['event_label'] == 'normal').sum()}")
    
    return df


def load_event_data(event_id: int, datasets_dir: str) -> pd.DataFrame:
    """Load a single event dataset."""
    file_path = os.path.join(datasets_dir, f"{event_id}.csv")
    df = pd.read_csv(file_path, sep=';')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df


def engineer_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert angle features to sin/cos for continuity."""
    df_copy = df.copy()
    
    for col in NBM_ANGLE_FEATURES:
        if col in df_copy.columns:
            radians = np.radians(df_copy[col])
            df_copy[f'{col}_sin'] = np.sin(radians)
            df_copy[f'{col}_cos'] = np.cos(radians)
            df_copy.drop(col, axis=1, inplace=True)
    
    return df_copy


def drop_counter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop counter features (Wh, VArh)."""
    cols_to_drop = [col for col in NBM_COUNTER_FEATURES if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
    return df


def get_nbm_feature_columns(df: pd.DataFrame) -> list:
    """Get final feature columns after engineering."""
    # Start with base features
    feature_cols = [col for col in NBM_FEATURE_COLUMNS if col in df.columns]
    
    # Add sin/cos engineered features
    for angle_col in NBM_ANGLE_FEATURES:
        sin_col = f'{angle_col}_sin'
        cos_col = f'{angle_col}_cos'
        if sin_col in df.columns:
            feature_cols.append(sin_col)
        if cos_col in df.columns:
            feature_cols.append(cos_col)
    
    # Verify exclusions
    exclude_all = EXCLUDE_COLUMNS + ['status_type_id'] + NBM_ANGLE_FEATURES + NBM_COUNTER_FEATURES
    feature_cols = [col for col in feature_cols if col not in exclude_all]
    
    return feature_cols


def preprocess_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Extract and preprocess feature values."""
    features = df[feature_cols].copy()
    features = features.ffill().bfill()
    features = features.fillna(0)
    return features.values


def process_all_events_train(farm_dir: str, datasets_dir: str, event_info: pd.DataFrame) -> tuple:
    """
    Process TRAIN data from ALL 22 events.
    
    For each event:
    - Take train_test == 'train' portion
    - Filter: status_type_id == 0, wind > 4, power > 0
    - Combine all into one large training dataset
    
    Returns:
        Tuple of (train_data, feature_cols, event_contributions)
    """
    all_train_data = []
    feature_cols = None
    event_contributions = {}
    
    print(f"\n{'='*70}")
    print(f"Processing TRAIN data from ALL {len(event_info)} events...")
    print(f"{'='*70}")
    
    for _, event in tqdm(event_info.iterrows(), total=len(event_info)):
        event_id = event['event_id']
        event_label = event['event_label']
        
        try:
            # Load event data
            df = load_event_data(event_id, datasets_dir)
            
            # Filter for TRAIN portion only
            df_train = df[df['train_test'] == 'train'].copy()
            
            if len(df_train) == 0:
                print(f"  Event {event_id} ({event_label}): No train data")
                continue
            
            # Filter for NORMAL OPERATION (status=0, wind>4, power>0)
            mask = (
                (df_train['status_type_id'] == 0) &
                (df_train['wind_speed_3_avg'] > NBM_CUT_IN_WIND_SPEED) &
                (df_train['power_29_avg'] > NBM_MIN_POWER)
            )
            
            df_normal = df_train[mask].copy()
            
            if len(df_normal) < NBM_WINDOW_SIZE + 1:
                print(f"  Event {event_id} ({event_label}): Insufficient normal data "
                      f"({len(df_normal)} timesteps, need {NBM_WINDOW_SIZE+1})")
                continue
            
            # Feature engineering
            df_normal = engineer_angle_features(df_normal)
            df_normal = drop_counter_features(df_normal)
            
            # Get feature columns (from first event)
            if feature_cols is None:
                feature_cols = get_nbm_feature_columns(df_normal)
                print(f"  Using {len(feature_cols)} features after engineering")
            
            # Preprocess features
            features = preprocess_features(df_normal, feature_cols)
            
            all_train_data.append(features)
            event_contributions[event_id] = {
                'label': event_label,
                'original_train_len': len(df_train),
                'filtered_len': len(df_normal),
                'features_shape': features.shape
            }
            
            print(f"  Event {event_id} ({event_label:7s}): "
                  f"{len(df_train):5d} → {len(df_normal):5d} timesteps "
                  f"({len(df_normal)/len(df_train)*100:.1f}% retained)")
                
        except Exception as e:
            print(f"  Event {event_id}: ERROR - {e}")
            continue
    
    if len(all_train_data) == 0:
        raise ValueError("No training data generated from any event!")
    
    # Concatenate all training data
    combined_train = np.concatenate(all_train_data, axis=0)
    
    print(f"\n{'='*70}")
    print(f"Combined training data: {combined_train.shape[0]} timesteps from {len(all_train_data)} events")
    print(f"{'='*70}")
    
    return combined_train, feature_cols, event_contributions


def process_all_events_test(farm_dir: str, datasets_dir: str, event_info: pd.DataFrame,
                            feature_cols: list) -> tuple:
    """
    Process TEST data from ALL 22 events.
    
    For each event:
    - Take train_test == 'prediction' portion
    - DO NOT filter by status (keep all status types)
    - This is where anomalies occur!
    
    Returns:
        Tuple of (test_data_dict, event_ids)
    """
    test_data_by_event = {}
    
    print(f"\n{'='*70}")
    print(f"Processing TEST (prediction) data from ALL {len(event_info)} events...")
    print(f"  NOTE: Keeping ALL status types (including anomalies)")
    print(f"{'='*70}")
    
    for _, event in tqdm(event_info.iterrows(), total=len(event_info)):
        event_id = event['event_id']
        event_label = event['event_label']
        
        try:
            # Load event data
            df = load_event_data(event_id, datasets_dir)
            
            # Filter for PREDICTION portion only
            df_pred = df[df['train_test'] == 'prediction'].copy()
            
            if len(df_pred) == 0:
                print(f"  Event {event_id} ({event_label}): No prediction data")
                continue
            
            if len(df_pred) < NBM_WINDOW_SIZE + 1:
                print(f"  Event {event_id} ({event_label}): Insufficient prediction data "
                      f"({len(df_pred)} timesteps, need {NBM_WINDOW_SIZE+1})")
                continue
            
            # Feature engineering (same as train)
            df_pred = engineer_angle_features(df_pred)
            df_pred = drop_counter_features(df_pred)
            
            # Preprocess features
            features = preprocess_features(df_pred, feature_cols)
            
            test_data_by_event[event_id] = {
                'features': features,
                'label': event_label,
                'n_timesteps': len(df_pred),
                'event_start': event['event_start'],
                'event_end': event['event_end']
            }
            
            print(f"  Event {event_id} ({event_label:7s}): {len(df_pred)} timesteps")
                
        except Exception as e:
            print(f"  Event {event_id}: ERROR - {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Processed {len(test_data_by_event)} events for testing")
    print(f"{'='*70}")
    
    return test_data_by_event


def temporal_split_train_val(train_data: np.ndarray, val_ratio: float = 0.15) -> tuple:
    """
    Split train data into train/val using temporal split.
    
    Take last val_ratio% of data for validation.
    This is better than random split for time-series.
    
    Args:
        train_data: Combined training data (n_timesteps, n_features)
        val_ratio: Fraction for validation (default 15%)
        
    Returns:
        Tuple of (train_data, val_data)
    """
    n_total = len(train_data)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size
    
    train_split = train_data[:train_size]
    val_split = train_data[train_size:]
    
    print(f"\nTemporal train/val split:")
    print(f"  Train: {len(train_split)} timesteps ({len(train_split)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_split)} timesteps ({len(val_split)/n_total*100:.1f}%)")
    
    return train_split, val_split


def save_raw_splits_as_csv(train_split: np.ndarray, val_split: np.ndarray, 
                           test_data_dict: dict, feature_cols: list, output_dir: str):
    """
    Save train/val/test splits as CSV files BEFORE normalization and sequencing.
    This preserves the raw feature values for analysis.
    
    Args:
        train_split: Training data (n_timesteps, n_features)
        val_split: Validation data (n_timesteps, n_features)
        test_data_dict: Dictionary of test data by event
        feature_cols: List of feature column names
        output_dir: Output directory
    """
    # nbm_dir = os.path.join(output_dir, 'NBM_v2')
    nbm_dir = os.path.join(output_dir, 'NBM_7day')

    os.makedirs(nbm_dir, exist_ok=True)
    csv_dir = os.path.join(nbm_dir, 'csv_splits')
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Saving raw data splits as CSV files...")
    print(f"{'='*70}")
    
    # Save train split
    train_df = pd.DataFrame(train_split, columns=feature_cols)
    train_path = os.path.join(csv_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    print(f"  Train: {train_df.shape} → {train_path}")
    
    # Save validation split
    val_df = pd.DataFrame(val_split, columns=feature_cols)
    val_path = os.path.join(csv_dir, 'val.csv')
    val_df.to_csv(val_path, index=False)
    print(f"  Val:   {val_df.shape} → {val_path}")
    
    # Save test data (combined all events)
    test_data_combined = []
    test_event_ids = []
    test_labels = []
    
    for event_id, data in test_data_dict.items():
        features = data['features']
        n_rows = features.shape[0]
        test_data_combined.append(features)
        test_event_ids.extend([event_id] * n_rows)
        test_labels.extend([data['label']] * n_rows)
    
    if test_data_combined:
        test_combined_array = np.vstack(test_data_combined)
        test_df = pd.DataFrame(test_combined_array, columns=feature_cols)
        test_df.insert(0, 'event_id', test_event_ids)
        test_df.insert(1, 'event_label', test_labels)
        test_path = os.path.join(csv_dir, 'test.csv')
        test_df.to_csv(test_path, index=False)
        print(f"  Test:  {test_df.shape} → {test_path}")
        
        # Also save test data by individual events
        test_by_event_dir = os.path.join(csv_dir, 'test_by_event')
        os.makedirs(test_by_event_dir, exist_ok=True)
        
        for event_id, data in test_data_dict.items():
            event_df = pd.DataFrame(data['features'], columns=feature_cols)
            event_path = os.path.join(test_by_event_dir, f'event_{event_id}.csv')
            event_df.to_csv(event_path, index=False)
        
        print(f"  Test by event: {len(test_data_dict)} files → {test_by_event_dir}")
    
    print(f"\nCSV splits saved to: {csv_dir}")


def create_sequences_nbm(data: np.ndarray, window_size: int, stride: int = 1) -> tuple:
    """
    Create sliding window sequences for NBM LSTM Prediction.
    
    X: sequences of length window_size
    y: next timestep values (target to predict)
    """
    n_timesteps, n_features = data.shape
    n_sequences = (n_timesteps - window_size) // stride
    
    if n_sequences <= 0:
        return np.array([]), np.array([])
    
    X = np.zeros((n_sequences, window_size, n_features), dtype=np.float32)
    y = np.zeros((n_sequences, n_features), dtype=np.float32)
    
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        X[i] = data[start_idx:end_idx]
        y[i] = data[end_idx]
    
    return X, y


def normalize_data_nbm(X_train: np.ndarray, X_val: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray,
                       test_data_dict: dict,
                       output_dir: str) -> tuple:
    """
    Normalize data using StandardScaler fit ONLY on train data.
    
    CRITICAL: Test data from prediction period (with anomalies) should NOT
    influence the scaler. This ensures anomalies will naturally have higher errors.
    """
    n_train, seq_len, n_features = X_train.shape
    n_val = X_val.shape[0]
    
    # Flatten for scaler
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    
    # Scale targets
    y_train_scaled = scaler.transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    
    # Reshape back
    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)
    
    # Scale test data (for each event separately, keep structure)
    test_data_scaled = {}
    for event_id, data in test_data_dict.items():
        features = data['features']
        
        # Create sequences
        X_test, y_test = create_sequences_nbm(features, seq_len, NBM_STRIDE)
        
        if len(X_test) == 0:
            continue
            
        # Scale using same scaler
        X_test_flat = X_test.reshape(-1, n_features)
        X_test_scaled_flat = scaler.transform(X_test_flat)
        X_test_scaled = X_test_scaled_flat.reshape(-1, seq_len, n_features)
        y_test_scaled = scaler.transform(y_test)
        
        test_data_scaled[event_id] = {
            'X': X_test_scaled,
            'y': y_test_scaled,
            'label': data['label'],
            'event_start': data['event_start'],
            'event_end': data['event_end']
        }
    
    # Save scaler
    # scaler_path = os.path.join(output_dir, "nbm_scaler_v2.pkl")
    scaler_path = os.path.join(output_dir, 'NBM_7day', "nbm_scaler_7day.pkl")

    joblib.dump(scaler, scaler_path)
    print(f"\nNBM Scaler saved to: {scaler_path}")
    
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, test_data_scaled


def save_nbm_data_v2(X_train, X_val, y_train, y_val, test_data_scaled,
                     feature_cols, event_contributions, output_dir):
    """Save NBM V2 preprocessed data."""
    ensure_dirs()
    
    # nbm_dir = os.path.join(output_dir, 'NBM_v2')
    nbm_dir = os.path.join(output_dir, 'NBM_7day')
    os.makedirs(nbm_dir, exist_ok=True)
    
    # Save train/val arrays
    np.save(os.path.join(nbm_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(nbm_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(nbm_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(nbm_dir, 'y_val.npy'), y_val)
    
    print(f"\nSaved training data:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    
    # Save test data (by event)
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    os.makedirs(test_dir, exist_ok=True)
    
    for event_id, data in test_data_scaled.items():
        event_file = os.path.join(test_dir, f'event_{event_id}.npz')
        np.savez(
            event_file,
            X=data['X'],
            y=data['y'],
            label=data['label'],
            event_start=str(data['event_start']),
            event_end=str(data['event_end'])
        )
        print(f"  Event {event_id} ({data['label']:7s}): X={data['X'].shape}, y={data['y'].shape}")
    
    # Save metadata
    metadata = {
        'version': 2,
        'window_size': NBM_WINDOW_SIZE,
        'stride': NBM_STRIDE,
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'event_contributions': event_contributions,
        'test_events': {k: v['label'] for k, v in test_data_scaled.items()},
        'filtering_criteria': {
            'status_type': NBM_NORMAL_STATUS,
            'min_wind_speed': NBM_CUT_IN_WIND_SPEED,
            'min_power': NBM_MIN_POWER,
        },
        'split_strategy': 'temporal_train_val_split_plus_prediction_test'
    }
    
    # metadata_path = os.path.join(nbm_dir, "nbm_metadata_v2.pkl")
    metadata_path = os.path.join(nbm_dir, "nbm_metadata_7day.pkl")

    joblib.dump(metadata, metadata_path)
    print(f"\nMetadata saved to: {metadata_path}")


def main():
    """Main NBM V2 preprocessing pipeline."""
    print("=" * 70)
    print("Normal Behavior Model (NBM) V2 - Data Preprocessing Pipeline")
    print("=" * 70)
    print("\nSTRATEGY:")
    print("  • Train: train_test=='train' from ALL 22 events (status=0 filter)")
    print("  • Val:   Last 15% of train data (temporal split)")
    print("  • Test:  train_test=='prediction' from ALL 22 events (ALL status)")
    print("=" * 70)
    
    # Load event info
    event_info = load_event_info(WIND_FARM_A_DIR)
    
    # Process train data from all events
    train_data, feature_cols, event_contributions = process_all_events_train(
        WIND_FARM_A_DIR,
        WIND_FARM_A_DATASETS,
        event_info
    )
    
    # Temporal split for train/val
    train_split, val_split = temporal_split_train_val(train_data, val_ratio=0.15)
    
    # Create sequences for train and val
    print(f"\nCreating sequences (window={NBM_WINDOW_SIZE}, stride={NBM_STRIDE})...")
    X_train, y_train = create_sequences_nbm(train_split, NBM_WINDOW_SIZE, NBM_STRIDE)
    X_val, y_val = create_sequences_nbm(val_split, NBM_WINDOW_SIZE, NBM_STRIDE)
    
    print(f"  Train sequences: {len(X_train)}")
    print(f"  Val sequences: {len(X_val)}")
    
    # Process test data (prediction periods) from all events
    test_data_dict = process_all_events_test(
        WIND_FARM_A_DIR,
        WIND_FARM_A_DATASETS,
        event_info,
        feature_cols
    )
    
    # Save raw splits as CSV (before normalization and sequencing)
    save_raw_splits_as_csv(
        train_split, val_split, test_data_dict,
        feature_cols, WIND_FARM_A_PROCESSED
    )
    
    # Normalize data
    print(f"\nNormalizing data (fit on train only)...")
    X_train_norm, X_val_norm, y_train_norm, y_val_norm, test_data_scaled = normalize_data_nbm(
        X_train, X_val, y_train, y_val,
        test_data_dict,
        WIND_FARM_A_PROCESSED
    )
    
    # Save processed data
    save_nbm_data_v2(
        X_train_norm, X_val_norm, y_train_norm, y_val_norm,
        test_data_scaled, feature_cols, event_contributions,
        WIND_FARM_A_PROCESSED
    )
    
    print("\n" + "=" * 70)
    print("NBM V2 Preprocessing Complete!")
    print("=" * 70)
    # print("\nData saved to: Dataset/processed/Wind Farm A/NBM_v2/")
    print("\nData saved to: Dataset/processed/Wind Farm A/NBM_7day/")

    print(f"  • Train/Val: Combined arrays")
    print(f"  • Test: Per-event files (for anomaly evaluation)")


if __name__ == "__main__":
    main()
