"""
Normal Behavior Model (NBM) - Data Preprocessing Pipeline

This module implements preprocessing for NBM approach:
- Filter NORMAL operation data only (status_type=0, wind>cut-in, power>0)
- Feature engineering (sin/cos for angles, drop counters)
- Normalization fitted on normal data only
- Create 14-day sequences for LSTM prediction
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WIND_FARM_A_DIR, WIND_FARM_A_DATASETS, WIND_FARM_A_PROCESSED,
    TIME_RESOLUTION, RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    NBM_WINDOW_SIZE, NBM_WINDOW_DAYS, NBM_STRIDE, NBM_CUT_IN_WIND_SPEED, NBM_MIN_POWER,
    NBM_NORMAL_STATUS, NBM_FEATURE_COLUMNS, NBM_ANGLE_FEATURES, 
    NBM_COUNTER_FEATURES, EXCLUDE_COLUMNS,
    ensure_dirs
)


def load_event_info(farm_dir: str) -> pd.DataFrame:
    """
    Load event information containing labels and fault timestamps.
    
    Args:
        farm_dir: Path to wind farm directory
        
    Returns:
        DataFrame with event_id, event_label, event_start, event_end, etc.
    """
    event_info_path = os.path.join(farm_dir, "event_info.csv")
    df = pd.read_csv(event_info_path, sep=';')
    
    # Convert timestamps
    df['event_start'] = pd.to_datetime(df['event_start'])
    df['event_end'] = pd.to_datetime(df['event_end'])
    
    print(f"Loaded event info: {len(df)} events")
    print(f"  - Anomalies: {(df['event_label'] == 'anomaly').sum()}")
    print(f"  - Normal: {(df['event_label'] == 'normal').sum()}")
    
    return df


def load_event_data(event_id: int, datasets_dir: str) -> pd.DataFrame:
    """
    Load a single event dataset.
    
    Args:
        event_id: ID of the event to load
        datasets_dir: Path to datasets directory
        
    Returns:
        DataFrame with sensor data and timestamps
    """
    file_path = os.path.join(datasets_dir, f"{event_id}.csv")
    df = pd.read_csv(file_path, sep=';')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df


def filter_normal_operation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to keep ONLY normal operation periods.
    
    Criteria (AND conditions):
    - train_test == 'train' (training period only)
    - status_type == 0 (normal operation)
    - wind_speed_3_avg > cut-in speed
    - power_29_avg > 0 (generating power)
    
    Args:
        df: Raw event data
        
    Returns:
        Filtered DataFrame with normal operation only
    """
    original_len = len(df)
    
    # Apply filters
    mask = (
        (df['train_test'] == 'train') &
        (df['status_type_id'].isin(NBM_NORMAL_STATUS)) &
        (df['wind_speed_3_avg'] > NBM_CUT_IN_WIND_SPEED) &
        (df['power_29_avg'] > NBM_MIN_POWER)
    )
    
    df_filtered = df[mask].copy()
    filtered_len = len(df_filtered)
    
    print(f"    Filtered: {original_len} → {filtered_len} timesteps "
          f"({filtered_len/original_len*100:.1f}% retained)")
    
    return df_filtered


def engineer_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert angle features to sin/cos for continuity.
    
    Angles are cyclic (0° ≈ 360°), so we convert to:
    - sin(angle) and cos(angle)
    
    Args:
        df: DataFrame with angle columns
        
    Returns:
        DataFrame with angle columns replaced by sin/cos
    """
    df_copy = df.copy()
    
    for col in NBM_ANGLE_FEATURES:
        if col in df_copy.columns:
            # Convert to radians and compute sin/cos
            radians = np.radians(df_copy[col])
            df_copy[f'{col}_sin'] = np.sin(radians)
            df_copy[f'{col}_cos'] = np.cos(radians)
            # Drop original angle column
            df_copy.drop(col, axis=1, inplace=True)
    
    return df_copy


def drop_counter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop counter features (Wh, VArh) as they are cumulative and not useful for NBM.
    
    Args:
        df: DataFrame with counter columns
        
    Returns:
        DataFrame without counter columns
    """
    cols_to_drop = [col for col in NBM_COUNTER_FEATURES if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
        print(f"    Dropped {len(cols_to_drop)} counter features")
    return df


def get_nbm_feature_columns(df: pd.DataFrame) -> list:
    """
    Get final feature columns after engineering.
    
    Includes:
    - NBM_FEATURE_COLUMNS (temp, RPM, electrical, wind/power)
    - Engineered sin/cos features
    
    Excludes:
    - EXCLUDE_COLUMNS (time_stamp, asset_id, id, train_test)
    - status_type (used for filtering only)
    - Original angle features (replaced by sin/cos)
    - Counter features (dropped)
    
    Args:
        df: Processed DataFrame
        
    Returns:
        List of feature column names
    """
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
    """
    Extract and preprocess feature values.
    
    - Handle missing values using forward fill, then backward fill
    - Convert to numpy array
    
    Args:
        df: DataFrame with sensor data
        feature_cols: List of feature column names
        
    Returns:
        Numpy array of shape (n_timesteps, n_features)
    """
    features = df[feature_cols].copy()
    
    # Forward fill then backward fill for missing values
    features = features.ffill().bfill()
    
    # Fill any remaining NaN with 0 (shouldn't happen but just in case)
    features = features.fillna(0)
    
    return features.values


def create_sequences_nbm(data: np.ndarray, window_size: int, stride: int = 1) -> tuple:
    """
    Create sliding window sequences for NBM LSTM Prediction.
    
    For prediction model:
    - X: sequences of length window_size (input to predict from)
    - y: next timestep values (target to predict)
    
    Args:
        data: Feature array of shape (n_timesteps, n_features)
        window_size: Length of input sequence (2016 for 14 days)
        stride: Step size between sequences
        
    Returns:
        Tuple of (X_sequences, y_targets) where:
            - X_sequences: shape (n_sequences, window_size, n_features)
            - y_targets: shape (n_sequences, n_features) - next timestep
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
        
        # Input: window_size timesteps
        X[i] = data[start_idx:end_idx]
        
        # Target: next timestep after the window
        y[i] = data[end_idx]
    
    return X, y


def process_normal_events(farm_dir: str, datasets_dir: str, event_info: pd.DataFrame,
                         window_size: int, stride: int) -> tuple:
    """
    Process only NORMAL events for NBM training.
    
    Args:
        farm_dir: Path to wind farm directory
        datasets_dir: Path to datasets directory
        event_info: Event information DataFrame
        window_size: Sequence window size
        stride: Sequence stride
        
    Returns:
        Tuple of (X_all, y_all, feature_cols, event_ids)
    """
    # Filter to normal events only
    normal_events = event_info[event_info['event_label'] == 'normal']
    
    X_all = []
    y_all = []
    event_ids = []
    feature_cols = None
    
    print(f"\nProcessing {len(normal_events)} NORMAL events for NBM training...")
    
    for _, event in tqdm(normal_events.iterrows(), total=len(normal_events)):
        event_id = event['event_id']
        
        try:
            # Load event data
            df = load_event_data(event_id, datasets_dir)
            
            # Filter normal operation periods
            df_normal = filter_normal_operation(df)
            
            if len(df_normal) < window_size + 1:
                print(f"    Event {event_id}: Insufficient data after filtering (need {window_size+1}, got {len(df_normal)})")
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
            
            # Create sequences
            X, y = create_sequences_nbm(features, window_size, stride)
            
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                event_ids.append(event_id)
                print(f"    Event {event_id}: {len(X)} sequences created")
                
        except Exception as e:
            print(f"  Error processing event {event_id}: {e}")
            continue
    
    return X_all, y_all, feature_cols, event_ids


def split_by_events_nbm(X_all: list, y_all: list, event_ids: list,
                        test_size: float = 0.2, val_size: float = 0.2,
                        random_seed: int = 42) -> dict:
    """
    Split normal events into train/val/test sets.
    
    Args:
        X_all: List of X sequence arrays per event
        y_all: List of y target arrays per event
        event_ids: List of event IDs
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_seed: Random seed
        
    Returns:
        Dictionary with splits
    """
    n_events = len(X_all)
    event_indices = list(range(n_events))
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        event_indices, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_seed
    )
    
    # Concatenate sequences for each split
    X_train = np.concatenate([X_all[i] for i in train_idx], axis=0)
    X_val = np.concatenate([X_all[i] for i in val_idx], axis=0)
    X_test = np.concatenate([X_all[i] for i in test_idx], axis=0)
    
    y_train = np.concatenate([y_all[i] for i in train_idx], axis=0)
    y_val = np.concatenate([y_all[i] for i in val_idx], axis=0)
    y_test = np.concatenate([y_all[i] for i in test_idx], axis=0)
    
    print(f"\nNBM Data split (Normal events only):")
    print(f"  Train: {len(X_train)} sequences from {len(train_idx)} events")
    print(f"  Val:   {len(X_val)} sequences from {len(val_idx)} events")
    print(f"  Test:  {len(X_test)} sequences from {len(test_idx)} events")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_events': [event_ids[i] for i in train_idx],
        'val_events': [event_ids[i] for i in val_idx],
        'test_events': [event_ids[i] for i in test_idx],
    }


def normalize_data_nbm(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                       output_dir: str) -> tuple:
    """
    Normalize data using StandardScaler fit ONLY on train-normal data.
    
    CRITICAL: NBM requires scaler fitted on normal data only.
    This ensures anomalies will naturally have higher prediction errors.
    
    Args:
        X_train, X_val, X_test: Input sequences
        y_train, y_val, y_test: Target sequences
        output_dir: Directory to save scaler
        
    Returns:
        Tuple of normalized data
    """
    n_train, seq_len, n_features = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    
    # Reshape for scaler: combine all timesteps from training sequences
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Also scale targets
    y_train_scaled = scaler.transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)
    
    # Reshape back to sequences
    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "nbm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\nNBM Scaler saved to: {scaler_path}")
    print(f"  Mean: {scaler.mean_[:5]}... (showing first 5)")
    print(f"  Std:  {scaler.scale_[:5]}... (showing first 5)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled


def save_nbm_data(data_dict: dict, output_dir: str, feature_cols: list):
    """
    Save NBM preprocessed data.
    
    Args:
        data_dict: Dictionary with data arrays
        output_dir: Directory to save files
        feature_cols: List of feature column names
    """
    ensure_dirs()
    
    # Create NBM subdirectory
    nbm_dir = os.path.join(output_dir, 'NBM')
    os.makedirs(nbm_dir, exist_ok=True)
    
    # Save arrays
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        file_path = os.path.join(nbm_dir, f"{key}.npy")
        np.save(file_path, data_dict[key])
        print(f"Saved {key}: shape {data_dict[key].shape}")
    
    # Save metadata
    metadata = {
        'window_size': NBM_WINDOW_SIZE,
        'stride': NBM_STRIDE,
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'train_events': data_dict['train_events'],
        'val_events': data_dict['val_events'],
        'test_events': data_dict['test_events'],
        'filtering_criteria': {
            'status_type': NBM_NORMAL_STATUS,
            'min_wind_speed': NBM_CUT_IN_WIND_SPEED,
            'min_power': NBM_MIN_POWER,
        }
    }
    metadata_path = os.path.join(nbm_dir, "nbm_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"NBM Metadata saved to: {metadata_path}")


def main():
    """Main NBM preprocessing pipeline."""
    print("=" * 70)
    print("Normal Behavior Model (NBM) - Data Preprocessing Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Window Size: {NBM_WINDOW_SIZE} timesteps ({NBM_WINDOW_DAYS} days)")
    print(f"  Stride: {NBM_STRIDE} timesteps ({NBM_STRIDE * TIME_RESOLUTION / 60:.1f} hours)")
    print(f"  Normal Status: {NBM_NORMAL_STATUS}")
    print(f"  Min Wind Speed: {NBM_CUT_IN_WIND_SPEED} m/s")
    print(f"  Min Power: {NBM_MIN_POWER} kW")
    
    # Load event info
    event_info = load_event_info(WIND_FARM_A_DIR)
    
    # Process normal events only
    X_all, y_all, feature_cols, event_ids = process_normal_events(
        WIND_FARM_A_DIR,
        WIND_FARM_A_DATASETS,
        event_info,
        NBM_WINDOW_SIZE,
        NBM_STRIDE
    )
    
    if len(X_all) == 0:
        print("\nERROR: No sequences created. Check filtering criteria.")
        return
    
    # Split by events
    data_dict = split_by_events_nbm(
        X_all, y_all, event_ids,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED
    )
    
    # Normalize data
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = normalize_data_nbm(
        data_dict['X_train'], data_dict['X_val'], data_dict['X_test'],
        data_dict['y_train'], data_dict['y_val'], data_dict['y_test'],
        WIND_FARM_A_PROCESSED
    )
    
    # Update with normalized data
    data_dict['X_train'] = X_train_norm
    data_dict['X_val'] = X_val_norm
    data_dict['X_test'] = X_test_norm
    data_dict['y_train'] = y_train_norm
    data_dict['y_val'] = y_val_norm
    data_dict['y_test'] = y_test_norm
    
    # Save processed data
    save_nbm_data(data_dict, WIND_FARM_A_PROCESSED, feature_cols)
    
    print("\n" + "=" * 70)
    print("NBM Preprocessing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
