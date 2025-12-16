"""
Data Preprocessing Pipeline for SCADA Fault Prediction.

This module handles:
- Loading raw SCADA data from Wind Farm A
- Merging with event labels
- Handling missing values
- Normalizing features
- Creating sequences for LSTM training
- Splitting into train/validation/test sets
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
    SEQUENCE_LENGTH, EXCLUDE_COLUMNS, RANDOM_SEED, TEST_SIZE, VAL_SIZE,
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


def create_binary_labels(df: pd.DataFrame, event_info: pd.Series, 
                         prediction_window: int) -> np.ndarray:
    """
    Create binary labels for each timestep.
    
    For anomaly events: 
        - Label = 1 if timestep is within prediction_window of event_start
        - Label = 0 otherwise
    
    For normal events:
        - Label = 0 for all timesteps (no fault to predict)
    
    Args:
        df: DataFrame with timestamps
        event_info: Series with event_label, event_start, etc.
        prediction_window: Number of timesteps to look ahead
        
    Returns:
        Binary labels array (1 = fault within window, 0 = normal)
    """
    n_samples = len(df)
    labels = np.zeros(n_samples, dtype=np.int32)
    
    if event_info['event_label'] == 'anomaly':
        event_start = event_info['event_start']
        
        # Find the index where the event starts
        # We want to label timesteps where a fault occurs within the next prediction_window
        for i in range(n_samples):
            current_time = df.iloc[i]['time_stamp']
            # If the event starts within the prediction window from current time
            time_to_event = (event_start - current_time).total_seconds() / 60  # in minutes
            timesteps_to_event = time_to_event / 10  # 10-minute intervals
            
            if 0 <= timesteps_to_event <= prediction_window:
                labels[i] = 1
    
    return labels


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns (exclude non-sensor columns).
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        List of feature column names
    """
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
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


def create_sequences(data: np.ndarray, labels: np.ndarray, 
                     sequence_length: int, stride: int = 1) -> tuple:
    """
    Create sliding window sequences for LSTM training.
    
    Args:
        data: Feature array of shape (n_timesteps, n_features)
        labels: Label array of shape (n_timesteps,)
        sequence_length: Length of each sequence (288 for 48 hours)
        stride: Step size between sequences (default: 1 for maximum overlap)
        
    Returns:
        Tuple of (X_sequences, y_labels) where:
            - X_sequences: shape (n_sequences, sequence_length, n_features)
            - y_labels: shape (n_sequences,) - label at the END of each sequence
    """
    n_timesteps, n_features = data.shape
    n_sequences = (n_timesteps - sequence_length) // stride + 1
    
    if n_sequences <= 0:
        return np.array([]), np.array([])
    
    X = np.zeros((n_sequences, sequence_length, n_features), dtype=np.float32)
    y = np.zeros(n_sequences, dtype=np.int32)
    
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        X[i] = data[start_idx:end_idx]
        # Label is for the END of the sequence (will fault occur in next 48h?)
        y[i] = labels[end_idx - 1]
    
    return X, y


def process_all_events(farm_dir: str, datasets_dir: str, 
                       sequence_length: int, 
                       anomaly_stride: int = 1,
                       normal_stride: int = 50) -> tuple:
    """
    Process all events from a wind farm with stratified sampling.
    
    Uses different strides for anomaly vs normal events to balance dataset:
    - anomaly_stride: Small stride for fault events (e.g., 1) to capture all patterns
    - normal_stride: Larger stride for normal events (e.g., 50) to reduce samples
    
    Args:
        farm_dir: Path to wind farm directory
        datasets_dir: Path to datasets directory
        sequence_length: Length of each sequence
        anomaly_stride: Stride for anomaly events (smaller = more samples)
        normal_stride: Stride for normal events (larger = fewer samples)
        
    Returns:
        Tuple of (X_all, y_all, feature_cols) where:
            - X_all: List of sequence arrays per event
            - y_all: List of label arrays per event
            - feature_cols: List of feature column names
    """
    # Load event info
    event_info = load_event_info(farm_dir)
    
    X_all = []
    y_all = []
    feature_cols = None
    
    print(f"\nProcessing {len(event_info)} events...")
    
    for _, event in tqdm(event_info.iterrows(), total=len(event_info)):
        event_id = event['event_id']
        
        try:
            # Load event data
            df = load_event_data(event_id, datasets_dir)
            
            # Get feature columns (from first event)
            if feature_cols is None:
                feature_cols = get_feature_columns(df)
                print(f"  Using {len(feature_cols)} features")
            
            # Create binary labels
            labels = create_binary_labels(df, event, sequence_length)
            
            # Preprocess features
            features = preprocess_features(df, feature_cols)
            
            # Use different stride based on event type
            current_stride = anomaly_stride if event['event_label'] == 'anomaly' else normal_stride
            
            # Create sequences
            X, y = create_sequences(features, labels, sequence_length, current_stride)
            
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                print(f"    Event {event_id} ({event['event_label']}): {len(X)} sequences (stride={current_stride})")
                
        except Exception as e:
            print(f"  Error processing event {event_id}: {e}")
            continue
    
    return X_all, y_all, feature_cols


def split_by_events(X_all: list, y_all: list, event_info: pd.DataFrame,
                    test_size: float = 0.15, val_size: float = 0.15,
                    random_seed: int = 42) -> dict:
    """
    Split data by events to prevent data leakage.
    
    Events are split into train/val/test sets before concatenating sequences.
    This ensures no data from the same event appears in multiple sets.
    
    Args:
        X_all: List of sequence arrays per event
        y_all: List of label arrays per event
        event_info: DataFrame with event labels
        test_size: Fraction of events for test set
        val_size: Fraction of events for validation set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_events = len(X_all)
    event_indices = list(range(n_events))
    event_labels = event_info['event_label'].values
    
    # Convert labels to binary for stratification
    binary_labels = [1 if l == 'anomaly' else 0 for l in event_labels]
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        event_indices, 
        test_size=test_size,
        random_state=random_seed,
        stratify=binary_labels
    )
    
    # Get labels for train_val split
    train_val_labels = [binary_labels[i] for i in train_val_idx]
    
    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=train_val_labels
    )
    
    # Concatenate sequences for each split
    X_train = np.concatenate([X_all[i] for i in train_idx], axis=0)
    X_val = np.concatenate([X_all[i] for i in val_idx], axis=0)
    X_test = np.concatenate([X_all[i] for i in test_idx], axis=0)
    
    y_train = np.concatenate([y_all[i] for i in train_idx], axis=0)
    y_val = np.concatenate([y_all[i] for i in val_idx], axis=0)
    y_test = np.concatenate([y_all[i] for i in test_idx], axis=0)
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} sequences from {len(train_idx)} events")
    print(f"  Val:   {len(X_val)} sequences from {len(val_idx)} events")
    print(f"  Test:  {len(X_test)} sequences from {len(test_idx)} events")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_events': train_idx, 'val_events': val_idx, 'test_events': test_idx
    }


def normalize_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                   output_dir: str) -> tuple:
    """
    Normalize data using StandardScaler fit on training data.
    
    Args:
        X_train, X_val, X_test: Sequence arrays
        output_dir: Directory to save scaler
        
    Returns:
        Tuple of normalized (X_train, X_val, X_test)
    """
    # Reshape for scaler: (n_samples * seq_length, n_features)
    n_train, seq_len, n_features = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back to sequences
    X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def save_processed_data(data_dict: dict, output_dir: str):
    """
    Save processed data as .npz files.
    
    Args:
        data_dict: Dictionary with X_train, y_train, etc.
        output_dir: Directory to save files
    """
    ensure_dirs()
    
    # Save arrays
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        file_path = os.path.join(output_dir, f"{key}.npy")
        np.save(file_path, data_dict[key])
        print(f"Saved {key}: shape {data_dict[key].shape}")
    
    # Save metadata
    metadata = {
        'train_events': data_dict.get('train_events', []),
        'val_events': data_dict.get('val_events', []),
        'test_events': data_dict.get('test_events', []),
        'sequence_length': SEQUENCE_LENGTH,
    }
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"Metadata saved to: {metadata_path}")


def print_class_distribution(y_train: np.ndarray, y_val: np.ndarray, 
                            y_test: np.ndarray):
    """Print class distribution for each split."""
    print("\nClass distribution:")
    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        ratio = n_pos / len(y) * 100 if len(y) > 0 else 0
        print(f"  {name}: {n_neg} normal, {n_pos} fault ({ratio:.2f}% positive)")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("SCADA Fault Prediction - Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Ensure directories exist
    # ensure_dirs()
    
    # Process all events with stratified sampling
    print("\nUsing stratified sampling (memory-optimized):")
    print("  - Anomaly events: stride=5 (good coverage, lower memory)")
    print("  - Normal events: stride=100 (minimal samples)")
    
    X_all, y_all, feature_cols = process_all_events(
        WIND_FARM_A_DIR,
        WIND_FARM_A_DATASETS,
        SEQUENCE_LENGTH,
        anomaly_stride=3,
        normal_stride=100
    )
    
    # Load event info for splitting
    event_info = load_event_info(WIND_FARM_A_DIR)
    
    # Split by events
    data_dict = split_by_events(
        X_all, y_all, event_info,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED
    )
    
    # Normalize data
    X_train_norm, X_val_norm, X_test_norm = normalize_data(
        data_dict['X_train'],
        data_dict['X_val'],
        data_dict['X_test'],
        WIND_FARM_A_PROCESSED
    )
    
    # Update with normalized data
    data_dict['X_train'] = X_train_norm
    data_dict['X_val'] = X_val_norm
    data_dict['X_test'] = X_test_norm
    
    # Print class distribution
    print_class_distribution(
        data_dict['y_train'],
        data_dict['y_val'],
        data_dict['y_test']
    )
    
    # Save processed data
    save_processed_data(data_dict, WIND_FARM_A_PROCESSED)
    
    # Save feature columns for reference
    feature_cols_path = os.path.join(WIND_FARM_A_PROCESSED, "feature_columns.pkl")
    joblib.dump(feature_cols, feature_cols_path)
    print(f"Feature columns saved to: {feature_cols_path}")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
