"""
Create Test Set Using Event Duration (event_start to event_end)
This script creates a test set by extracting all timesteps between event_start and event_end
from event_info.csv, similar to the approach in preprocess_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WIND_FARM_A_DIR, WIND_FARM_A_DATASETS, WIND_FARM_A_PROCESSED,
    NBM_WINDOW_SIZE, NBM_STRIDE, EXCLUDE_COLUMNS,
    NBM_FEATURE_COLUMNS, NBM_ANGLE_FEATURES, NBM_COUNTER_FEATURES
)


def load_event_info():
    """Load event information."""
    event_info_path = os.path.join(WIND_FARM_A_DIR, 'event_info.csv')
    events_df = pd.read_csv(event_info_path, delimiter=';')
    
    # Convert timestamps
    events_df['event_start'] = pd.to_datetime(events_df['event_start'])
    events_df['event_end'] = pd.to_datetime(events_df['event_end'])
    
    print(f"Loaded {len(events_df)} events")
    print(f"  - Anomaly: {(events_df['event_label'] == 'anomaly').sum()}")
    print(f"  - Normal: {(events_df['event_label'] == 'normal').sum()}")
    return events_df


def load_event_data(event_id, datasets_dir):
    """Load a single event dataset."""
    file_path = os.path.join(datasets_dir, f'{event_id}.csv')
    df = pd.read_csv(file_path, sep=';')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df


def engineer_angle_features(df):
    """Convert angle features to sin/cos for continuity."""
    df_copy = df.copy()
    
    for col in NBM_ANGLE_FEATURES:
        if col in df_copy.columns:
            radians = np.radians(df_copy[col])
            df_copy[f'{col}_sin'] = np.sin(radians)
            df_copy[f'{col}_cos'] = np.cos(radians)
            df_copy.drop(col, axis=1, inplace=True)
    
    return df_copy


def drop_counter_features(df):
    """Drop counter features (Wh, VArh)."""
    cols_to_drop = [col for col in NBM_COUNTER_FEATURES if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
    return df


def get_nbm_feature_columns(df):
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


def get_feature_columns(df):
    """Get list of feature columns (exclude non-sensor columns)."""
    exclude = EXCLUDE_COLUMNS + ['time_stamp']
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


def preprocess_features(df, feature_cols):
    """Extract and preprocess feature values."""
    # Handle missing values using forward fill, then backward fill
    features = df[feature_cols].copy()
    features = features.ffill().bfill()
    
    # Convert to numpy array
    return features.values


def create_sequences(data, window_size, stride):
    """Create sequences using sliding window."""
    sequences = []
    targets = []
    
    n_samples = len(data)
    
    for i in range(0, n_samples - window_size + 1, stride):
        window = data[i:i + window_size]
        target = data[i + window_size - 1]  # Last timestep
        
        sequences.append(window)
        targets.append(target)
    
    if len(sequences) == 0:
        return None, None
    
    return np.array(sequences), np.array(targets)


def load_scaler_and_metadata():
    """Load scaler and metadata from NBM_7day."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    
    scaler = joblib.load(os.path.join(nbm_dir, 'nbm_scaler_7day.pkl'))
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_7day.pkl'))
    
    return scaler, metadata


def filter_event_duration(df, event_start, event_end):
    """
    Filter dataframe to only include timesteps between event_start and event_end.
    
    Args:
        df: DataFrame with time_stamp column
        event_start: Start timestamp
        event_end: End timestamp
    
    Returns:
        Filtered DataFrame
    """
    mask = (df['time_stamp'] >= event_start) & (df['time_stamp'] <= event_end)
    return df[mask].copy()


def process_event(event_row, datasets_dir, scaler, window_size, stride):
    """
    Process a single event using only event_start to event_end period.
    
    Args:
        event_row: Row from event_info.csv
        datasets_dir: Path to datasets directory
        scaler: Fitted scaler
        window_size: Sequence window size
        stride: Stride for sliding window
    
    Returns:
        dict with X, y, and metadata
    """
    event_id = event_row['event_id']
    event_start = event_row['event_start']
    event_end = event_row['event_end']
    label = event_row['event_label']
    
    # Load event data
    df = load_event_data(event_id, datasets_dir)
    
    # Apply feature engineering (same as NBM_7day preprocessing)
    df = drop_counter_features(df)
    df = engineer_angle_features(df)
    
    # Get feature columns after engineering
    feature_cols = get_nbm_feature_columns(df)
    
    # Filter to event duration only (event_start to event_end)
    df_filtered = filter_event_duration(df, event_start, event_end)
    
    if len(df_filtered) == 0:
        print(f"  Warning: Event {event_id} has no data in duration range")
        return None
    
    # Preprocess features
    features = preprocess_features(df_filtered, feature_cols)
    
    # Normalize using the pre-fitted scaler
    features_normalized = scaler.transform(features)
    
    # Create sequences
    X_seq, y_seq = create_sequences(features_normalized, window_size, stride)
    
    if X_seq is None or len(X_seq) == 0:
        print(f"  Warning: Event {event_id} cannot create sequences (insufficient data)")
        return None
    
    duration_hours = (event_end - event_start).total_seconds() / 3600
    
    return {
        'X': X_seq,
        'y': y_seq,
        'label': label,
        'event_id': event_id,
        'event_start': event_start,
        'event_end': event_end,
        'n_sequences': len(X_seq),
        'n_timesteps': len(df_filtered),
        'duration_hours': duration_hours
    }


def main():
    print("=" * 80)
    print("Creating Test Set from Event Duration (event_start to event_end)")
    print("=" * 80)
    
    # Load event info
    print("\nLoading event information...")
    events_df = load_event_info()
    
    # Load scaler and metadata
    print("\nLoading scaler and metadata from NBM_7day...")
    scaler, metadata = load_scaler_and_metadata()
    feature_cols = metadata['feature_columns']
    window_size = NBM_WINDOW_SIZE
    stride = NBM_STRIDE
    
    print(f"\nConfiguration:")
    print(f"  Window size: {window_size} timesteps")
    print(f"  Stride: {stride} timesteps")
    print(f"  Features: {len(feature_cols)}")
    
    # Create output directory
    output_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day_event_duration', 'test_by_event')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each event
    print("\nProcessing events (using event_start to event_end only)...")
    results_summary = []
    
    for idx, event_row in tqdm(events_df.iterrows(), total=len(events_df), desc="Processing"):
        result = process_event(
            event_row, WIND_FARM_A_DATASETS, scaler, window_size, stride
        )
        
        if result is None:
            continue
        
        # Save event
        event_id = result['event_id']
        output_file = os.path.join(output_dir, f'event_{event_id}.npz')
        
        np.savez(
            output_file,
            X=result['X'],
            y=result['y'],
            label=result['label'],
            event_start=str(result['event_start']),
            event_end=str(result['event_end'])
        )
        
        results_summary.append({
            'event_id': event_id,
            'label': result['label'],
            'n_sequences': result['n_sequences'],
            'n_timesteps': result['n_timesteps'],
            'duration_hours': result['duration_hours']
        })
        
        print(f"  Event {event_id:3d} ({result['label']:8s}): "
              f"{result['n_sequences']:4d} sequences, "
              f"{result['n_timesteps']:5d} timesteps, "
              f"Duration: {result['duration_hours']:.1f}h")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary_df = pd.DataFrame(results_summary)
    
    print(f"\nTotal events processed: {len(summary_df)}")
    print(f"  Anomaly events: {len(summary_df[summary_df['label'] == 'anomaly'])}")
    print(f"  Normal events: {len(summary_df[summary_df['label'] == 'normal'])}")
    
    print(f"\nTotal sequences: {summary_df['n_sequences'].sum():,}")
    print(f"  Anomaly sequences: {summary_df[summary_df['label'] == 'anomaly']['n_sequences'].sum():,}")
    print(f"  Normal sequences: {summary_df[summary_df['label'] == 'normal']['n_sequences'].sum():,}")
    
    print(f"\nTotal timesteps: {summary_df['n_timesteps'].sum():,}")
    print(f"  Anomaly timesteps: {summary_df[summary_df['label'] == 'anomaly']['n_timesteps'].sum():,}")
    print(f"  Normal timesteps: {summary_df[summary_df['label'] == 'normal']['n_timesteps'].sum():,}")
    
    print(f"\nAverage duration:")
    print(f"  Anomaly: {summary_df[summary_df['label'] == 'anomaly']['duration_hours'].mean():.1f} hours")
    print(f"  Normal: {summary_df[summary_df['label'] == 'normal']['duration_hours'].mean():.1f} hours")
    
    print(f"\nTest set saved to: {output_dir}")
    print("=" * 80)
    
    # Save summary
    summary_file = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day_event_duration', 'test_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
