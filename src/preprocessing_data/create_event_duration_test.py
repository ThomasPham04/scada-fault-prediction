"""
Create Test Set Using Event Duration Only (event_start to event_end)
Instead of using train_test='prediction', use exact event occurrence period
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WIND_FARM_A_DIR, WIND_FARM_A_PROCESSED,
    NBM_WINDOW_SIZE, NBM_STRIDE
)

def load_event_info():
    """Load event information."""
    event_info_path = os.path.join(WIND_FARM_A_DIR, 'event_info.csv')
    events_df = pd.read_csv(event_info_path, delimiter=';')
    
    # Convert timestamps
    events_df['event_start'] = pd.to_datetime(events_df['event_start'])
    events_df['event_end'] = pd.to_datetime(events_df['event_end'])
    
    print(f"Loaded {len(events_df)} events")
    return events_df

def load_scaler_and_metadata():
    """Load scaler and metadata from NBM_7day."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    
    scaler = joblib.load(os.path.join(nbm_dir, 'nbm_scaler_7day.pkl'))
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_7day.pkl'))
    
    return scaler, metadata

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

def process_event_duration(event_row, all_data_df, feature_cols, scaler, window_size, stride):
    """
    Process a single event using only event_start to event_end period.
    
    Args:
        event_row: Row from event_info.csv
        all_data_df: Full SCADA data with timestamp index
        feature_cols: List of feature columns
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
    
    # Extract data between event_start and event_end
    mask = (all_data_df.index >= event_start) & (all_data_df.index <= event_end)
    event_data = all_data_df.loc[mask, feature_cols].copy()
    
    if len(event_data) == 0:
        print(f"  Warning: Event {event_id} has no data in range")
        return None
    
    # Normalize
    event_data_normalized = scaler.transform(event_data)
    
    # Create sequences
    X_seq, y_seq = create_sequences(event_data_normalized, window_size, stride)
    
    if X_seq is None or len(X_seq) == 0:
        print(f"  Warning: Event {event_id} cannot create sequences (insufficient data)")
        return None
    
    return {
        'X': X_seq,
        'y': y_seq,
        'label': label,
        'event_id': event_id,
        'event_start': event_start,
        'event_end': event_end,
        'n_sequences': len(X_seq),
        'duration_hours': (event_end - event_start).total_seconds() / 3600
    }

def main():
    print("=" * 80)
    print("Creating Test Set from Event Duration (event_start to event_end)")
    print("=" * 80)
    
    # Load event info
    print("\nLoading event information...")
    events_df = load_event_info()
    
    # Load scaler and metadata
    print("\nLoading scaler and metadata...")
    scaler, metadata = load_scaler_and_metadata()
    feature_cols = metadata['feature_columns']
    window_size = NBM_WINDOW_SIZE
    stride = NBM_STRIDE
    
    print(f"\nWindow size: {window_size} timesteps")
    print(f"Stride: {stride} timesteps")
    print(f"Features: {len(feature_cols)}")
    
    # Load all SCADA data (we need to merge from all events)
    print("\nLoading SCADA data from all events...")
    all_data = []
    
    for event_id in tqdm(events_df['event_id'].values, desc="Loading events"):
        event_file = os.path.join(WIND_FARM_A_DIR, f'event_{event_id}.csv')
        if os.path.exists(event_file):
            df = pd.read_csv(event_file, delimiter=';')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_data.append(df)
    
    # Concatenate and sort
    print("Concatenating data...")
    all_data_df = pd.concat(all_data, ignore_index=True)
    all_data_df = all_data_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    all_data_df.set_index('timestamp', inplace=True)
    
    print(f"Total timesteps: {len(all_data_df):,}")
    print(f"Date range: {all_data_df.index.min()} to {all_data_df.index.max()}")
    
    # Process each event
    print("\nProcessing events (event_start to event_end only)...")
    output_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day_event_duration', 'test_by_event')
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    
    for idx, event_row in tqdm(events_df.iterrows(), total=len(events_df), desc="Processing"):
        result = process_event_duration(
            event_row, all_data_df, feature_cols, scaler, window_size, stride
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
            'duration_hours': result['duration_hours']
        })
        
        print(f"  Event {event_id:3d} ({result['label']:8s}): {result['n_sequences']:4d} sequences, Duration: {result['duration_hours']:.1f}h")
    
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
