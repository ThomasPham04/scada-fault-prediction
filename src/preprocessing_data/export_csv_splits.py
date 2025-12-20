"""
Export Train/Val/Test Splits to CSV

This script extracts and exports the train/val/test splits as separate CSV files
without running the full preprocessing pipeline. It reuses the logic from 
preprocess_nbm_v2.py but only generates CSV output.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WIND_FARM_A_DIR, WIND_FARM_A_DATASETS, WIND_FARM_A_PROCESSED,
    NBM_WINDOW_SIZE, NBM_CUT_IN_WIND_SPEED, NBM_MIN_POWER,
    NBM_NORMAL_STATUS, NBM_FEATURE_COLUMNS, NBM_ANGLE_FEATURES, 
    NBM_COUNTER_FEATURES, EXCLUDE_COLUMNS
)

# Import functions from preprocess_nbm_v2
from preprocess_nbm_v2 import (
    load_event_info, load_event_data, engineer_angle_features,
    drop_counter_features, get_nbm_feature_columns, preprocess_features
)


def export_csv_splits(output_dir: str, val_ratio: float = 0.15):
    """
    Export train/val/test splits as CSV files.
    
    Args:
        output_dir: Directory to save CSV files
        val_ratio: Validation split ratio (default 15%)
    """
    print("=" * 70)
    print("Export Train/Val/Test Splits to CSV")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load event info
    print("\nLoading event information...")
    event_info = load_event_info(WIND_FARM_A_DIR)
    
    # ========== PROCESS TRAIN DATA ==========
    print(f"\n{'='*70}")
    print("Processing TRAIN data from all events...")
    print(f"{'='*70}")
    
    all_train_data = []
    feature_cols = None
    
    for _, event in tqdm(event_info.iterrows(), total=len(event_info), desc="Train data"):
        event_id = event['event_id']
        event_label = event['event_label']
        
        try:
            # Load event data
            df = load_event_data(event_id, WIND_FARM_A_DATASETS)
            
            # Filter for TRAIN portion only
            df_train = df[df['train_test'] == 'train'].copy()
            
            if len(df_train) == 0:
                continue
            
            # Filter for NORMAL OPERATION
            mask = (
                (df_train['status_type_id'] == 0) &
                (df_train['wind_speed_3_avg'] > NBM_CUT_IN_WIND_SPEED) &
                (df_train['power_29_avg'] > NBM_MIN_POWER)
            )
            
            df_normal = df_train[mask].copy()
            
            if len(df_normal) < NBM_WINDOW_SIZE + 1:
                continue
            
            # Feature engineering
            df_normal = engineer_angle_features(df_normal)
            df_normal = drop_counter_features(df_normal)
            
            # Get feature columns (from first event)
            if feature_cols is None:
                feature_cols = get_nbm_feature_columns(df_normal)
            
            # Preprocess features
            features = preprocess_features(df_normal, feature_cols)
            all_train_data.append(features)
                
        except Exception as e:
            print(f"Event {event_id}: ERROR - {e}")
            continue
    
    # Concatenate all training data
    combined_train = np.concatenate(all_train_data, axis=0)
    print(f"\nCombined training data: {combined_train.shape[0]} timesteps")
    
    # Temporal split for train/val
    n_total = len(combined_train)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size
    
    train_split = combined_train[:train_size]
    val_split = combined_train[train_size:]
    
    print(f"\nTrain/Val split:")
    print(f"  Train: {len(train_split)} timesteps ({len(train_split)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_split)} timesteps ({len(val_split)/n_total*100:.1f}%)")
    
    # ========== PROCESS TEST DATA ==========
    print(f"\n{'='*70}")
    print("Processing TEST data from all events...")
    print(f"{'='*70}")
    
    test_data_by_event = {}
    
    for _, event in tqdm(event_info.iterrows(), total=len(event_info), desc="Test data"):
        event_id = event['event_id']
        event_label = event['event_label']
        
        try:
            # Load event data
            df = load_event_data(event_id, WIND_FARM_A_DATASETS)
            
            # Filter for PREDICTION portion only
            df_pred = df[df['train_test'] == 'prediction'].copy()
            
            if len(df_pred) == 0:
                continue
            
            if len(df_pred) < NBM_WINDOW_SIZE + 1:
                continue
            
            # Feature engineering (same as train)
            df_pred = engineer_angle_features(df_pred)
            df_pred = drop_counter_features(df_pred)
            
            # Preprocess features
            features = preprocess_features(df_pred, feature_cols)
            
            test_data_by_event[event_id] = {
                'features': features,
                'label': event_label
            }
                
        except Exception as e:
            print(f"Event {event_id}: ERROR - {e}")
            continue
    
    print(f"\nProcessed {len(test_data_by_event)} test events")
    
    # ========== SAVE CSV FILES ==========
    print(f"\n{'='*70}")
    print("Saving CSV files...")
    print(f"{'='*70}")
    
    # Save train split
    train_df = pd.DataFrame(train_split, columns=feature_cols)
    train_path = os.path.join(output_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    print(f"✓ Train: {train_df.shape} → {train_path}")
    
    # Save validation split
    val_df = pd.DataFrame(val_split, columns=feature_cols)
    val_path = os.path.join(output_dir, 'val.csv')
    val_df.to_csv(val_path, index=False)
    print(f"✓ Val:   {val_df.shape} → {val_path}")
    
    # Save test data (combined all events)
    test_data_combined = []
    test_event_ids = []
    test_labels = []
    
    for event_id, data in test_data_by_event.items():
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
        test_path = os.path.join(output_dir, 'test.csv')
        test_df.to_csv(test_path, index=False)
        print(f"✓ Test:  {test_df.shape} → {test_path}")
        
        # Also save test data by individual events
        test_by_event_dir = os.path.join(output_dir, 'test_by_event')
        os.makedirs(test_by_event_dir, exist_ok=True)
        
        for event_id, data in test_data_by_event.items():
            event_df = pd.DataFrame(data['features'], columns=feature_cols)
            event_path = os.path.join(test_by_event_dir, f'event_{event_id}.csv')
            event_df.to_csv(event_path, index=False)
        
        print(f"✓ Test by event: {len(test_data_by_event)} files → {test_by_event_dir}")
    
    # Save feature names
    feature_info_path = os.path.join(output_dir, 'feature_columns.txt')
    with open(feature_info_path, 'w') as f:
        f.write("Feature Columns:\n")
        f.write("=" * 50 + "\n")
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {col}\n")
    print(f"✓ Feature info: {feature_info_path}")
    
    print(f"\n{'='*70}")
    print("✓ CSV export complete!")
    print(f"{'='*70}")
    print(f"\nFiles saved to: {output_dir}")
    print(f"  • train.csv:        {len(train_split):,} rows × {len(feature_cols)} features")
    print(f"  • val.csv:          {len(val_split):,} rows × {len(feature_cols)} features")
    print(f"  • test.csv:         {len(test_combined_array):,} rows × {len(feature_cols)+2} features")
    print(f"  • test_by_event/:   {len(test_data_by_event)} files")


def main():
    """Main function to run CSV export."""
    # Default output directory
    default_output = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2', 'csv_splits')
    
    # Allow custom output directory from command line
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = default_output
    
    print(f"\nOutput directory: {output_dir}")
    
    # Run export
    export_csv_splits(output_dir, val_ratio=0.15)


if __name__ == "__main__":
    main()
