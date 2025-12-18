"""
Display detailed information about Train/Val/Test splits for comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import WIND_FARM_A_DIR, WIND_FARM_A_PROCESSED


def load_event_info():
    """Load event information."""
    event_info_path = os.path.join(WIND_FARM_A_DIR, "event_info.csv")
    event_info = pd.read_csv(event_info_path, sep=';')
    return event_info


def get_event_label(event_id, event_info):
    """Get label for an event."""
    match = event_info[event_info['event_id'] == event_id]
    if len(match) > 0:
        return match.iloc[0]['event_label']
    return 'unknown'


def analyze_nbm_v2_data():
    """Analyze NBM V2 preprocessed data splits."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    
    print("="*80)
    print("TRAIN/VAL/TEST SPLIT INFORMATION - NBM V2")
    print("="*80)
    
    # Load data shapes
    X_train = np.load(os.path.join(nbm_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(nbm_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(nbm_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(nbm_dir, 'y_val.npy'))
    
    # Load metadata
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_v2.pkl'))
    
    # Load event info
    event_info = load_event_info()
    
    print("\n" + "="*80)
    print("1. TRAINING SET (from all 22 events, filtered for normal operation)")
    print("="*80)
    print(f"\nData Source: train_test=='train' from ALL events")
    print(f"Filtering: status_type_id==0, wind>4m/s, power>0")
    print(f"\nShape Information:")
    print(f"  X_train: {X_train.shape} (sequences, window_size, features)")
    print(f"  y_train: {y_train.shape} (sequences, features)")
    print(f"  Window size: {metadata['window_size']} timesteps ({metadata['window_size']*10/60/24:.1f} days)")
    print(f"  Number of features: {metadata['n_features']}")
    print(f"  Stride: {metadata['stride']} timesteps ({metadata['stride']*10/60:.1f} hours)")
    
    print(f"\nEvent Contributions:")
    if 'event_contributions' in metadata:
        contributions = metadata['event_contributions']
        print(f"  Total events contributed: {len(contributions)}")
        
        # Count by label
        anomaly_events = [eid for eid, info in contributions.items() 
                         if info['label'] == 'anomaly']
        normal_events = [eid for eid, info in contributions.items() 
                        if info['label'] == 'normal']
        
        print(f"  Anomaly events: {len(anomaly_events)}")
        print(f"  Normal events: {len(normal_events)}")
        
        # Show top contributors
        sorted_events = sorted(contributions.items(), 
                              key=lambda x: x[1]['filtered_len'], 
                              reverse=True)
        
        print(f"\n  Top 5 Contributors (by filtered timesteps):")
        for i, (eid, info) in enumerate(sorted_events[:5], 1):
            print(f"    {i}. Event {eid:3d} ({info['label']:7s}): "
                  f"{info['filtered_len']:6d} timesteps "
                  f"({info['filtered_len']/info['original_train_len']*100:.1f}% retained)")
    
    print("\n" + "="*80)
    print("2. VALIDATION SET (temporal split from training data)")
    print("="*80)
    print(f"\nData Source: Last 15% of combined training data (temporal split)")
    print(f"\nShape Information:")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  Percentage of total: {len(X_val)/(len(X_train)+len(X_val))*100:.1f}%")
    
    print("\n" + "="*80)
    print("3. TEST SET (from prediction periods)")
    print("="*80)
    print(f"\nData Source: train_test=='prediction' from events")
    print(f"Filtering: NONE (keep all status types for realistic testing)")
    
    # Load test events
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    test_events = []
    test_total_sequences = 0
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            X_test = data['X']
            label = str(data['label'])
            
            test_events.append({
                'event_id': event_id,
                'label': label,
                'n_sequences': len(X_test)
            })
            test_total_sequences += len(X_test)
    
    # Sort by event_id
    test_events = sorted(test_events, key=lambda x: x['event_id'])
    
    print(f"\nTotal test events: {len(test_events)}")
    print(f"Total test sequences: {test_total_sequences}")
    
    # Breakdown by label
    anomaly_test = [e for e in test_events if e['label'] == 'anomaly']
    normal_test = [e for e in test_events if e['label'] == 'normal']
    
    print(f"\nBreakdown:")
    print(f"  Anomaly events: {len(anomaly_test)} ({sum(e['n_sequences'] for e in anomaly_test)} sequences)")
    print(f"  Normal events: {len(normal_test)} ({sum(e['n_sequences'] for e in normal_test)} sequences)")
    
    print(f"\n  Test Events Detail:")
    print(f"  {'Event ID':<12} {'Label':<10} {'Sequences':<12}")
    print(f"  {'-'*34}")
    for event in test_events:
        print(f"  {event['event_id']:<12} {event['label']:<10} {event['n_sequences']:<12}")
    
    print("\n" + "="*80)
    print("4. OVERALL SUMMARY")
    print("="*80)
    
    total_sequences = len(X_train) + len(X_val) + test_total_sequences
    
    print(f"\nTotal Sequences: {total_sequences:,}")
    print(f"  Train:      {len(X_train):6,} ({len(X_train)/total_sequences*100:5.1f}%)")
    print(f"  Validation: {len(X_val):6,} ({len(X_val)/total_sequences*100:5.1f}%)")
    print(f"  Test:       {test_total_sequences:6,} ({test_total_sequences/total_sequences*100:5.1f}%)")
    
    print(f"\nData Split Strategy:")
    print(f"  ✓ Train: Combined normal data from all 22 events (train portion)")
    print(f"  ✓ Val: Temporal split (last 15% of train)")
    print(f"  ✓ Test: Prediction periods from {len(test_events)} events (unfiltered)")
    
    print(f"\nKey Characteristics:")
    print(f"  • Training on NORMAL data only (status=0, wind>4, power>0)")
    print(f"  • Testing on REAL prediction periods (all status types)")
    print(f"  • No data leakage (different time periods)")
    print(f"  • Realistic evaluation setup")
    
    # Feature statistics
    print("\n" + "="*80)
    print("5. FEATURE STATISTICS")
    print("="*80)
    
    print(f"\nFeatures: {metadata['n_features']}")
    print(f"Feature engineering:")
    print(f"  • Angle features → sin/cos transformation")
    print(f"  • Counter features → removed")
    print(f"  • Final feature list: {len(metadata['feature_columns'])} features")
    
    # Normalization info
    print(f"\nNormalization:")
    print(f"  • Method: StandardScaler (Z-score)")
    print(f"  • Fitted on: Training data only (normal operation)")
    print(f"  • Applied to: Train, Val, Test")
    
    print("\n" + "="*80)
    print("6. COMPARISON WITH GROUND TRUTH")
    print("="*80)
    
    print(f"\nTotal events in dataset: 22")
    print(f"  • Anomaly: 12")
    print(f"  • Normal: 10")
    
    print(f"\nEvents used for training: 22 (train portion only)")
    print(f"Events available for testing: {len(test_events)}")
    print(f"  • {len(anomaly_test)} anomaly events (to detect)")
    print(f"  • {len(normal_test)} normal events (should have low error)")
    
    print(f"\nGround Truth for Evaluation:")
    print(f"  • Source: event_info.csv labels")
    print(f"  • Anomaly=1, Normal=0")
    print(f"  • Used for computing Precision, Recall, F1, etc.")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)


if __name__ == "__main__":
    analyze_nbm_v2_data()
