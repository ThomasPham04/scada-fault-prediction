"""
Test Multiple Thresholds for NBM LSTM 7-day Model
Find optimal threshold for anomaly detection
"""

import os
import sys
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR

def load_test_data():
    """Load test events from NBM 7-day."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    
    test_events = {}
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            
            test_events[event_id] = {
                'X': data['X'],
                'y': data['y'],
                'label': str(data['label'])
            }
    
    return test_events

def get_thresholds():
    """Get different threshold percentiles from validation errors."""
    errors_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'nbm_7day_errors.npz')
    data = np.load(errors_path)
    val_mae = data['val_mae']
    
    percentiles = [90, 95, 97, 99, 99.5, 99.9]
    thresholds = {}
    
    for p in percentiles:
        thresholds[f'p{p}'] = np.percentile(val_mae, p)
    
    return thresholds, val_mae

def evaluate_with_threshold(model, test_events, threshold, threshold_name):
    """Evaluate model with a specific threshold."""
    TP = FP = TN = FN = 0
    
    for event_id, data in test_events.items():
        X = data['X']
        y = data['y']
        true_label = data['label']
        
        # Predict
        y_pred = model.predict(X, verbose=0, batch_size=256)
        
        # Compute MAE per sample
        mae = np.mean(np.abs(y - y_pred), axis=1)
        
        # Use p95 of event MAE as anomaly score
        event_p95 = np.percentile(mae, 95)
        detected = event_p95 > threshold
        
        # Update confusion matrix
        is_anomaly = (true_label == 'anomaly')
        if is_anomaly and detected:
            TP += 1
        elif not is_anomaly and not detected:
            TN += 1
        elif not is_anomaly and detected:
            FP += 1
        else:
            FN += 1
    
    # Calculate metrics
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    return {
        'threshold_name': threshold_name,
        'threshold': threshold,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'recall': recall,
        'precision': precision,
        'far': far,
        'f1': f1,
        'accuracy': accuracy
    }

def main():
    print("=" * 80)
    print("Testing Multiple Thresholds for NBM LSTM 7-day")
    print("=" * 80)
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_7day.keras')
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load test data
    print("Loading test events...")
    test_events = load_test_data()
    print(f"  Loaded {len(test_events)} test events")
    
    # Get thresholds
    print("\nComputing thresholds from validation MAE...")
    thresholds, val_mae = get_thresholds()
    
    print("\nValidation MAE Statistics:")
    print(f"  Mean: {np.mean(val_mae):.6f}")
    print(f"  Std:  {np.std(val_mae):.6f}")
    print(f"  Min:  {np.min(val_mae):.6f}")
    print(f"  Max:  {np.max(val_mae):.6f}")
    
    print("\nThresholds:")
    for name, value in thresholds.items():
        print(f"  {name}: {value:.6f}")
    
    # Evaluate with each threshold
    print("\n" + "=" * 80)
    print("Evaluating with different thresholds...")
    print("=" * 80)
    
    results = []
    for name, threshold in thresholds.items():
        print(f"\nTesting {name} = {threshold:.6f}...")
        result = evaluate_with_threshold(model, test_events, threshold, name)
        results.append(result)
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("THRESHOLD COMPARISON")
    print("=" * 100)
    print(f"{'Threshold':<12} {'Value':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Recall':<10} {'Precision':<10} {'FAR':<10} {'F1':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['threshold_name']:<12} {r['threshold']:<12.6f} {r['TP']:<6} {r['FP']:<6} {r['TN']:<6} {r['FN']:<6} "
              f"{r['recall']:<10.2%} {r['precision']:<10.2%} {r['far']:<10.2%} {r['f1']:<10.4f}")
    
    # Find best threshold
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    # Best F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\nBest F1 Score: {best_f1['threshold_name']} (F1={best_f1['f1']:.4f})")
    print(f"  Threshold: {best_f1['threshold']:.6f}")
    print(f"  Recall: {best_f1['recall']:.2%}, Precision: {best_f1['precision']:.2%}")
    print(f"  FAR: {best_f1['far']:.2%}")
    
    # Best Recall (with acceptable FAR < 70%)
    best_recall = max([r for r in results if r['far'] < 0.7], 
                      key=lambda x: x['recall'], 
                      default=None)
    if best_recall:
        print(f"\nBest Recall (FAR < 70%): {best_recall['threshold_name']} (Recall={best_recall['recall']:.2%})")
        print(f"  Threshold: {best_recall['threshold']:.6f}")
        print(f"  FAR: {best_recall['far']:.2%}, Precision: {best_recall['precision']:.2%}")
        print(f"  F1: {best_recall['f1']:.4f}")
    
    # Lowest FAR (with acceptable Recall > 50%)
    best_far = min([r for r in results if r['recall'] > 0.5],
                   key=lambda x: x['far'],
                   default=None)
    if best_far:
        print(f"\nLowest FAR (Recall > 50%): {best_far['threshold_name']} (FAR={best_far['far']:.2%})")
        print(f"  Threshold: {best_far['threshold']:.6f}")
        print(f"  Recall: {best_far['recall']:.2%}, Precision: {best_far['precision']:.2%}")
        print(f"  F1: {best_far['f1']:.4f}")
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'threshold_comparison.json')
    with open(output_path, 'w') as f:
        json.dump({
            'val_mae_stats': {
                'mean': float(np.mean(val_mae)),
                'std': float(np.std(val_mae)),
                'min': float(np.min(val_mae)),
                'max': float(np.max(val_mae))
            },
            'thresholds': {k: float(v) for k, v in thresholds.items()},
            'results': [{k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                        for k, v in r.items()} for r in results]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
