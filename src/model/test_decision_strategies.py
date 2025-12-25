"""
Test Multi-Metric Decision Strategies for NBM LSTM
Find strategy that reduces False Alarm Rate while maintaining good Recall
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

def get_base_threshold(percentile=95):
    """Get threshold from validation errors."""
    errors_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'nbm_7day_errors.npz')
    data = np.load(errors_path)
    val_mae = data['val_mae']
    return np.percentile(val_mae, percentile)

def compute_event_mae_stats(mae):
    """Compute various statistics from MAE array."""
    return {
        'mean': np.mean(mae),
        'p50': np.percentile(mae, 50),
        'p95': np.percentile(mae, 95),
        'p99': np.percentile(mae, 99),
        'max': np.max(mae),
        'std': np.std(mae)
    }

def apply_strategies(mae_stats, thresholds):
    """
    Apply different decision strategies.
    
    Strategies:
    1. p95_only: p95 > threshold (current baseline)
    2. mean_only: mean > threshold
    3. dual_condition: (mean > t1) AND (p95 > t2)
    4. triple_condition: (mean > t1) AND (p95 > t2) AND (pct_exceed > X%)
    5. max_based: max > threshold (very conservative)
    6. mean_and_percent: (mean > t1) AND (pct_exceed > 10%)
    """
    
    # Precompute
    mean = mae_stats['mean']
    p95 = mae_stats['p95']
    max_val = mae_stats['max']
    
    # For percent exceed, we need the full MAE array (not just stats)
    # We'll pass it separately
    
    decisions = {}
    
    # Strategy 1: p95 only (baseline)
    decisions['p95_only'] = p95 > thresholds['p95']
    
    # Strategy 2: mean only
    decisions['mean_only'] = mean > thresholds['mean']
    
    # Strategy 3: Dual - both mean AND p95
    decisions['dual_mean_p95'] = (mean > thresholds['mean']) and (p95 > thresholds['p95'])
    
    # Strategy 4: max based (conservative)
    decisions['max_based'] = max_val > thresholds['max']
    
    return decisions

def evaluate_strategy(model, test_events, strategy_name, thresholds):
    """Evaluate a specific decision strategy."""
    TP = FP = TN = FN = 0
    
    for event_id, data in test_events.items():
        X = data['X']
        y = data['y']
        true_label = data['label']
        
        # Predict
        y_pred = model.predict(X, verbose=0, batch_size=256)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        
        # Compute stats
        mae_stats = compute_event_mae_stats(mae)
        
        # Apply strategy
        if strategy_name == 'p95_only':
            detected = mae_stats['p95'] > thresholds['p95']
        
        elif strategy_name == 'mean_only':
            detected = mae_stats['mean'] > thresholds['mean']
        
        elif strategy_name == 'dual_mean_p95':
            detected = (mae_stats['mean'] > thresholds['mean']) and \
                      (mae_stats['p95'] > thresholds['p95'])
        
        elif strategy_name == 'triple_combined':
            # Mean AND p95 AND percent exceed
            pct_exceed = np.mean(mae > thresholds['p95'])
            detected = (mae_stats['mean'] > thresholds['mean']) and \
                      (mae_stats['p95'] > thresholds['p95']) and \
                      (pct_exceed > 0.1)  # At least 10% samples exceed
        
        elif strategy_name == 'max_based':
            detected = mae_stats['max'] > thresholds['max']
        
        elif strategy_name == 'mean_and_percent':
            pct_exceed = np.mean(mae > thresholds['mean'])
            detected = (mae_stats['mean'] > thresholds['mean']) and \
                      (pct_exceed > 0.15)  # 15% samples exceed mean threshold
        
        elif strategy_name == 'strict_dual':
            # Stricter thresholds - use p99 for one condition
            detected = (mae_stats['mean'] > thresholds['p95']) and \
                      (mae_stats['p95'] > thresholds['p99'])
        
        elif strategy_name == 'relaxed_triple':
            # More relaxed version
            pct_exceed = np.mean(mae > thresholds['mean'])
            detected = (mae_stats['mean'] > thresholds['mean']) or \
                      ((mae_stats['p95'] > thresholds['p95']) and (pct_exceed > 0.2))
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
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
    
    return {
        'strategy': strategy_name,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'recall': recall,
        'precision': precision,
        'far': far,
        'f1': f1
    }

def main():
    print("=" * 100)
    print("Testing Multi-Metric Decision Strategies for NBM LSTM 7-day")
    print("=" * 100)
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_7day.keras')
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load test data
    print("Loading test events...")
    test_events = load_test_data()
    print(f"  Loaded {len(test_events)} test events")
    
    # Get thresholds from validation
    print("\nComputing thresholds from validation MAE...")
    errors_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'nbm_7day_errors.npz')
    data = np.load(errors_path)
    val_mae = data['val_mae']
    
    thresholds = {
        'mean': np.mean(val_mae) + 2 * np.std(val_mae),  # Mean + 2*std
        'p95': np.percentile(val_mae, 95),
        'p99': np.percentile(val_mae, 99),
        'max': np.percentile(val_mae, 99.9)
    }
    
    print("\nThresholds:")
    for name, value in thresholds.items():
        print(f"  {name}: {value:.6f}")
    
    # Define strategies to test
    strategies = [
        'p95_only',           # Baseline
        'mean_only',          # Just mean
        'dual_mean_p95',      # Mean AND p95
        'triple_combined',    # Mean AND p95 AND percent
        'max_based',          # Max only
        'mean_and_percent',   # Mean AND percent exceed
        'strict_dual',        # Stricter version
        'relaxed_triple'      # More relaxed
    ]
    
    # Evaluate all strategies
    print("\n" + "=" * 100)
    print("Evaluating strategies...")
    print("=" * 100)
    
    results = []
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        result = evaluate_strategy(model, test_events, strategy, thresholds)
        results.append(result)
    
    # Print comparison table
    print("\n" + "=" * 110)
    print("STRATEGY COMPARISON")
    print("=" * 110)
    print(f"{'Strategy':<20} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Recall':<10} {'Precision':<10} {'FAR':<10} {'F1':<10}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['strategy']:<20} {r['TP']:<6} {r['FP']:<6} {r['TN']:<6} {r['FN']:<6} "
              f"{r['recall']:<10.2%} {r['precision']:<10.2%} {r['far']:<10.2%} {r['f1']:<10.4f}")
    
    # Find best strategies
    print("\n" + "=" * 110)
    print("RECOMMENDATIONS")
    print("=" * 110)
    
    # Best F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n1. Best F1 Score: {best_f1['strategy']} (F1={best_f1['f1']:.4f})")
    print(f"   TP={best_f1['TP']}, FP={best_f1['FP']}, TN={best_f1['TN']}, FN={best_f1['FN']}")
    print(f"   Recall: {best_f1['recall']:.2%}, Precision: {best_f1['precision']:.2%}, FAR: {best_f1['far']:.2%}")
    
    # Lowest FAR with acceptable recall (>60%)
    acceptable = [r for r in results if r['recall'] > 0.6]
    if acceptable:
        best_far = min(acceptable, key=lambda x: x['far'])
        print(f"\n2. Lowest FAR (Recall > 60%): {best_far['strategy']} (FAR={best_far['far']:.2%})")
        print(f"   TP={best_far['TP']}, FP={best_far['FP']}, TN={best_far['TN']}, FN={best_far['FN']}")
        print(f"   Recall: {best_far['recall']:.2%}, Precision: {best_far['precision']:.2%}, F1: {best_far['f1']:.4f}")
    
    # Best balance (FAR < 50% and Recall > 70%)
    balanced = [r for r in results if r['far'] < 0.5 and r['recall'] > 0.7]
    if balanced:
        best_balanced = max(balanced, key=lambda x: x['f1'])
        print(f"\n3. Best Balanced (FAR < 50%, Recall > 70%): {best_balanced['strategy']}")
        print(f"   TP={best_balanced['TP']}, FP={best_balanced['FP']}, TN={best_balanced['TN']}, FN={best_balanced['FN']}")
        print(f"   Recall: {best_balanced['recall']:.2%}, Precision: {best_balanced['precision']:.2%}")
        print(f"   FAR: {best_balanced['far']:.2%}, F1: {best_balanced['f1']:.4f}")
    else:
        print("\n3. No strategy achieves FAR < 50% with Recall > 70%")
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'strategy_comparison.json')
    with open(output_path, 'w') as f:
        json.dump({
            'thresholds': {k: float(v) for k, v in thresholds.items()},
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 110)

if __name__ == "__main__":
    main()
