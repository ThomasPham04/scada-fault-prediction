"""
Test NBM LSTM with Multiple Thresholds

Try different threshold values to find the best balance between
recall and false alarm rate.
"""

import os
import sys
import numpy as np
import json
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR


def load_nbm_data():
    """Load NBM V2 test data."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
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


def compute_event_scores(model, test_events):
    """Compute p95 MAE for all events."""
    scores = []
    
    for event_id, event_data in sorted(test_events.items()):
        X = event_data['X']
        y = event_data['y']
        
        y_pred = model.predict(X, verbose=0)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        p95_mae = np.percentile(mae, 95)
        
        scores.append({
            'event_id': event_id,
            'p95_mae': float(p95_mae),
            'is_anomaly': (event_data['label'] == 'anomaly')
        })
    
    return scores


def evaluate_at_threshold(scores, threshold):
    """Evaluate performance at a specific threshold."""
    TP = FP = TN = FN = 0
    
    for score in scores:
        detected = score['p95_mae'] > threshold
        actual = score['is_anomaly']
        
        if actual and detected:
            TP += 1
        elif actual and not detected:
            FN += 1
        elif not actual and detected:
            FP += 1
        else:
            TN += 1
    
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    return {
        'threshold': threshold,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'false_alarm_rate': fpr
    }


def print_comparison_table(results):
    """Print comparison table for different thresholds."""
    print("\n" + "=" * 100)
    print("THRESHOLD COMPARISON")
    print("=" * 100)
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1 Score':<12} {'False Alarm':<14} {'TP':<5} {'FP':<5} {'TN':<5} {'FN':<5}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['threshold']:<12.2f} {r['recall']:<10.1%} {r['precision']:<12.1%} "
              f"{r['f1_score']:<12.4f} {r['false_alarm_rate']:<14.1%} "
              f"{r['TP']:<5} {r['FP']:<5} {r['TN']:<5} {r['FN']:<5}")
    
    print("=" * 100)


def highlight_best(results):
    """Highlight the best thresholds."""
    print("\n" + "=" * 100)
    print("BEST THRESHOLDS BY CRITERIA")
    print("=" * 100)
    
    # Best F1
    best_f1 = max(results, key=lambda x: x['f1_score'])
    print(f"\n🏆 BEST F1 SCORE: {best_f1['f1_score']:.4f}")
    print(f"   Threshold: {best_f1['threshold']:.2f}")
    print(f"   Recall: {best_f1['recall']:.1%}, Precision: {best_f1['precision']:.1%}, False Alarm: {best_f1['false_alarm_rate']:.1%}")
    
    # Best with 100% recall
    perfect_recall = [r for r in results if r['recall'] == 1.0]
    if perfect_recall:
        best_perfect = min(perfect_recall, key=lambda x: x['false_alarm_rate'])
        print(f"\n✅ BEST WITH 100% RECALL (No Missed Faults):")
        print(f"   Threshold: {best_perfect['threshold']:.2f}")
        print(f"   F1 Score: {best_perfect['f1_score']:.4f}, Precision: {best_perfect['precision']:.1%}, False Alarm: {best_perfect['false_alarm_rate']:.1%}")
    
    # Best balanced (high recall + low false alarm)
    for r in results:
        r['score'] = r['recall'] - r['false_alarm_rate']
    best_balanced = max(results, key=lambda x: x['score'])
    print(f"\n⚖️  BEST BALANCED (Recall - False Alarm):")
    print(f"   Threshold: {best_balanced['threshold']:.2f}")
    print(f"   Recall: {best_balanced['recall']:.1%}, Precision: {best_balanced['precision']:.1%}, False Alarm: {best_balanced['false_alarm_rate']:.1%}")
    print(f"   F1 Score: {best_balanced['f1_score']:.4f}")
    
    # Lowest false alarm with decent recall
    decent_recall = [r for r in results if r['recall'] >= 0.75]
    if decent_recall:
        best_low_fa = min(decent_recall, key=lambda x: x['false_alarm_rate'])
        print(f"\n🎯 LOWEST FALSE ALARM (with Recall ≥75%):")
        print(f"   Threshold: {best_low_fa['threshold']:.2f}")
        print(f"   Recall: {best_low_fa['recall']:.1%}, Precision: {best_low_fa['precision']:.1%}, False Alarm: {best_low_fa['false_alarm_rate']:.1%}")
        print(f"   F1 Score: {best_low_fa['f1_score']:.4f}")
    
    print("\n" + "=" * 100)
    
    return {
        'best_f1': best_f1,
        'best_perfect_recall': best_perfect if perfect_recall else None,
        'best_balanced': best_balanced,
        'best_low_false_alarm': best_low_fa if decent_recall else None
    }


def main():
    """Main testing pipeline."""
    print("=" * 100)
    print("TESTING MULTIPLE THRESHOLDS FOR NBM LSTM")
    print("=" * 100)
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_v2.keras')
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded!")
    
    # Load test data
    print("\nLoading test data...")
    test_events = load_nbm_data()
    print(f"Loaded {len(test_events)} test events")
    
    # Compute scores
    print("\nComputing event scores...")
    scores = compute_event_scores(model, test_events)
    
    # Test different thresholds
    thresholds_to_test = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00, 1.10]
    
    print(f"\nTesting {len(thresholds_to_test)} different thresholds...")
    results = []
    for threshold in thresholds_to_test:
        result = evaluate_at_threshold(scores, threshold)
        results.append(result)
    
    # Print comparison
    print_comparison_table(results)
    
    # Highlight best
    best_thresholds = highlight_best(results)
    
    # Recommendation
    print("\n" + "=" * 100)
    print("💡 RECOMMENDATION")
    print("=" * 100)
    
    if best_thresholds['best_low_false_alarm']:
        rec = best_thresholds['best_low_false_alarm']
        print(f"\nFor fault prediction system, use threshold = {rec['threshold']:.2f}")
        print(f"  ✅ Recall: {rec['recall']:.1%} (catches most faults)")
        print(f"  ✅ False Alarm: {rec['false_alarm_rate']:.1%} (manageable)")
        print(f"  ✅ F1 Score: {rec['f1_score']:.4f}")
    else:
        print("\nUse best F1 threshold for balanced performance")
    
    # Save results
    output_dir = os.path.join(RESULTS_DIR, 'nbm_lstm_v2')
    output_path = os.path.join(output_dir, 'threshold_comparison.json')
    with open(output_path, 'w') as f:
        json.dump({
            'thresholds_tested': thresholds_to_test,
            'results': results,
            'best_thresholds': {k: v for k, v in best_thresholds.items() if v is not None}
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
