"""
Compare p95 vs p99 threshold for Naive Baseline
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

def evaluate_with_threshold(events, threshold, threshold_name):
    """Evaluate performance with a given threshold."""
    TP = FP = TN = FN = 0
    
    for event in events:
        p95_score = event['p95_score']
        is_anomaly = (event['true_label'] == 'anomaly')
        detected = p95_score > threshold
        
        if is_anomaly and detected:
            TP += 1
        elif is_anomaly and not detected:
            FN += 1
        elif not is_anomaly and detected:
            FP += 1
        else:
            TN += 1
    
    # Calculate metrics
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = detection_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': threshold,
        'threshold_name': threshold_name,
        'confusion_matrix': {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN},
        'metrics': {
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    }

def print_comparison():
    """Compare p95 vs p99 thresholds."""
    
    # Load test results
    test_results_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_test_evaluation.json')
    with open(test_results_path, 'r') as f:
        test_results = json.load(f)
    
    events = test_results['event_details']
    
    # Load baseline results to get p99
    baseline_results_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_baseline_results.json')
    with open(baseline_results_path, 'r') as f:
        baseline_results = json.load(f)
    
    p95_threshold = baseline_results['val']['p95']
    p99_threshold = baseline_results['val']['p99']
    
    # Evaluate with both thresholds
    p95_eval = evaluate_with_threshold(events, p95_threshold, 'p95')
    p99_eval = evaluate_with_threshold(events, p99_threshold, 'p99')
    
    # Print comparison
    print("=" * 100)
    print("THRESHOLD COMPARISON: p95 vs p99")
    print("=" * 100)
    
    print(f"\nThreshold Values:")
    print(f"  p95 (95th percentile): {p95_threshold:.4f}")
    print(f"  p99 (99th percentile): {p99_threshold:.4f}")
    print(f"  Difference: {p99_threshold - p95_threshold:.4f} ({(p99_threshold/p95_threshold - 1)*100:.1f}% higher)")
    
    print("\n" + "=" * 100)
    print(f"{'Metric':<30} {'p95 Threshold':<20} {'p99 Threshold':<20} {'Change':<20}")
    print("-" * 100)
    
    # Confusion matrix
    cm_p95 = p95_eval['confusion_matrix']
    cm_p99 = p99_eval['confusion_matrix']
    
    print(f"{'True Positive (TP)':<30} {cm_p95['TP']:<20} {cm_p99['TP']:<20} {cm_p99['TP'] - cm_p95['TP']:+d}")
    print(f"{'False Positive (FP)':<30} {cm_p95['FP']:<20} {cm_p99['FP']:<20} {cm_p99['FP'] - cm_p95['FP']:+d}")
    print(f"{'True Negative (TN)':<30} {cm_p95['TN']:<20} {cm_p99['TN']:<20} {cm_p99['TN'] - cm_p95['TN']:+d}")
    print(f"{'False Negative (FN)':<30} {cm_p95['FN']:<20} {cm_p99['FN']:<20} {cm_p99['FN'] - cm_p95['FN']:+d}")
    
    print("-" * 100)
    
    # Performance metrics
    m_p95 = p95_eval['metrics']
    m_p99 = p99_eval['metrics']
    
    print(f"{'Accuracy':<30} {m_p95['accuracy']:<20.2%} {m_p99['accuracy']:<20.2%} {(m_p99['accuracy'] - m_p95['accuracy'])*100:+.1f}%")
    print(f"{'Detection Rate (Recall)':<30} {m_p95['detection_rate']:<20.2%} {m_p99['detection_rate']:<20.2%} {(m_p99['detection_rate'] - m_p95['detection_rate'])*100:+.1f}%")
    print(f"{'False Alarm Rate':<30} {m_p95['false_alarm_rate']:<20.2%} {m_p99['false_alarm_rate']:<20.2%} {(m_p99['false_alarm_rate'] - m_p95['false_alarm_rate'])*100:+.1f}%")
    print(f"{'Precision':<30} {m_p95['precision']:<20.2%} {m_p99['precision']:<20.2%} {(m_p99['precision'] - m_p95['precision'])*100:+.1f}%")
    print(f"{'F1 Score':<30} {m_p95['f1_score']:<20.4f} {m_p99['f1_score']:<20.4f} {m_p99['f1_score'] - m_p95['f1_score']:+.4f}")
    
    print("=" * 100)
    
    # Analysis
    print("\nANALYSIS:")
    print("-" * 100)
    
    # Which is better?
    if m_p99['f1_score'] > m_p95['f1_score']:
        better = "p99"
        worse = "p95"
        better_eval = p99_eval
        worse_eval = p95_eval
    else:
        better = "p95"
        worse = "p99"
        better_eval = p95_eval
        worse_eval = p99_eval
    
    print(f"\n[WINNER] {better.upper()} threshold performs better (F1: {better_eval['metrics']['f1_score']:.4f})")
    
    print(f"\nTrade-offs when using p99 instead of p95:")
    print(f"  [+] Detection Rate: {m_p95['detection_rate']:.1%} -> {m_p99['detection_rate']:.1%} ({(m_p99['detection_rate'] - m_p95['detection_rate'])*100:+.1f}%)")
    print(f"  [+] False Alarm Rate: {m_p95['false_alarm_rate']:.1%} -> {m_p99['false_alarm_rate']:.1%} ({(m_p99['false_alarm_rate'] - m_p95['false_alarm_rate'])*100:+.1f}%)")
    print(f"  [+] Precision: {m_p95['precision']:.1%} -> {m_p99['precision']:.1%} ({(m_p99['precision'] - m_p95['precision'])*100:+.1f}%)")
    
    if cm_p99['FP'] < cm_p95['FP']:
        print(f"\n  [GOOD] Reduces False Positives: {cm_p95['FP']} -> {cm_p99['FP']} ({cm_p99['FP'] - cm_p95['FP']:+d})")
    else:
        print(f"\n  [BAD] Increases False Positives: {cm_p95['FP']} -> {cm_p99['FP']} ({cm_p99['FP'] - cm_p95['FP']:+d})")
    
    if cm_p99['FN'] < cm_p95['FN']:
        print(f"  [GOOD] Reduces Missed Anomalies: {cm_p95['FN']} -> {cm_p99['FN']} ({cm_p99['FN'] - cm_p95['FN']:+d})")
    else:
        print(f"  [BAD] Increases Missed Anomalies: {cm_p95['FN']} -> {cm_p99['FN']} ({cm_p99['FN'] - cm_p95['FN']:+d})")
    
    print("\n" + "=" * 100)
    
    # Recommendation
    print("\nRECOMMENDATION:")
    print("-" * 100)
    
    if m_p99['false_alarm_rate'] < m_p95['false_alarm_rate'] and m_p99['detection_rate'] >= 0.7:
        print("[RECOMMEND p99] Use p99 threshold:")
        print(f"  - Significantly reduces false alarms ({m_p95['false_alarm_rate']:.1%} -> {m_p99['false_alarm_rate']:.1%})")
        print(f"  - Maintains acceptable detection rate ({m_p99['detection_rate']:.1%})")
        print(f"  - Better precision ({m_p99['precision']:.1%})")
    elif m_p95['detection_rate'] > m_p99['detection_rate'] and m_p95['false_alarm_rate'] < 0.8:
        print("[RECOMMEND p95] Keep p95 threshold:")
        print(f"  - Higher detection rate ({m_p95['detection_rate']:.1%})")
        print(f"  - Acceptable false alarm rate")
    else:
        print("[NEUTRAL] Both thresholds have trade-offs:")
        print(f"  - p95: Better detection ({m_p95['detection_rate']:.1%}) but more false alarms ({m_p95['false_alarm_rate']:.1%})")
        print(f"  - p99: Fewer false alarms ({m_p99['false_alarm_rate']:.1%}) but lower detection ({m_p99['detection_rate']:.1%})")
        print("  - Choose based on your priority: safety (p95) vs operational cost (p99)")
    
    print("=" * 100)


if __name__ == "__main__":
    print_comparison()
