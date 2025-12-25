"""
Generate Ground Truth Table from Naive Baseline Test Results
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

def print_ground_truth_table():
    """Print a formatted ground truth comparison table."""
    
    # Load results
    results_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_test_evaluation.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract event details
    events = results['event_details']
    
    # Print header
    print("=" * 120)
    print("GROUND TRUTH vs PREDICTION - NAIVE BASELINE (7-day window)")
    print("=" * 120)
    print(f"{'Event':<8} {'True Label':<12} {'Predicted':<12} {'Result':<8} {'Mean Score':<12} {'P95 Score':<12} {'Max Score':<12} {'MSE':<10}")
    print("-" * 120)
    
    # Sort by event_id
    events_sorted = sorted(events, key=lambda x: x['event_id'])
    
    # Print each event
    for event in events_sorted:
        event_id = event['event_id']
        true_label = event['true_label']
        predicted = 'Anomaly' if event['detected'] else 'Normal'
        result_type = event['result_type']
        mean_score = event['mean_score']
        p95_score = event['p95_score']
        max_score = event['max_score']
        mse = event['reconstruction_metrics']['mse']
        
        # Color coding (using symbols for terminal)
        if result_type == 'TP':
            symbol = '[OK] TP'
        elif result_type == 'TN':
            symbol = '[OK] TN'
        elif result_type == 'FP':
            symbol = '[X]  FP'
        else:  # FN
            symbol = '[X]  FN'
        
        print(f"{event_id:<8} {true_label:<12} {predicted:<12} {symbol:<8} {mean_score:<12.4f} {p95_score:<12.4f} {max_score:<12.4f} {mse:<10.4f}")
    
    # Print summary
    print("=" * 120)
    cm = results['confusion_matrix']
    metrics = results['metrics']
    
    print("\nCONFUSION MATRIX:")
    print(f"  True Positive (TP):   {cm['TP']:3d}  - Correctly detected anomalies")
    print(f"  True Negative (TN):   {cm['TN']:3d}  - Correctly identified normal")
    print(f"  False Positive (FP):  {cm['FP']:3d}  - Normal wrongly flagged as anomaly")
    print(f"  False Negative (FN):  {cm['FN']:3d}  - Anomaly missed")
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Accuracy:           {metrics['accuracy']:.2%}")
    print(f"  Detection Rate:     {metrics['detection_rate']:.2%} (Recall/Sensitivity)")
    print(f"  False Alarm Rate:   {metrics['false_alarm_rate']:.2%}")
    print(f"  Precision:          {metrics['precision']:.2%}")
    print(f"  F1 Score:           {metrics['f1_score']:.4f}")
    
    print("\nKEY INSIGHTS:")
    print(f"  • Total Events: {len(events)}")
    print(f"  • Anomalies: {cm['TP'] + cm['FN']} (Detected: {cm['TP']}, Missed: {cm['FN']})")
    print(f"  • Normal: {cm['TN'] + cm['FP']} (Correct: {cm['TN']}, False Alarms: {cm['FP']})")
    print(f"  • Detection Rate: {metrics['detection_rate']:.1%} - Good at finding anomalies")
    print(f"  • False Alarm Rate: {metrics['false_alarm_rate']:.1%} - [!] VERY HIGH - Most normal events flagged as anomaly")
    
    print("=" * 120)
    
    # Detailed breakdown by result type
    print("\nDETAILED BREAKDOWN:")
    
    print("\n[OK] TRUE POSITIVES (Correctly Detected Anomalies):")
    tp_events = [e for e in events_sorted if e['result_type'] == 'TP']
    for e in tp_events:
        print(f"  Event {e['event_id']:3d}: p95={e['p95_score']:7.4f}, MSE={e['reconstruction_metrics']['mse']:7.4f}")
    
    print("\n[X] FALSE NEGATIVES (Missed Anomalies):")
    fn_events = [e for e in events_sorted if e['result_type'] == 'FN']
    for e in fn_events:
        print(f"  Event {e['event_id']:3d}: p95={e['p95_score']:7.4f}, MSE={e['reconstruction_metrics']['mse']:7.4f} - [!] MISSED")
    
    print("\n[OK] TRUE NEGATIVES (Correctly Identified Normal):")
    tn_events = [e for e in events_sorted if e['result_type'] == 'TN']
    for e in tn_events:
        print(f"  Event {e['event_id']:3d}: p95={e['p95_score']:7.4f}, MSE={e['reconstruction_metrics']['mse']:7.4f}")
    
    print("\n[X] FALSE POSITIVES (Normal Wrongly Flagged):")
    fp_events = [e for e in events_sorted if e['result_type'] == 'FP']
    for e in fp_events:
        print(f"  Event {e['event_id']:3d}: p95={e['p95_score']:7.4f}, MSE={e['reconstruction_metrics']['mse']:7.4f} - [!] FALSE ALARM")
    
    print("=" * 120)


if __name__ == "__main__":
    print_ground_truth_table()
