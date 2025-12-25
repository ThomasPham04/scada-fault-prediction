"""
Run NBM LSTM (7-day) Model on Test Set - Final Evaluation
Optimized with best threshold (p95) and detailed analysis
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
    print("Loading test events...")
    
    if not os.path.exists(test_dir):
        print(f"Error: Directory not found: {test_dir}")
        return {}

    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            try:
                event_id = int(filename.split('_')[1].split('.')[0])
                data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
                
                if 'X' in data and 'y' in data:
                    test_events[event_id] = {
                        'X': data['X'],
                        'y': data['y'],
                        'label': str(data['label'])
                    }
                else: 
                    print(f"Skipping {filename}: Keys not found")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"  Loaded {len(test_events)} test events")
    return test_events

def determine_threshold(percentile=95):
    """Determine threshold from validation errors."""
    errors_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'nbm_7day_errors.npz')
    
    if not os.path.exists(errors_path):
        print(f"Error: Errors file not found: {errors_path}")
        return None

    data = np.load(errors_path)
    val_mae = data['val_mae']
    
    threshold = np.percentile(val_mae, percentile)
    
    # Also show other percentiles for reference
    print(f"\nValidation MAE Statistics:")
    print(f"  Mean:  {np.mean(val_mae):.6f}")
    print(f"  Std:   {np.std(val_mae):.6f}")
    print(f"  p90:   {np.percentile(val_mae, 90):.6f}")
    print(f"  p95:   {np.percentile(val_mae, 95):.6f}")
    print(f"  p99:   {np.percentile(val_mae, 99):.6f}")
    print(f"  p99.9: {np.percentile(val_mae, 99.9):.6f}")
    
    print(f"\nUsing threshold (p{percentile}): {threshold:.6f}")
    return threshold

def evaluate(model, test_events, threshold):
    """Evaluate model on all test events."""
    print(f"\nEvaluating with threshold: {threshold:.6f}")
    
    TP = FP = TN = FN = 0
    results = []
    
    print("\n" + "=" * 120)
    print(f"{'Event':<8} {'True Label':<12} {'Predicted':<12} {'Result':<10} {'Mean MAE':<12} {'p95 MAE':<12} {'Max MAE':<12} {'Samples':<10}")
    print("-" * 120)
    
    for event_id, data in sorted(test_events.items()):
        X = data['X']
        y = data['y']
        true_label = data['label']
        
        # Predict
        y_pred = model.predict(X, verbose=0, batch_size=256)
        
        # Compute MAE per sample
        mae = np.mean(np.abs(y - y_pred), axis=1)
        
        # Use p95 of event's MAE as anomaly score (best performing threshold)
        event_p95 = np.percentile(mae, 95)
        detected = event_p95 > threshold
        
        # Additional stats for analysis
        mean_mae = np.mean(mae)
        max_mae = np.max(mae)
        
        # Determine confusion matrix type
        is_anomaly = (true_label == 'anomaly')
        if is_anomaly and detected:
            res_type = "TP"
            TP += 1
            symbol = "[OK] TP"
        elif not is_anomaly and not detected:
            res_type = "TN"
            TN += 1
            symbol = "[OK] TN"
        elif not is_anomaly and detected:
            res_type = "FP"
            FP += 1
            symbol = "[X]  FP"
        else:
            res_type = "FN"
            FN += 1
            symbol = "[X]  FN"
            
        print(f"{event_id:<8} {true_label:<12} {'Anomaly' if detected else 'Normal':<12} {symbol:<10} "
              f"{mean_mae:<12.4f} {event_p95:<12.4f} {max_mae:<12.4f} {len(mae):<10}")
        
        results.append({
            'event_id': int(event_id),
            'true_label': true_label,
            'detected': bool(detected),
            'result_type': res_type,
            'mae_mean': float(mean_mae),
            'mae_p95': float(event_p95),
            'mae_max': float(max_mae),
            'n_samples': int(len(mae))
        })

    # Summary
    print("=" * 120)
    print("\nCONFUSION MATRIX:")
    print(f"                    Predicted Anomaly | Predicted Normal")
    print(f"  Actual Anomaly  |       {TP:3d}        |      {FN:3d}")
    print(f"  Actual Normal   |       {FP:3d}        |      {TN:3d}")
    
    # Metrics
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    
    if (TP + FN) > 0: 
        recall = TP / (TP + FN)
    else: 
        recall = 0
    
    if (TP + FP) > 0: 
        precision = TP / (TP + FP)
    else: 
        precision = 0
    
    if (FP + TN) > 0: 
        far = FP / (FP + TN)
    else: 
        far = 0
    
    if (precision + recall) > 0: 
        f1 = 2 * precision * recall / (precision + recall)
    else: 
        f1 = 0
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Accuracy:             {accuracy:.2%}")
    print(f"  Detection Rate:       {recall:.2%} (Recall/Sensitivity)")
    print(f"  False Alarm Rate:     {far:.2%}")
    print(f"  Precision:            {precision:.2%}")
    print(f"  F1 Score:             {f1:.4f}")
    
    print(f"\nDETAILED BREAKDOWN:")
    print(f"  Total Events:         {total}")
    print(f"  Anomalies:            {TP + FN} (Detected: {TP}, Missed: {FN})")
    print(f"  Normal:               {TN + FP} (Correct: {TN}, False Alarms: {FP})")
    
    # Comparison with Naive Baseline
    print("\n" + "=" * 120)
    print("COMPARISON WITH NAIVE BASELINE")
    print("=" * 120)
    
    naive_results_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_test_evaluation.json')
    if os.path.exists(naive_results_path):
        with open(naive_results_path, 'r') as f:
            naive_data = json.load(f)
        
        naive_metrics = naive_data['metrics']
        
        print(f"\n{'Metric':<30} {'Naive Baseline':<20} {'NBM LSTM (7-day)':<20} {'Difference':<15}")
        print("-" * 85)
        print(f"{'Detection Rate (Recall)':<30} {naive_metrics['recall']:<20.2%} {recall:<20.2%} {(recall - naive_metrics['recall'])*100:+.1f}%")
        print(f"{'Precision':<30} {naive_metrics['precision']:<20.2%} {precision:<20.2%} {(precision - naive_metrics['precision'])*100:+.1f}%")
        print(f"{'False Alarm Rate':<30} {naive_metrics['false_alarm_rate']:<20.2%} {far:<20.2%} {(far - naive_metrics['false_alarm_rate'])*100:+.1f}%")
        print(f"{'F1 Score':<30} {naive_metrics['f1_score']:<20.4f} {f1:<20.4f} {f1 - naive_metrics['f1_score']:+.4f}")
        
        # Interpretation
        print("\nINTERPRETATION:")
        if f1 > naive_metrics['f1_score']:
            improvement = (f1 - naive_metrics['f1_score']) / naive_metrics['f1_score'] * 100
            print(f"  ✓ LSTM outperforms Naive Baseline by {improvement:.1f}% (F1 Score)")
        elif f1 < naive_metrics['f1_score']:
            degradation = (naive_metrics['f1_score'] - f1) / naive_metrics['f1_score'] * 100
            print(f"  ✗ LSTM underperforms Naive Baseline by {degradation:.1f}% (F1 Score)")
        else:
            print(f"  = LSTM and Naive Baseline have similar performance")
        
        if recall > naive_metrics['recall']:
            print(f"  ✓ LSTM has better anomaly detection ({recall:.1%} vs {naive_metrics['recall']:.1%})")
        
        if far < naive_metrics['false_alarm_rate']:
            print(f"  ✓ LSTM has lower false alarm rate ({far:.1%} vs {naive_metrics['false_alarm_rate']:.1%})")
    else:
        print("  Naive baseline results not found. Skipping comparison.")
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'lstm_test_evaluation.json')
    save_data = {
        'model': 'NBM_LSTM_7day',
        'threshold': float(threshold),
        'threshold_type': 'p95_of_val_mae',
        'metrics': {
            'accuracy': accuracy,
            'recall': recall, 
            'precision': precision, 
            'far': far, 
            'f1': f1
        },
        'confusion_matrix': {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN},
        'events': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 120)

def main():
    print("=" * 120)
    print("NBM LSTM (7-day) - Event Duration Test Set Evaluation")
    print("Using test data from event_start to event_end only")
    print("=" * 120)
    
    # Load Model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_7day.keras')
    print(f"\nLoading model: {model_path}")
    if not os.path.exists(model_path):
        print("Error: Model file not found!")
        return
        
    model = keras.models.load_model(model_path)
    print("  Model loaded successfully!")
    
    # Load Data
    test_events = load_test_data()
    if not test_events:
        print("Error: No test events loaded!")
        return
        
    # Threshold (p95 gives best F1 score based on multi-threshold analysis)
    threshold = determine_threshold(percentile=95)
    if threshold is None:
        return
        
    # Evaluate
    evaluate(model, test_events, threshold)

if __name__ == "__main__":
    main()
