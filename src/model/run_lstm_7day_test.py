"""
Run NBM LSTM (7-day) Model on Test Set - Final Evaluation
Optimized with dynamic threshold from PR curve on val set
"""

import os
import sys
import numpy as np
import json
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR

def load_events(data_split='test'):
    """Load events from NBM 7-day (test or val)."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    split_dir = os.path.join(nbm_dir, f'{data_split}_by_event')
    
    events = {}
    print(f"Loading {data_split} events...")
    
    if not os.path.exists(split_dir):
        print(f"Error: Directory not found: {split_dir}")
        return {}

    for filename in os.listdir(split_dir):
        if filename.endswith('.npz'):
            try:
                event_id = int(filename.split('_')[1].split('.')[0])
                data = np.load(os.path.join(split_dir, filename), allow_pickle=True)
                
                if 'X' in data and 'y' in data:
                    events[event_id] = {
                        'X': data['X'],
                        'y': data['y'],
                        'label': str(data['label'])
                    }
                else: 
                    print(f"Skipping {filename}: Keys not found")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"  Loaded {len(events)} {data_split} events")
    return events

def smooth_mae(mae, window=3):
    """Simple moving average smoothing for MAE to reduce noise spikes."""
    if len(mae) <= window:
        return mae
    return np.convolve(mae, np.ones(window)/window, mode='same')

def compute_event_scores(model, events):
    """Compute event_p95 MAE scores for PR curve or evaluation."""
    scores = []
    labels = []
    detailed = []  # For val stats
    
    for event_id, data in sorted(events.items()):
        X = data['X']
        y = data['y']
        true_label = data['label']
        
        # Predict
        y_pred = model.predict(X, verbose=0, batch_size=256)
        
        # Compute MAE per sample
        mae = np.mean(np.abs(y - y_pred), axis=1)
        
        # Optional: Smooth MAE
        mae = smooth_mae(mae, window=3)
        
        # Event score: p95 MAE (consistent with original)
        event_p95 = np.percentile(mae, 95)
        
        scores.append(event_p95)
        labels.append(1 if true_label == 'anomaly' else 0)
        
        # Detailed for output
        detailed.append({
            'event_id': int(event_id),
            'true_label': true_label,
            'mae_mean': float(np.mean(mae)),
            'mae_p95': float(event_p95),
            'mae_max': float(np.max(mae)),
            'n_samples': int(len(mae))
        })
    
    return np.array(scores), np.array(labels), detailed

def determine_threshold(model, val_events, min_recall=0.7, smoothing_window=3):
    print("\nComputing val scores for threshold optimization...")
    val_scores, val_labels, val_detailed = compute_event_scores(model, val_events)
    
    # PR Curve
    prec, rec, thresholds = precision_recall_curve(val_labels, val_scores)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-6)
    
    # Find best threshold: max F1 with recall >= min_recall
    valid_idx = np.where(rec >= min_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
        opt_threshold = thresholds[best_idx]
        best_prec = prec[best_idx]
        best_rec = rec[best_idx]
        best_f1 = f1_scores[best_idx]
    else:
        # FIXED: Fallback to p99 for strict FAR reduction (instead of max F1)
        opt_threshold = np.percentile(val_scores, 99)
        best_prec = np.mean(val_labels)  # Approximate
        best_rec = 0.5  # Approximate
        best_f1 = 0.0
        print(f"Warning: No threshold meets recall >= {min_recall}. Using p99 fallback for low FAR: {opt_threshold:.6f}")
    
    # Val MAE stats (giữ nguyên)
    print(f"\nValidation MAE Statistics (event_p95):")
    print(f"  Mean:  {np.mean(val_scores):.6f}")
    print(f"  Std:   {np.std(val_scores):.6f}")
    print(f"  p90:   {np.percentile(val_scores, 90):.6f}")
    print(f"  p95:   {np.percentile(val_scores, 95):.6f}")
    print(f"  p99:   {np.percentile(val_scores, 99):.6f}")
    
    print(f"\nOptimized Threshold (PR Curve, min_recall={min_recall}): {opt_threshold:.6f}")
    print(f"  Val Prec: {best_prec:.4f} | Rec: {best_rec:.4f} | F1: {best_f1:.4f}")
    
    # Save val detailed (giữ nguyên)
    val_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'val_detailed.json')
    with open(val_path, 'w') as f:
        json.dump({'val_detailed': val_detailed}, f, indent=2)
    print(f"Val details saved to: {val_path}")
    
    return opt_threshold

def evaluate(model, test_events, threshold, min_recall=0.7, smoothing_window=3):
    """Evaluate model on test events (giống code cũ, nhưng dùng smooth nếu cần)."""
    print(f"\nEvaluating with optimized threshold: {threshold:.6f}")
    
    TP = FP = TN = FN = 0
    results = []
    
    print("\n" + "=" * 120)
    print(f"{'Event':<8} {'True Label':<12} {'Predicted':<12} {'Result':<10} {'Mean MAE':<12} {'p95 MAE':<12} {'Max MAE':<12} {'Samples':<10}")
    print("-" * 120)
    
    for event_id, data in sorted(test_events.items()):
        X = data['X']
        y = data['y']
        true_label = data['label']
        
        # Predict & MAE (with smoothing)
        y_pred = model.predict(X, verbose=0, batch_size=256)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        mae = smooth_mae(mae, window=smoothing_window)
        
        # Event score
        event_p95 = np.percentile(mae, 95)
        above_th_count = np.sum(mae > threshold)
        duration = above_th_count / len(mae)
        detected = (event_p95 > threshold) and (duration >= 0.25)
        
        # Stats
        mean_mae = np.mean(mae)
        max_mae = np.max(mae)
        
        # Confusion
        is_anomaly = (true_label == 'anomaly')
        if is_anomaly and detected:
            res_type = "TP"; TP += 1; symbol = "[OK] TP"
        elif not is_anomaly and not detected:
            res_type = "TN"; TN += 1; symbol = "[OK] TN"
        elif not is_anomaly and detected:
            res_type = "FP"; FP += 1; symbol = "[X]  FP"
        else:
            res_type = "FN"; FN += 1; symbol = "[X]  FN"
            
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

    # Metrics (giống code cũ)
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("=" * 120)
    print("\nCONFUSION MATRIX:")
    print(f"                    Predicted Anomaly | Predicted Normal")
    print(f"  Actual Anomaly  |       {TP:3d}        |      {FN:3d}")
    print(f"  Actual Normal   |       {FP:3d}        |      {TN:3d}")
    
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
    
    # Naive baseline comparison (giữ nguyên)
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
    output_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'lstm_test_evaluation_optimized.json')
    save_data = {
        'model': 'NBM_LSTM_7day_Optimized',
        'threshold': float(threshold),
        'threshold_type': 'PR_Curve_Optimized',
        'min_recall_target': min_recall,  # Fixed: now param
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
    print("NBM LSTM (7-day) - Optimized Threshold Evaluation")
    print("Using PR curve on val set for dynamic threshold")
    print("=" * 120)
    
    # Load Model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_7day.keras')
    print(f"\nLoading model: {model_path}")
    if not os.path.exists(model_path):
        print("Error: Model file not found!")
        return
        
    model = keras.models.load_model(model_path)
    print("  Model loaded successfully!")
    
    # Load Val & Test Data
    val_events = load_events('val')
    test_events = load_events('test')
    
    if not val_events:
        print("Error: No val events loaded! Need val_by_event dir.")
        return
    if not test_events:
        print("Error: No test events loaded!")
        return
        
    # Optimize Threshold (min_recall=0.7, adjust if needed)
    threshold = determine_threshold(model, val_events, min_recall=0.7, smoothing_window=5)
    if threshold is None:
        return
        
    # Evaluate Test (Fixed: pass min_recall)
    evaluate(model, test_events, threshold, min_recall=0.7, smoothing_window=5)

if __name__ == "__main__":
    main()