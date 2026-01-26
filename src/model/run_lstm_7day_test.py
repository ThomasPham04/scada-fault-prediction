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

def determine_threshold(
    model,
    val_events,
    min_recall=0.7,
    smoothing_window=3,
    iqr_multiplier=1.5
):
    print("\n[Threshold] Computing validation scores...")
    val_scores, val_labels, val_detailed = compute_event_scores(model, val_events)

    val_scores = np.asarray(val_scores)
    val_labels = np.asarray(val_labels)  # 0 = normal, 1 = anomaly

    # ======================================================
    # 1. PR-based threshold (recall-constrained)
    # ======================================================
    prec, rec, thresholds = precision_recall_curve(val_labels, val_scores)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-6)

    valid_idx = np.where(rec >= min_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
    else:
        best_idx = np.argmax(f1_scores)
        print(
            f"[WARN] No PR threshold meets recall >= {min_recall}. "
            f"Using max-F1 fallback."
        )

    pr_upper_th = thresholds[best_idx]
    best_prec = prec[best_idx]
    best_rec = rec[best_idx]
    best_f1 = f1_scores[best_idx]

    # ======================================================
    # 2. IQR threshold (NORMAL ONLY)
    # ======================================================
    print("\n[Threshold] Calculating IQR using NORMAL validation events only...")
    normal_scores = val_scores[val_labels == 0]

    if len(normal_scores) >= 4:
        Q1 = np.percentile(normal_scores, 25)
        Q3 = np.percentile(normal_scores, 75)
        IQR_val = Q3 - Q1

        iqr_upper = Q3 + iqr_multiplier * IQR_val
        iqr_lower = max(0.0, Q1 - iqr_multiplier * IQR_val)

        print(f"  #Normal events: {len(normal_scores)}")
        print(f"  Q1: {Q1:.6f}")
        print(f"  Q3: {Q3:.6f}")
        print(f"  IQR: {IQR_val:.6f}")
        print(f"  IQR Upper: {iqr_upper:.6f}")
    else:
        iqr_upper = np.percentile(normal_scores, 99) if len(normal_scores) > 0 else np.percentile(val_scores, 99)
        iqr_lower = 0.0
        print(
            "[WARN] Too few normal events for IQR. "
            f"Using p99 fallback: upper={iqr_upper:.6f}"
        )

    # ======================================================
    # 3. HYBRID threshold
    # ======================================================
    final_upper = min(iqr_upper, pr_upper_th)
    final_lower = iqr_lower  # usually unused for MAE, kept for completeness

    print("\n[Threshold] HYBRID THRESHOLD SELECTION")
    print(f"  IQR Upper (normal-only): {iqr_upper:.6f}")
    print(f"  PR Upper  (rec>={min_recall}): {pr_upper_th:.6f}")
    print(f"  ==> Final Upper Threshold: {final_upper:.6f}")

    print(
        f"  PR Performance @ chosen region: "
        f"Prec={best_prec:.4f}, Rec={best_rec:.4f}, F1={best_f1:.4f}"
    )

    # ======================================================
    # 4. Statistics (for report & debugging)
    # ======================================================
    print("\nValidation Score Statistics:")
    print(f"  Mean (all):    {np.mean(val_scores):.6f}")
    print(f"  Mean (normal): {np.mean(normal_scores):.6f}")
    print(f"  p95  (normal): {np.percentile(normal_scores, 95):.6f}")
    print(f"  p99  (normal): {np.percentile(normal_scores, 99):.6f}")

    # ======================================================
    # 5. Save
    # ======================================================
    val_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'val_detailed.json')
    with open(val_path, 'w') as f:
        json.dump({
            'method': 'IQR_NORMAL_PLUS_PR_HYBRID',
            'upper_threshold': float(final_upper),
            'lower_threshold': float(final_lower),
            'iqr_upper': float(iqr_upper),
            'pr_upper': float(pr_upper_th),
            'iqr_multiplier': iqr_multiplier,
            'min_recall_target': min_recall,
            'val_detailed': val_detailed
        }, f, indent=2)

    print(f"\n[Threshold v3] Saved validation details to: {val_path}")

    return final_upper, final_lower


def longest_run(mask):
    """Return the longest consecutive True run in a boolean mask.
    For Example: [False, True, True, False, True] -> 2
    """
    max_run = run = 0
    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def evaluate(model, test_events, upper_threshold, lower_threshold, min_recall=0.7, smoothing_window=3):
    """Evaluate model on test events with global IQR thresholds (like author's val-based adaptive)."""
    print(f"\nEvaluating with global IQR thresholds: upper={upper_threshold:.6f}, lower={lower_threshold:.6f}")
    
    TP = FP = TN = FN = 0
    results = []
    
    print("\n" + "=" * 140)
    print(f"{'Event':<8} {'True Label':<12} {'Predicted':<12} {'Result':<10} {'Mean MAE':<12} {'p95 MAE':<12} {'Max MAE':<12} {'Samples':<10}")
    print("-" * 140)
    
    for event_id, data in sorted(test_events.items()):
        X = data['X']
        y = data['y']
        true_label = data['label']
        
        # Predict & MAE (with smoothing)
        y_pred = model.predict(X, verbose=0, batch_size=256)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        mae = smooth_mae(mae, window=smoothing_window)
        
        event_p90 = np.percentile(mae, 90)
        event_p95 = np.percentile(mae, 95)

        outlier_ratio = np.mean(mae > upper_threshold) # Calculate the ratio of samples exceeding the upper threshold
        min_ratio = max(0.15, 5 / len(mae)) # The outlier ratio must be greater than or equal to this value
        
        above_mask = mae > upper_threshold
        longest_anomaly_run = longest_run(above_mask) # The longest consecutive run of anomalies
        min_run = max(3, int(0.2 * len(mae))) # The longest run must be at least this long

        severity_score = 0.6 * event_p95 + 0.4 * event_p90

        # The Final decision, anomalies found if: 
            # The severity score exceeds the upper threshold, 
            # The outlier ratio is sufficient, and 
            # The longest run is long enough
        detected = (
            severity_score > upper_threshold and
            outlier_ratio >= min_ratio and
            longest_anomaly_run >= min_run
        )

        
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
            
        print(f"{event_id:<8} {true_label:<12} {'anomaly' if detected else 'normal':<12} {symbol:<10} "
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

    # Metrics
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("=" * 140)
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
    
    # Naive baseline comparison
    print("\n" + "=" * 140)
    print("COMPARISON WITH NAIVE BASELINE")
    print("=" * 140)
    
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
        'upper_threshold': float(upper_threshold),
        'lower_threshold': float(lower_threshold),
        'threshold_type': 'IQR_Adaptive_Global',
        'min_recall_target': min_recall,
        'iqr_multiplier': 1.5,
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
    print("=" * 140)

def main():
    print("=" * 120)
    print("NBM LSTM (7-day) - Optimized Threshold Evaluation")
    print("Using IQR adaptive threshold on val set")
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
        
    # Optimize Threshold (min_recall=0.7, adjust if needed; smoothing=5 for robustness)
    upper_th, lower_th = determine_threshold(model, val_events, min_recall=0.7, smoothing_window=5)
    if upper_th is None:
        return
        
    # Evaluate Test (Fixed: pass min_recall; use upper_th for detect MAE > th)
    evaluate(model, test_events, upper_th, lower_th, min_recall=0.7, smoothing_window=5)

if __name__ == "__main__":
    main()