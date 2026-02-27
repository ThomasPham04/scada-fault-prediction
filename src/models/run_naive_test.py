"""
Run Naive Baseline Model on Test Set
Direct evaluation using the same code structure as baseline_naive.py
"""

import os
import sys
import numpy as np
import joblib
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR

# Import the detector class
from baseline_naive import NaiveThresholdDetector


def load_test_data():
    """Load test events from NBM V2."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    
    test_events = {}
    
    print("Loading test events...")
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            y_test = data['y']  # Using last timestep only
            
            test_events[event_id] = {
                'X': y_test,
                'label': str(data['label'])
            }
    
    print(f"  Loaded {len(test_events)} test events")
    return test_events


def recreate_model_from_results():
    """Recreate the model using saved statistics."""
    # Load saved results to get the trained statistics
    scores_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_baseline_scores.npz')
    
    # We need to retrain using the mean and std
    # Load training data
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    y_train = np.load(os.path.join(nbm_dir, 'y_train.npy'))
    
    # Create and train detector
    detector = NaiveThresholdDetector()
    detector.fit(y_train)
    
    return detector


def compute_reconstruction_metrics(detector, data):
    """
    Compute reconstruction metrics (MSE, MAE, RMSE).
    
    Naive baseline predicts the mean value for all samples.
    
    Args:
        detector: Trained NaiveThresholdDetector
        data: Actual data array (n_samples, n_features)
    
    Returns:
        dict with mse, mae, rmse
    """
    # Naive baseline prediction = mean values
    predictions = np.tile(detector.mean_, (len(data), 1))
    
    # Compute errors
    errors = data - predictions
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse)
    }


def run_evaluation(detector, test_events, threshold):
    """Run evaluation on test events."""
    print(f"\nEvaluating with threshold: {threshold:.4f}")
    print("=" * 70)
    
    # Track confusion matrix
    TP = 0  # True Positive: Correctly detected anomaly
    FP = 0  # False Positive: False alarm
    TN = 0  # True Negative: Correctly identified normal
    FN = 0  # False Negative: Missed anomaly
    
    # Detailed results
    results = []
    
    # Track reconstruction metrics
    all_recon_metrics = []
    
    for event_id, data in sorted(test_events.items()):
        # Compute anomaly scores for this event
        scores = detector.compute_anomaly_score(data['X'])
        
        # Compute reconstruction metrics for this event
        recon_metrics = compute_reconstruction_metrics(detector, data['X'])
        all_recon_metrics.append(recon_metrics)
        
        # Event statistics
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        p95_score = np.percentile(scores, 95)
        
        # Decision: Use p95 for detection (same as threshold calculation)
        detected = p95_score > threshold
        
        # True label
        is_anomaly = (data['label'] == 'anomaly')
        
        # Update confusion matrix
        if is_anomaly and detected:
            TP += 1
            result_type = "TP"
        elif is_anomaly and not detected:
            FN += 1
            result_type = "FN"
        elif not is_anomaly and detected:
            FP += 1
            result_type = "FP"
        else:
            TN += 1
            result_type = "TN"
        
        # Store results
        results.append({
            'event_id': event_id,
            'true_label': data['label'],
            'detected': bool(detected),  # Convert to Python bool for JSON
            'result_type': result_type,
            'mean_score': float(mean_score),
            'max_score': float(max_score),
            'p95_score': float(p95_score),
            'n_samples': int(len(scores)),
            'reconstruction_metrics': recon_metrics
        })
        
        # Print event details
        status = "✓" if (is_anomaly == detected) else "✗"
        print(f"{status} Event {event_id:3d} | True: {data['label']:8s} | "
              f"Detected: {'Anomaly' if detected else 'Normal':8s} | "
              f"p95={p95_score:.4f} | MSE={recon_metrics['mse']:.4f} MAE={recon_metrics['mae']:.4f} | ({result_type})")
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX:")
    print(f"                  Predicted Anomaly | Predicted Normal")
    print(f"Actual Anomaly  |       {TP:3d}        |      {FN:3d}")
    print(f"Actual Normal   |       {FP:3d}        |      {TN:3d}")
    print("=" * 70)
    
    # Performance metrics
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = detection_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Accuracy:          {accuracy:.2%}")
    print(f"  Detection Rate:    {detection_rate:.2%} (Sensitivity/Recall)")
    print(f"  False Alarm Rate:  {false_alarm_rate:.2%}")
    print(f"  Precision:         {precision:.2%}")
    print(f"  F1 Score:          {f1:.4f}")
    print("=" * 70)
    
    # Print reconstruction metrics summary
    avg_mse = np.mean([m['mse'] for m in all_recon_metrics])
    avg_mae = np.mean([m['mae'] for m in all_recon_metrics])
    avg_rmse = np.mean([m['rmse'] for m in all_recon_metrics])
    
    print("\nRECONSTRUCTION METRICS (Average across all test events):")
    print(f"  MSE:   {avg_mse:.6f}")
    print(f"  MAE:   {avg_mae:.6f}")
    print(f"  RMSE:  {avg_rmse:.6f}")
    print("=" * 70)
    
    return {
        'confusion_matrix': {
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN
        },
        'metrics': {
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'event_details': results
    }


def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("NAIVE BASELINE MODEL - TEST SET EVALUATION")
    print("=" * 70)
    
    # Recreate model
    print("\nRecreating model from training data...")
    detector = recreate_model_from_results()
    print("  Model recreated successfully!")
    
    # Load threshold from validation results
    results_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_baseline_results.json')
    with open(results_path, 'r') as f:
        saved_results = json.load(f)
    
    threshold = saved_results['val']['p95']
    print(f"  Using threshold (Val 95th percentile): {threshold:.4f}")
    
    # Load test data
    test_events = load_test_data()
    
    # Run evaluation
    eval_results = run_evaluation(detector, test_events, threshold)
    
    # Save detailed results
    output_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_test_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
