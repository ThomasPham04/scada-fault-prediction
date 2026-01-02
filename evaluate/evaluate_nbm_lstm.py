"""
Evaluate NBM LSTM Model on Test Set

Load the trained NBM LSTM model and evaluate it on the test set.
Compute performance metrics and compare with baseline.
"""

import os
import sys
import numpy as np
import json
import joblib
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR


def load_nbm_data():
    """Load NBM V2 test data and metadata."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    
    print("Loading NBM V2 data...")
    
    # Load test events
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    test_events = {}
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            
            test_events[event_id] = {
                'X': data['X'],    # Sequences (n_samples, window_size, n_features)
                'y': data['y'],    # Targets (n_samples, n_features)
                'label': str(data['label'])
            }
    
    # Load metadata
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_v2.pkl'))
    
    print(f"  Test events loaded: {len(test_events)}")
    
    return test_events, metadata


def compute_sample_errors(model, X, y):
    """
    Compute prediction errors for each sample.
    
    Returns:
        mae: Mean Absolute Error per sample
        mse: Mean Squared Error per sample
    """
    # Predict
    y_pred = model.predict(X, verbose=0)
    
    # Error per sample (averaged across features)
    mae = np.mean(np.abs(y - y_pred), axis=1)
    mse = np.mean((y - y_pred) ** 2, axis=1)
    
    return mae, mse, y_pred


def evaluate_event(model, event_data, threshold):
    """
    Evaluate model on a single event.
    
    Args:
        model: Trained LSTM model
        event_data: Dict with 'X', 'y', 'label'
        threshold: Anomaly detection threshold
        
    Returns:
        Event evaluation results
    """
    X = event_data['X']
    y = event_data['y']
    
    # Compute errors
    mae, mse, y_pred = compute_sample_errors(model, X, y)
    
    # Event statistics
    mean_mae = np.mean(mae)
    max_mae = np.max(mae)
    p95_mae = np.percentile(mae, 95)
    
    # Decision: Use p95 for detection (same as threshold calculation)
    detected = p95_mae > threshold
    
    return {
        'mae': mae,
        'mse': mse,
        'mean_mae': float(mean_mae),
        'max_mae': float(max_mae),
        'p95_mae': float(p95_mae),
        'detected': detected,
        'n_samples': len(mae)
    }


def run_evaluation(model, test_events, threshold):
    """Run evaluation on all test events."""
    print(f"\nEvaluating with threshold: {threshold:.6f}")
    print("=" * 70)
    
    # Track confusion matrix
    TP = 0  # True Positive: Correctly detected anomaly
    FP = 0  # False Positive: False alarm
    TN = 0  # True Negative: Correctly identified normal
    FN = 0  # False Negative: Missed anomaly
    
    # Detailed results
    results = []
    
    for event_id, event_data in sorted(test_events.items()):
        # Evaluate this event
        eval_result = evaluate_event(model, event_data, threshold)
        
        # True label
        is_anomaly = (event_data['label'] == 'anomaly')
        detected = eval_result['detected']
        
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
            'true_label': event_data['label'],
            'detected': bool(detected),
            'result_type': result_type,
            'mean_mae': eval_result['mean_mae'],
            'max_mae': eval_result['max_mae'],
            'p95_mae': eval_result['p95_mae'],
            'n_samples': eval_result['n_samples']
        })
        
        # Print event details
        status = "✓" if (is_anomaly == detected) else "✗"
        print(f"{status} Event {event_id:3d} | True: {event_data['label']:8s} | "
              f"Detected: {'Anomaly' if detected else 'Normal':8s} | "
              f"p95_mae={eval_result['p95_mae']:.6f} | ({result_type})")
    
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
    print("NBM LSTM MODEL - TEST SET EVALUATION")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_v2.keras')
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"  Model loaded successfully!")
    print(f"  Model input shape: {model.input_shape}")
    print(f"  Model output shape: {model.output_shape}")
    
    # Load threshold from training results
    results_path = os.path.join(RESULTS_DIR, 'nbm_lstm', 'nbm_lstm_evaluation.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            training_results = json.load(f)
        
        # Try to get threshold from stored value or calculate from val errors
        if 'threshold' in training_results:
            threshold = training_results['threshold']
        elif 'val_errors' in training_results:
            threshold = training_results['val_errors']['mae_p95']
        else:
            # Use validation MAE as fallback
            threshold = training_results['val']['mae']
    else:
        # Default threshold if no results file
        print("  Warning: No training results found, using default threshold")
        threshold = 0.5
    
    print(f"  Using threshold (Val MAE threshold): {threshold:.6f}")
    
    # Load test data
    test_events, metadata = load_nbm_data()
    
    # Run evaluation
    eval_results = run_evaluation(model, test_events, threshold)
    
    # Save detailed results
    output_dir = os.path.join(RESULTS_DIR, 'nbm_lstm_v2')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'nbm_lstm_test_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Compare with Naive Baseline
    print("\n" + "=" * 70)
    print("COMPARISON WITH NAIVE BASELINE")
    print("=" * 70)
    
    naive_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_test_evaluation.json')
    if os.path.exists(naive_path):
        with open(naive_path, 'r') as f:
            naive_results = json.load(f)
        
        print("\n                      NBM LSTM  |  Naive Baseline")
        print("-" * 70)
        print(f"Accuracy:            {eval_results['metrics']['accuracy']:6.2%}    |    {naive_results['metrics']['accuracy']:6.2%}")
        print(f"Detection Rate:      {eval_results['metrics']['detection_rate']:6.2%}    |    {naive_results['metrics']['detection_rate']:6.2%}")
        print(f"False Alarm Rate:    {eval_results['metrics']['false_alarm_rate']:6.2%}    |    {naive_results['metrics']['false_alarm_rate']:6.2%}")
        print(f"Precision:           {eval_results['metrics']['precision']:6.2%}    |    {naive_results['metrics']['precision']:6.2%}")
        print(f"F1 Score:            {eval_results['metrics']['f1_score']:6.4f}    |    {naive_results['metrics']['f1_score']:6.4f}")
        print("=" * 70)
        
        # Calculate improvement
        f1_improvement = ((eval_results['metrics']['f1_score'] - naive_results['metrics']['f1_score']) 
                         / naive_results['metrics']['f1_score'] * 100)
        print(f"\nF1 Score Improvement: {f1_improvement:+.1f}%")


if __name__ == "__main__":
    main()
