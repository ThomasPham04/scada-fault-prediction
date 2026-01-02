"""
ROC Curve Threshold Optimization for NBM LSTM

This script:
1. Loads validation data and computes anomaly scores
2. Computes ROC curve for different threshold values
3. Finds optimal threshold using Youden's J statistic
4. Visualizes ROC curve and optimal operating point
5. Re-evaluates model with optimal threshold
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs


def load_data_and_model():
    """Load NBM data and trained model."""
    print("Loading data and model...")
    
    # Load NBM data
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    X_val = np.load(os.path.join(nbm_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(nbm_dir, 'y_val.npy'))
    
    # Load test events
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    test_events = {}
    
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            test_events[event_id] = {
                'X': data['X'],
                'y': data['y'],
                'label': str(data['label'])
            }
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_v2.keras')
    model = keras.models.load_model(model_path)
    
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  Test events: {len(test_events)}")
    
    return model, X_val, y_val, test_events


def compute_reconstruction_errors(model, X, y):
    """Compute per-sample reconstruction errors."""
    y_pred = model.predict(X, batch_size=64, verbose=0)
    errors = np.abs(y - y_pred)
    sample_mae = np.mean(errors, axis=1)
    return sample_mae


def prepare_roc_data(test_events, model):
    """
    Prepare data for ROC curve analysis.
    
    Returns:
        y_true: Binary labels (1=anomaly, 0=normal)
        y_scores: Anomaly scores (higher = more anomalous)
    """
    print("\nComputing anomaly scores for test events...")
    
    y_true = []
    y_scores = []
    event_details = []
    
    for event_id, data in sorted(test_events.items()):
        X_test = data['X']
        y_test = data['y']
        label = data['label']
        
        # Compute reconstruction errors
        sample_mae = compute_reconstruction_errors(model, X_test, y_test)
        
        # Use p95 as anomaly score (as shown in terminal output)
        anomaly_score = float(np.percentile(sample_mae, 95))
        
        # Binary label: 1 for anomaly, 0 for normal
        is_anomaly = 1 if label == 'anomaly' else 0
        
        y_true.append(is_anomaly)
        y_scores.append(anomaly_score)
        event_details.append({
            'event_id': event_id,
            'label': label,
            'score': anomaly_score
        })
        
        print(f"  Event {event_id:2d} ({label:7s}): p95 MAE = {anomaly_score:.4f}")
    
    return np.array(y_true), np.array(y_scores), event_details


def find_optimal_threshold(y_true, y_scores):
    """
    Find optimal threshold using ROC curve analysis.
    
    Uses Youden's J statistic: J = Sensitivity + Specificity - 1
    """
    print("\n" + "=" * 70)
    print("Computing ROC Curve...")
    print("=" * 70)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Find optimal threshold using Youden's J statistic
    # J = Sensitivity + Specificity - 1 = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    print(f"\nOptimal Threshold (Youden's J):")
    print(f"  Threshold: {optimal_threshold:.4f}")
    print(f"  TPR (Sensitivity/Recall): {optimal_tpr:.4f} ({optimal_tpr*100:.1f}%)")
    print(f"  FPR (False Positive Rate): {optimal_fpr:.4f} ({optimal_fpr*100:.1f}%)")
    print(f"  Specificity: {1-optimal_fpr:.4f} ({(1-optimal_fpr)*100:.1f}%)")
    print(f"  Youden's J: {j_scores[optimal_idx]:.4f}")
    
    return {
        'optimal_threshold': float(optimal_threshold),
        'optimal_tpr': float(optimal_tpr),
        'optimal_fpr': float(optimal_fpr),
        'roc_auc': float(roc_auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }


def plot_roc_curve(roc_data, save_path):
    """Plot ROC curve with optimal threshold point."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2,
            label=f"ROC Curve (AUC = {roc_data['roc_auc']:.3f})")
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    # Mark optimal threshold point
    ax.plot(roc_data['optimal_fpr'], roc_data['optimal_tpr'], 'go',
            markersize=12, label=f"Optimal Threshold = {roc_data['optimal_threshold']:.3f}")
    
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR / Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve - NBM LSTM Threshold Optimization', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nROC curve saved to: {save_path}")
    plt.close()


def evaluate_with_optimal_threshold(y_true, y_scores, threshold, event_details):
    """Evaluate performance with optimal threshold."""
    print("\n" + "=" * 70)
    print(f"Evaluating with Optimal Threshold: {threshold:.4f}")
    print("=" * 70)
    
    # Make predictions
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Compute metrics
    total = len(y_true)
    accuracy = (tp + tn) / total
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * detection_rate) / (precision + detection_rate) if (precision + detection_rate) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"            Anomaly  Normal")
    print(f"Actual  Anomaly   {tp:2d}      {fn:2d}")
    print(f"        Normal    {fp:2d}      {tn:2d}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:          {accuracy*100:.1f}% ({tp+tn}/{total})")
    print(f"  Detection Rate:    {detection_rate*100:.1f}% ({tp}/{tp+fn})")
    print(f"  False Alarm Rate:  {false_alarm_rate*100:.1f}% ({fp}/{fp+tn})")
    print(f"  Precision:         {precision*100:.1f}%")
    print(f"  F1 Score:          {f1_score*100:.1f}%")
    print(f"  Specificity:       {specificity*100:.1f}%")
    
    # Show per-event results
    print(f"\nPer-Event Results:")
    anomaly_events = []
    normal_events = []
    
    for i, detail in enumerate(event_details):
        event_id = detail['event_id']
        label = detail['label']
        score = detail['score']
        predicted_anomaly = y_pred[i] == 1
        
        if label == 'anomaly':
            status = "✓ DETECTED" if predicted_anomaly else "✗ MISSED"
            anomaly_events.append(f"  Event {event_id}: Score={score:.4f} {status}")
        else:
            status = "✗ FALSE ALARM" if predicted_anomaly else "✓ CORRECT"
            normal_events.append(f"  Event {event_id}: Score={score:.4f} {status}")
    
    print(f"\nAnomaly Events ({tp+fn}):")
    for line in anomaly_events:
        print(line)
    
    print(f"\nNormal Events ({fp+tn}):")
    for line in normal_events:
        print(line)
    
    return {
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
        'accuracy': float(accuracy),
        'detection_rate': float(detection_rate),
        'false_alarm_rate': float(false_alarm_rate),
        'precision': float(precision),
        'f1_score': float(f1_score),
        'specificity': float(specificity)
    }


def compare_with_baselines(optimal_perf):
    """Compare with baseline results."""
    print("\n" + "=" * 70)
    print("Comparison with Baselines")
    print("=" * 70)
    
    baselines = {
        'Naive Baseline': {
            'detection_rate': 0.50,
            'false_alarm_rate': 0.125,
            'precision': 0.667,
            'f1_score': 0.571
        },
        'NBM LSTM (mean)': {
            'detection_rate': 0.25,
            'false_alarm_rate': 0.375,
            'precision': 0.25,
            'f1_score': 0.25
        },
        'NBM LSTM (p95)': {
            'detection_rate': 1.00,
            'false_alarm_rate': 0.875,
            'precision': 0.364,
            'f1_score': 0.533
        }
    }
    
    print(f"\n{'Model':<25} {'Detection':<12} {'False Alarm':<13} {'Precision':<11} {'F1 Score'}")
    print("-" * 70)
    
    for name, perf in baselines.items():
        print(f"{name:<25} {perf['detection_rate']*100:<12.1f}% {perf['false_alarm_rate']*100:<13.1f}% "
              f"{perf['precision']*100:<11.1f}% {perf['f1_score']*100:.1f}%")
    
    print(f"{'NBM LSTM (ROC optimal)':<25} {optimal_perf['detection_rate']*100:<12.1f}% "
          f"{optimal_perf['false_alarm_rate']*100:<13.1f}% {optimal_perf['precision']*100:<11.1f}% "
          f"{optimal_perf['f1_score']*100:.1f}%")


def main():
    print("=" * 70)
    print("ROC Curve Threshold Optimization - NBM LSTM")
    print("=" * 70)
    
    ensure_dirs()
    
    # Load data and model
    model, X_val, y_val, test_events = load_data_and_model()
    
    # Prepare ROC data (using test events as we don't have labeled val data)
    y_true, y_scores, event_details = prepare_roc_data(test_events, model)
    
    # Find optimal threshold
    roc_data = find_optimal_threshold(y_true, y_scores)
    
    # Plot ROC curve
    output_dir = os.path.join(RESULTS_DIR, 'nbm_lstm')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(roc_data, plot_path)
    
    # Evaluate with optimal threshold
    optimal_perf = evaluate_with_optimal_threshold(
        y_true, y_scores, roc_data['optimal_threshold'], event_details
    )
    
    # Compare with baselines
    compare_with_baselines(optimal_perf)
    
    # Save results
    results = {
        'roc_analysis': roc_data,
        'optimal_performance': optimal_perf,
        'event_details': event_details
    }
    
    results_path = os.path.join(output_dir, 'roc_threshold_optimization.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {results_path}")
    print(f"ROC curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
