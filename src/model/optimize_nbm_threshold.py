"""
Optimize Threshold for NBM LSTM Model

Scan multiple threshold values to find the optimal balance between
detection rate (recall) and false alarm rate.
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR


def load_nbm_data():
    """Load NBM V2 test data."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    test_events = {}
    
    print("Loading NBM V2 test data...")
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            
            test_events[event_id] = {
                'X': data['X'],
                'y': data['y'],
                'label': str(data['label'])
            }
    
    print(f"  Loaded {len(test_events)} test events")
    return test_events


def compute_event_scores(model, test_events):
    """
    Compute anomaly scores (p95 MAE) for all test events.
    
    Returns:
        scores: List of (event_id, p95_mae, is_anomaly)
    """
    print("\nComputing anomaly scores for all events...")
    scores = []
    
    for event_id, event_data in sorted(test_events.items()):
        X = event_data['X']
        y = event_data['y']
        
        # Predict and compute MAE
        y_pred = model.predict(X, verbose=0)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        
        # Event score: p95 MAE
        p95_mae = np.percentile(mae, 95)
        is_anomaly = (event_data['label'] == 'anomaly')
        
        scores.append({
            'event_id': event_id,
            'p95_mae': float(p95_mae),
            'is_anomaly': is_anomaly
        })
        
        print(f"  Event {event_id}: p95_mae={p95_mae:.4f}, label={event_data['label']}")
    
    return scores


def evaluate_at_threshold(scores, threshold):
    """
    Evaluate performance at a specific threshold.
    
    Returns:
        metrics dict with TP, FP, TN, FN, and derived metrics
    """
    TP = FP = TN = FN = 0
    
    for score in scores:
        predicted_anomaly = score['p95_mae'] > threshold
        actual_anomaly = score['is_anomaly']
        
        if actual_anomaly and predicted_anomaly:
            TP += 1
        elif actual_anomaly and not predicted_anomaly:
            FN += 1
        elif not actual_anomaly and predicted_anomaly:
            FP += 1
        else:
            TN += 1
    
    # Calculate metrics
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    tpr = recall  # True Positive Rate = Recall
    
    return {
        'threshold': threshold,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'fpr': fpr,
        'tpr': tpr
    }


def scan_thresholds(scores, threshold_range):
    """
    Scan multiple thresholds and compute metrics for each.
    """
    print(f"\nScanning thresholds from {threshold_range[0]:.2f} to {threshold_range[-1]:.2f}...")
    
    results = []
    for threshold in threshold_range:
        metrics = evaluate_at_threshold(scores, threshold)
        results.append(metrics)
    
    return results


def find_optimal_threshold(results):
    """
    Find optimal threshold based on different criteria.
    """
    # Strategy 1: Maximize F1 Score
    best_f1 = max(results, key=lambda x: x['f1_score'])
    
    # Strategy 2: Balance Recall >= 0.9 with highest Precision
    high_recall = [r for r in results if r['recall'] >= 0.9]
    best_balanced = max(high_recall, key=lambda x: x['precision']) if high_recall else best_f1
    
    # Strategy 3: Maximize (Recall - FPR)
    for r in results:
        r['youden_index'] = r['tpr'] - r['fpr']  # Youden's J statistic
    best_youden = max(results, key=lambda x: x['youden_index'])
    
    return {
        'best_f1': best_f1,
        'best_balanced': best_balanced,
        'best_youden': best_youden
    }


def plot_results(results, optimal_thresholds, save_dir):
    """
    Plot ROC curve, Precision-Recall curve, and metrics vs threshold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    thresholds = [r['threshold'] for r in results]
    recalls = [r['recall'] for r in results]
    precisions = [r['precision'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    fprs = [r['fpr'] for r in results]
    tprs = [r['tpr'] for r in results]
    
    # 1. ROC Curve
    axes[0, 0].plot(fprs, tprs, 'b-', linewidth=2, label='ROC Curve')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    
    # Mark optimal points
    axes[0, 0].scatter(optimal_thresholds['best_f1']['fpr'], 
                      optimal_thresholds['best_f1']['tpr'],
                      color='red', s=100, zorder=5, label='Best F1')
    axes[0, 0].scatter(optimal_thresholds['best_youden']['fpr'], 
                      optimal_thresholds['best_youden']['tpr'],
                      color='green', s=100, zorder=5, label='Best Youden')
    
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[0, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    axes[0, 1].plot(recalls, precisions, 'b-', linewidth=2)
    axes[0, 1].scatter(optimal_thresholds['best_f1']['recall'],
                      optimal_thresholds['best_f1']['precision'],
                      color='red', s=100, zorder=5, label='Best F1')
    axes[0, 1].scatter(optimal_thresholds['best_balanced']['recall'],
                      optimal_thresholds['best_balanced']['precision'],
                      color='orange', s=100, zorder=5, label='Best Balanced')
    
    axes[0, 1].set_xlabel('Recall (Detection Rate)', fontsize=12)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Metrics vs Threshold
    axes[1, 0].plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    axes[1, 0].plot(thresholds, precisions, 'g-', label='Precision', linewidth=2)
    axes[1, 0].plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2)
    
    # Mark optimal thresholds
    axes[1, 0].axvline(optimal_thresholds['best_f1']['threshold'], 
                       color='red', linestyle='--', alpha=0.5, label='Best F1 Threshold')
    axes[1, 0].axvline(optimal_thresholds['best_balanced']['threshold'],
                       color='orange', linestyle='--', alpha=0.5, label='Best Balanced Threshold')
    
    axes[1, 0].set_xlabel('Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. False Alarm Rate vs Threshold
    axes[1, 1].plot(thresholds, fprs, 'r-', linewidth=2, label='False Alarm Rate')
    axes[1, 1].plot(thresholds, recalls, 'b-', linewidth=2, label='Detection Rate')
    
    axes[1, 1].axvline(optimal_thresholds['best_f1']['threshold'],
                       color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(optimal_thresholds['best_youden']['threshold'],
                       color='green', linestyle='--', alpha=0.5)
    
    axes[1, 1].set_xlabel('Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Rate', fontsize=12)
    axes[1, 1].set_title('Detection Rate vs False Alarm Rate', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'threshold_optimization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_path}")
    plt.close()


def print_recommendations(optimal_thresholds):
    """Print threshold recommendations."""
    print("\n" + "=" * 70)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. BEST F1 SCORE (Balanced Overall Performance)")
    best_f1 = optimal_thresholds['best_f1']
    print(f"   Threshold: {best_f1['threshold']:.4f}")
    print(f"   - Recall (Detection Rate): {best_f1['recall']:.2%}")
    print(f"   - Precision: {best_f1['precision']:.2%}")
    print(f"   - F1 Score: {best_f1['f1_score']:.4f}")
    print(f"   - False Alarm Rate: {best_f1['fpr']:.2%}")
    print(f"   - Confusion Matrix: TP={best_f1['TP']}, FP={best_f1['FP']}, TN={best_f1['TN']}, FN={best_f1['FN']}")
    
    print("\n2. BEST BALANCED (High Recall ≥90% + Best Precision)")
    best_bal = optimal_thresholds['best_balanced']
    print(f"   Threshold: {best_bal['threshold']:.4f}")
    print(f"   - Recall (Detection Rate): {best_bal['recall']:.2%}")
    print(f"   - Precision: {best_bal['precision']:.2%}")
    print(f"   - F1 Score: {best_bal['f1_score']:.4f}")
    print(f"   - False Alarm Rate: {best_bal['fpr']:.2%}")
    print(f"   - Confusion Matrix: TP={best_bal['TP']}, FP={best_bal['FP']}, TN={best_bal['TN']}, FN={best_bal['FN']}")
    
    print("\n3. BEST YOUDEN INDEX (Maximize TPR - FPR)")
    best_you = optimal_thresholds['best_youden']
    print(f"   Threshold: {best_you['threshold']:.4f}")
    print(f"   - Recall (Detection Rate): {best_you['recall']:.2%}")
    print(f"   - Precision: {best_you['precision']:.2%}")
    print(f"   - F1 Score: {best_you['f1_score']:.4f}")
    print(f"   - False Alarm Rate: {best_you['fpr']:.2%}")
    print(f"   - Youden Index: {best_you['youden_index']:.4f}")
    print(f"   - Confusion Matrix: TP={best_you['TP']}, FP={best_you['FP']}, TN={best_you['TN']}, FN={best_you['FN']}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("Choose 'Best F1 Score' threshold for balanced performance.")
    print("=" * 70)


def main():
    """Main threshold optimization pipeline."""
    print("=" * 70)
    print("NBM LSTM THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_v2.keras')
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("  Model loaded successfully!")
    
    # Load test data
    test_events = load_nbm_data()
    
    # Compute scores for all events
    scores = compute_event_scores(model, test_events)
    
    # Define threshold range to scan
    threshold_range = np.arange(0.3, 1.6, 0.05)
    
    # Scan thresholds
    results = scan_thresholds(scores, threshold_range)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_threshold(results)
    
    # Print recommendations
    print_recommendations(optimal_thresholds)
    
    # Save results
    output_dir = os.path.join(RESULTS_DIR, 'nbm_lstm_v2')
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        'threshold_range': threshold_range.tolist(),
        'all_results': results,
        'optimal_thresholds': {
            'best_f1': optimal_thresholds['best_f1'],
            'best_balanced': optimal_thresholds['best_balanced'],
            'best_youden': optimal_thresholds['best_youden']
        },
        'event_scores': scores
    }
    
    output_path = os.path.join(output_dir, 'threshold_optimization.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Plot results
    plot_results(results, optimal_thresholds, output_dir)


if __name__ == "__main__":
    main()
