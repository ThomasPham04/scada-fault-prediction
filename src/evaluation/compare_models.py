"""
Ground Truth Setup and Model Comparison Framework

Purpose:
- Create ground truth labels from event_info
- Load predictions from all models (NBM LSTM, PCA, Naive)
- Compute comparison metrics (Precision, Recall, F1, ROC-AUC, Detection Rate)
- Generate comparative visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_DIR, WIND_FARM_A_PROCESSED, RESULTS_DIR, ensure_dirs


def create_ground_truth():
    """
    Create ground truth labels from test events.
    
    Returns:
        Dictionary mapping event_id to ground truth (0=normal, 1=anomaly)
    """
    print("Creating Ground Truth from event labels...")
    
    # Load event info
    event_info_path = os.path.join(WIND_FARM_A_DIR, "event_info.csv")
    event_info = pd.read_csv(event_info_path, sep=';')
    
    # Create ground truth mapping
    ground_truth = {}
    for _, event in event_info.iterrows():
        event_id = event['event_id']
        label = 1 if event['event_label'] == 'anomaly' else 0
        ground_truth[event_id] = {
            'label': label,
            'event_label': event['event_label'],
            'event_start': event['event_start'],
            'event_end': event['event_end']
        }
    
    print(f"  Total events: {len(ground_truth)}")
    print(f"  Anomaly events: {sum(1 for v in ground_truth.values() if v['label'] == 1)}")
    print(f"  Normal events: {sum(1 for v in ground_truth.values() if v['label'] == 0)}")
    
    return ground_truth


def load_test_events():
    """Load test event IDs from NBM V2."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    
    test_event_ids = []
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            test_event_ids.append(event_id)
    
    print(f"\nTest events available: {len(test_event_ids)}")
    return sorted(test_event_ids)


def load_model_predictions(model_name, test_event_ids):
    """
    Load predictions/scores from a model.
    
    Args:
        model_name: 'pca', 'naive', or 'nbm_lstm'
        test_event_ids: List of test event IDs
        
    Returns:
        Dictionary mapping event_id to anomaly scores
    """
    print(f"\nLoading {model_name.upper()} predictions...")
    
    results_dir = os.path.join(RESULTS_DIR, 'baselines')
    predictions = {}
    
    if model_name == 'pca':
        # Load PCA errors
        errors_file = os.path.join(results_dir, 'pca_baseline_errors.npz')
        errors_data = np.load(errors_file)
        
        for event_id in test_event_ids:
            key = f'event_{event_id}_errors'
            if key in errors_data:
                predictions[event_id] = errors_data[key]
        
        # Load threshold
        results_file = os.path.join(results_dir, 'pca_baseline_results.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        threshold = results['val']['p95']
        
    elif model_name == 'naive':
        # Load Naive scores
        scores_file = os.path.join(results_dir, 'naive_baseline_scores.npz')
        scores_data = np.load(scores_file)
        
        for event_id in test_event_ids:
            key = f'event_{event_id}_scores'
            if key in scores_data:
                predictions[event_id] = scores_data[key]
        
        # Load threshold
        results_file = os.path.join(results_dir, 'naive_baseline_results.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        threshold = results['val']['p95']
        
    elif model_name == 'nbm_lstm':
        # Load NBM LSTM errors (if available)
        nbm_results_dir = os.path.join(RESULTS_DIR, 'NBM_v2')
        errors_file = os.path.join(nbm_results_dir, 'nbm_v2_errors.npz')
        
        if os.path.exists(errors_file):
            errors_data = np.load(errors_file)
            for event_id in test_event_ids:
                key = f'event_{event_id}_errors'
                if key in errors_data:
                    predictions[event_id] = errors_data[key]
            
            # Load threshold
            results_file = os.path.join(nbm_results_dir, 'nbm_v2_results.json')
            with open(results_file, 'r') as f:
                results = json.load(f)
            threshold = results['val_error_stats']['mean_mse'] + 2 * results['val_error_stats']['std_mse']
        else:
            print(f"  WARNING: NBM LSTM results not found. Skipping.")
            return None, None
    
    print(f"  Loaded predictions for {len(predictions)} events")
    print(f"  Threshold: {threshold:.6f}")
    
    return predictions, threshold


def compute_event_level_predictions(predictions, threshold):
    """
    Convert per-sample scores to event-level predictions.
    
    If ANY sample in an event exceeds threshold → predict anomaly (1)
    Otherwise → predict normal (0)
    """
    event_predictions = {}
    
    for event_id, scores in predictions.items():
        # Event is anomaly if max score > threshold
        max_score = np.max(scores)
        event_predictions[event_id] = 1 if max_score > threshold else 0
    
    return event_predictions


def compute_metrics(ground_truth, event_predictions, test_event_ids, model_name):
    """
    Compute evaluation metrics.
    
    Metrics:
    - Precision: TP / (TP + FP)
    - Recall (Detection Rate): TP / (TP + FN)
    - F1 Score
    - False Alarm Rate: FP / (FP + TN)
    - Accuracy
    """
    # Get labels and predictions for test events
    y_true = []
    y_pred = []
    
    for event_id in test_event_ids:
        if event_id in ground_truth and event_id in event_predictions:
            y_true.append(ground_truth[event_id]['label'])
            y_pred.append(event_predictions[event_id])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        'model': model_name,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'detection_rate': float(recall),  # Same as recall
        'false_alarm_rate': float(false_alarm_rate),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_test_events': len(y_true),
        'total_anomalies': int(np.sum(y_true)),
        'total_normal': int(len(y_true) - np.sum(y_true))
    }
    
    return metrics


def print_metrics(metrics):
    """Pretty print metrics."""
    print(f"\n{'='*70}")
    print(f"Model: {metrics['model'].upper()}")
    print(f"{'='*70}")
    print(f"\nPerformance Metrics:")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1 Score:           {metrics['f1_score']:.4f}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Detection Rate:     {metrics['detection_rate']:.4f}")
    print(f"  False Alarm Rate:   {metrics['false_alarm_rate']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:     {metrics['true_positives']}")
    print(f"  False Positives:    {metrics['false_positives']}")
    print(f"  True Negatives:     {metrics['true_negatives']}")
    print(f"  False Negatives:    {metrics['false_negatives']}")
    
    print(f"\nTest Set:")
    print(f"  Total events:       {metrics['total_test_events']}")
    print(f"  Anomaly events:     {metrics['total_anomalies']}")
    print(f"  Normal events:      {metrics['total_normal']}")


def compare_models(all_metrics):
    """
    Create comparison table and visualization.
    """
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    # Create comparison table
    df = pd.DataFrame(all_metrics)
    df = df.set_index('model')
    
    print(df[['precision', 'recall', 'f1_score', 'accuracy', 'false_alarm_rate']].to_string())
    
    return df


def plot_comparison(all_metrics, save_dir):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = [m['model'] for m in all_metrics]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 1. Precision, Recall, F1
    metrics_to_plot = ['precision', 'recall', 'f1_score']
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] for m in all_metrics]
        axes[0, 0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title(),
                       alpha=0.8)
    
    axes[0, 0].set_xlabel('Model', fontsize=12)
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Precision, Recall, F1 Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels([m.upper() for m in models])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim(0, 1.1)
    
    # 2. Detection Rate vs False Alarm Rate
    detection_rates = [m['detection_rate'] for m in all_metrics]
    false_alarm_rates = [m['false_alarm_rate'] for m in all_metrics]
    
    for i, model in enumerate(models):
        axes[0, 1].scatter(false_alarm_rates[i], detection_rates[i], 
                          s=300, alpha=0.7, color=colors[i], 
                          label=model.upper(), edgecolors='black', linewidth=2)
    
    axes[0, 1].set_xlabel('False Alarm Rate', fontsize=12)
    axes[0, 1].set_ylabel('Detection Rate', fontsize=12)
    axes[0, 1].set_title('Detection Rate vs False Alarm Rate', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-0.05, max(false_alarm_rates) + 0.1)
    axes[0, 1].set_ylim(-0.05, 1.05)
    
    # Add ideal point
    axes[0, 1].scatter([0], [1], s=100, color='green', marker='*', 
                       label='Ideal', edgecolors='black', linewidth=2, zorder=10)
    
    # 3. Confusion Matrix Comparison
    categories = ['TP', 'FP', 'TN', 'FN']
    x = np.arange(len(categories))
    width = 0.25
    
    for i, model_metrics in enumerate(all_metrics):
        values = [
            model_metrics['true_positives'],
            model_metrics['false_positives'],
            model_metrics['true_negatives'],
            model_metrics['false_negatives']
        ]
        axes[1, 0].bar(x + i*width, values, width, 
                       label=model_metrics['model'].upper(),
                       alpha=0.8, color=colors[i])
    
    axes[1, 0].set_xlabel('Category', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Overall Scores Radar/Spider Chart would be good but let's do bar chart
    overall_metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    x = np.arange(len(overall_metrics))
    width = 0.25
    
    for i, model_metrics in enumerate(all_metrics):
        values = [model_metrics[m] for m in overall_metrics]
        axes[1, 1].bar(x + i*width, values, width,
                       label=model_metrics['model'].upper(),
                       alpha=0.8, color=colors[i])
    
    axes[1, 1].set_xlabel('Metric', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels([m.replace('_', '\n').title() for m in overall_metrics])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    plt.close()


def save_comparison_results(all_metrics, save_dir):
    """Save comparison results."""
    results_file = os.path.join(save_dir, 'model_comparison.json')
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Comparison results saved to: {results_file}")


def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("Ground Truth Setup & Model Comparison")
    print("=" * 70)
    
    ensure_dirs()
    comparison_dir = os.path.join(RESULTS_DIR, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create ground truth
    ground_truth = create_ground_truth()
    
    # Get test events
    test_event_ids = load_test_events()
    
    # Evaluate each model
    all_metrics = []
    
    for model_name in ['pca', 'naive']:  # Add 'nbm_lstm' when available
        predictions, threshold = load_model_predictions(model_name, test_event_ids)
        
        if predictions is None:
            continue
        
        # Convert to event-level predictions
        event_predictions = compute_event_level_predictions(predictions, threshold)
        
        # Compute metrics
        metrics = compute_metrics(ground_truth, event_predictions, test_event_ids, model_name)
        
        # Print
        print_metrics(metrics)
        
        all_metrics.append(metrics)
    
    # Compare models
    if len(all_metrics) > 0:
        df_comparison = compare_models(all_metrics)
        plot_comparison(all_metrics, comparison_dir)
        save_comparison_results(all_metrics, comparison_dir)
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
