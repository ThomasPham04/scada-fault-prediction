"""
Baseline Model: Naive Statistical Threshold with Top 30 Features

This version uses only the top 30 features selected from Pearson correlation analysis.
"""

import os
import sys
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs
from utils.feature_selector import get_top_features, get_feature_indices, filter_features


def load_event_duration_data_top30():
    """Load event duration test data and filter to top 30 features."""
    # Load metadata to get all feature names
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_7day.pkl'))
    all_features = metadata['feature_columns']
    
    # Get top 30 features
    top_features = get_top_features(30)
    feature_indices = get_feature_indices(all_features, top_features)
    
    print(f"\nFiltering from {len(all_features)} to {len(top_features)} features")
    print(f"Feature indices: {len(feature_indices)}")
    
    # Load training data
    X_train = np.load(os.path.join(nbm_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(nbm_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(nbm_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(nbm_dir, 'y_val.npy'))
    
    # Use last timestep and filter to top 30
    X_train_last = y_train[:, feature_indices]
    X_val_last = y_val[:, feature_indices]
    
    print(f"  Train samples: {X_train_last.shape}")
    print(f"  Val samples: {X_val_last.shape}")
    
    # Load test from event duration
    test_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day_event_duration', 'test_by_event')
    test_events = {}
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            y_test = data['y']
            
            # Filter to top 30 features
            y_test_filtered = y_test[:, feature_indices]
            
            test_events[event_id] = {
                'X': y_test_filtered,
                'label': str(data['label'])
            }
    
    print(f"  Test events loaded: {len(test_events)}")
    
    return {
        'X_train': X_train_last,
        'X_val': X_val_last,
        'test_events': test_events,
        'feature_names': top_features,
        'n_features': len(top_features)
    }


class NaiveThresholdDetector:
    """
    Naive anomaly detector using statistical thresholds.
    
    For each feature:
    - Compute mean and std from training data
    - Anomaly score = max z-score across all features
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X_train):
        """Learn mean and std from training data."""
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1e-8, self.std)
        
        print(f"\nNaive Detector trained:")
        print(f"  Mean shape: {self.mean.shape}")
        print(f"  Std shape: {self.std.shape}")
    
    def compute_anomaly_score(self, X):
        """
        Compute anomaly score as max absolute z-score.
        
        z_score = |x - mean| / std
        anomaly_score = max(z_scores across all features)
        """
        z_scores = np.abs((X - self.mean) / self.std)
        anomaly_scores = np.max(z_scores, axis=1)
        return anomaly_scores


def evaluate_naive_baseline(detector, X_train, X_val, test_events):
    """Evaluate naive baseline."""
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Compute scores
    train_scores = detector.compute_anomaly_score(X_train)
    val_scores = detector.compute_anomaly_score(X_val)
    
    # Determine threshold from validation
    threshold = np.percentile(val_scores, 95)
    
    print(f"\nTraining scores:")
    print(f"  Mean: {np.mean(train_scores):.4f}")
    print(f"  Std: {np.std(train_scores):.4f}")
    print(f"  p95: {np.percentile(train_scores, 95):.4f}")
    
    print(f"\nValidation scores:")
    print(f"  Mean: {np.mean(val_scores):.4f}")
    print(f"  Std: {np.std(val_scores):.4f}")
    print(f"  p95 (threshold): {threshold:.4f}")
    
    # Test on events
    print(f"\nTest Events:")
    print(f"{'Event':<8} {'Label':<10} {'Mean Score':<12} {'p95 Score':<12} {'Detected':<10}")
    print("-"*70)
    
    test_results = {}
    for event_id, event_data in sorted(test_events.items()):
        scores = detector.compute_anomaly_score(event_data['X'])
        mean_score = np.mean(scores)
        p95_score = np.percentile(scores, 95)
        detected = p95_score > threshold
        
        test_results[event_id] = {
            'label': event_data['label'],
            'mean_score': float(mean_score),
            'p95_score': float(p95_score),
            'detected': bool(detected)
        }
        
        print(f"{event_id:<8} {event_data['label']:<10} {mean_score:<12.4f} {p95_score:<12.4f} {'Yes' if detected else 'No':<10}")
    
    return {
        'train': {
            'mean': float(np.mean(train_scores)),
            'std': float(np.std(train_scores)),
            'p95': float(np.percentile(train_scores, 95))
        },
        'val': {
            'mean': float(np.mean(val_scores)),
            'std': float(np.std(val_scores)),
            'p95': float(threshold)
        },
        'threshold': float(threshold),
        'test_events': test_results
    }


def compute_metrics(results):
    """Compute confusion matrix and metrics."""
    TP = FP = TN = FN = 0
    
    for event_id, data in results['test_events'].items():
        is_anomaly = (data['label'] == 'anomaly')
        detected = data['detected']
        
        if is_anomaly and detected:
            TP += 1
        elif not is_anomaly and not detected:
            TN += 1
        elif not is_anomaly and detected:
            FP += 1
        else:
            FN += 1
    
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'confusion_matrix': {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN},
        'metrics': {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'false_alarm_rate': far,
            'f1_score': f1
        }
    }


def save_results(detector, results, save_dir):
    """Save naive model and results."""
    # Save model
    model_path = os.path.join(save_dir, 'naive_top30_model.pkl')
    joblib.dump(detector, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Compute metrics
    metrics_data = compute_metrics(results)
    
    # Save results
    results_path = os.path.join(save_dir, 'naive_top30_results.json')
    save_data = {
        'model': 'Naive_Threshold_Top30',
        'n_features': 30,
        'threshold': results['threshold'],
        **metrics_data,
        'train_stats': results['train'],
        'val_stats': results['val'],
        'test_events': results['test_events']
    }
    
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Print metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    cm = metrics_data['confusion_matrix']
    m = metrics_data['metrics']
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted Anomaly | Predicted Normal")
    print(f"  Actual Anomaly  |       {cm['TP']:3d}        |      {cm['FN']:3d}")
    print(f"  Actual Normal   |       {cm['FP']:3d}        |      {cm['TN']:3d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:             {m['accuracy']:.2%}")
    print(f"  Detection Rate:       {m['recall']:.2%}")
    print(f"  False Alarm Rate:     {m['false_alarm_rate']:.2%}")
    print(f"  Precision:            {m['precision']:.2%}")
    print(f"  F1 Score:             {m['f1_score']:.4f}")


def main():
    """Main naive baseline pipeline with top 30 features."""
    print("="*70)
    print("Naive Statistical Threshold Baseline - Top 30 Features")
    print("="*70)
    
    ensure_dirs()
    results_dir = os.path.join(RESULTS_DIR, 'baselines')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data with top 30 features
    data = load_event_duration_data_top30()
    
    print(f"\nUsing {data['n_features']} features:")
    for i, feat in enumerate(data['feature_names'][:10], 1):
        print(f"  {i}. {feat}")
    print(f"  ... and {data['n_features'] - 10} more")
    
    # Train
    detector = NaiveThresholdDetector()
    detector.fit(data['X_train'])
    
    # Evaluate
    results = evaluate_naive_baseline(
        detector,
        data['X_train'],
        data['X_val'],
        data['test_events']
    )
    
    # Save
    save_results(detector, results, results_dir)
    
    print("\n" + "="*70)
    print("Naive Baseline Top 30 Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
