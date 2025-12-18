"""
Baseline Model 2: Naive Statistical Threshold

Approach:
- Compute simple statistics (mean, std) from normal training data
- Use z-score thresholds for anomaly detection
- Extremely simple, interpretable baseline
"""

import os
import sys
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs


def load_nbm_v2_data():
    """Load NBM V2 preprocessed data."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    
    print("Loading NBM V2 data for naive baseline...")
    X_train = np.load(os.path.join(nbm_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(nbm_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(nbm_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(nbm_dir, 'y_val.npy'))
    
    # OPTIMIZATION: Use last timestep only
    X_train_last = y_train
    X_val_last = y_val
    
    print(f"  Train samples: {X_train_last.shape} (using last timestep only)")
    print(f"  Val samples: {X_val_last.shape}")
    
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_v2.pkl'))
    
    # Load test
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    test_events = {}
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            y_test = data['y']
            
            test_events[event_id] = {
                'X': y_test,
                'label': str(data['label'])
            }
    
    print(f"  Test events loaded: {len(test_events)}")
    
    return {
        'X_train': X_train_last,
        'X_val': X_val_last,
        'test_events': test_events,
        'metadata': metadata,
        'n_features': X_train_last.shape[1]
    }


class NaiveThresholdDetector:
    """
    Naive anomaly detector using statistical thresholds.
    
    For each feature:
    - Compute mean and std from training data
    - Anomaly score = max z-score across all features
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X_train):
        """Learn mean and std from training data."""
        print("\nTraining Naive Threshold baseline...")
        
        self.mean_ = np.mean(X_train, axis=0)
        self.std_ = np.std(X_train, axis=0)
        
        # Avoid division by zero
        self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)
        
        print(f"  Learned statistics for {len(self.mean_)} features")
        
        return self
    
    def compute_anomaly_score(self, X):
        """
        Compute anomaly score as max absolute z-score.
        
        z_score = |x - mean| / std
        anomaly_score = max(z_scores across all features)
        """
        z_scores = np.abs((X - self.mean_) / self.std_)
        
        # Max z-score per sample
        anomaly_scores = np.max(z_scores, axis=1)
        
        return anomaly_scores


def evaluate_naive_baseline(detector, X_train, X_val, test_events):
    """Evaluate naive baseline."""
    print("\nEvaluating Naive baseline...")
    
    results = {}
    
    # Compute scores
    print("  Computing anomaly scores...")
    train_scores = detector.compute_anomaly_score(X_train)
    val_scores = detector.compute_anomaly_score(X_val)
    
    results['train'] = {
        'scores': train_scores,
        'mean': float(np.mean(train_scores)),
        'std': float(np.std(train_scores)),
        'p95': float(np.percentile(train_scores, 95)),
        'p99': float(np.percentile(train_scores, 99))
    }
    
    results['val'] = {
        'scores': val_scores,
        'mean': float(np.mean(val_scores)),
        'std': float(np.std(val_scores)),
        'p95': float(np.percentile(val_scores, 95)),
        'p99': float(np.percentile(val_scores, 99))
    }
    
    print(f"    Train score: {results['train']['mean']:.4f} ± {results['train']['std']:.4f}")
    print(f"    Val score: {results['val']['mean']:.4f} ± {results['val']['std']:.4f}")
    
    # Test events
    results['test_events'] = {}
    for event_id, data in test_events.items():
        scores = detector.compute_anomaly_score(data['X'])
        results['test_events'][event_id] = {
            'label': data['label'],
            'scores': scores,
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'p95': float(np.percentile(scores, 95)),
            'max': float(np.max(scores))
        }
    
    return results


def plot_naive_results(results, save_dir):
    """Plot naive baseline results."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Score distribution
    axes[0, 0].hist(results['train']['scores'], bins=100, alpha=0.6, label='Train', density=True)
    axes[0, 0].hist(results['val']['scores'], bins=100, alpha=0.6, label='Val', density=True)
    axes[0, 0].axvline(results['val']['p95'], color='r', linestyle='--', label='95th percentile')
    axes[0, 0].set_xlabel('Anomaly Score (Max Z-Score)', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Naive Baseline Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test events comparison
    event_ids = []
    event_means = []
    colors = []
    
    for event_id, data in sorted(results['test_events'].items()):
        event_ids.append(event_id)
        event_means.append(data['mean'])
        colors.append('red' if data['label'] == 'anomaly' else 'green')
    
    axes[0, 1].bar(range(len(event_ids)), event_means, color=colors, alpha=0.7)
    axes[0, 1].axhline(results['val']['p95'], color='orange', linestyle='--',
                       linewidth=2, label='Val 95th percentile')
    axes[0, 1].set_xlabel('Event Index', fontsize=12)
    axes[0, 1].set_ylabel('Mean Anomaly Score', fontsize=12)
    axes[0, 1].set_title('Naive Baseline Score by Test Event', fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Anomaly'),
        Patch(facecolor='green', alpha=0.7, label='Normal'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Threshold (95%)')
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper left')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Violin plot per event
    for i, (event_id, data) in enumerate(sorted(results['test_events'].items())):
        scores = data['scores']
        axes[1, 0].violinplot([scores], positions=[i], widths=0.7,
                              showmeans=True, showmedians=True)
    
    axes[1, 0].axhline(results['val']['p95'], color='orange', linestyle='--',
                       linewidth=2, label='Val 95th percentile')
    axes[1, 0].set_xlabel('Event Index', fontsize=12)
    axes[1, 0].set_ylabel('Anomaly Score', fontsize=12)
    axes[1, 0].set_title('Score Distribution per Event (Violin Plot)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Normal vs Anomaly
    anomaly_scores = []
    normal_scores = []
    
    for event_id, data in results['test_events'].items():
        if data['label'] == 'anomaly':
            anomaly_scores.extend(data['scores'])
        else:
            normal_scores.extend(data['scores'])
    
    axes[1, 1].hist(normal_scores, bins=100, alpha=0.6, label='Normal Events',
                    color='green', density=True)
    axes[1, 1].hist(anomaly_scores, bins=100, alpha=0.6, label='Anomaly Events',
                    color='red', density=True)
    axes[1, 1].axvline(results['val']['p95'], color='orange', linestyle='--',
                       linewidth=2, label='95th threshold')
    axes[1, 1].set_xlabel('Anomaly Score', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Normal vs Anomaly Score Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'naive_baseline_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_path}")
    plt.close()


def save_naive_results(detector, results, save_dir):
    """Save naive model and results."""
    # Save model
    model_path = os.path.join(MODELS_DIR, 'naive_baseline.pkl')
    joblib.dump(detector, model_path)
    print(f"\nNaive detector saved to: {model_path}")
    
    # Save results
    results_summary = {
        'model_type': 'Naive_Threshold_Baseline',
        'method': 'Max_Z_Score',
        'train': {k: v for k, v in results['train'].items() if k != 'scores'},
        'val': {k: v for k, v in results['val'].items() if k != 'scores'},
        'test_events': {
            event_id: {k: v for k, v in data.items() if k != 'scores'}
            for event_id, data in results['test_events'].items()
        }
    }
    
    results_path = os.path.join(save_dir, 'naive_baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Save scores
    scores_path = os.path.join(save_dir, 'naive_baseline_scores.npz')
    np.savez(
        scores_path,
        train_scores=results['train']['scores'],
        val_scores=results['val']['scores'],
        **{f'event_{eid}_scores': data['scores']
           for eid, data in results['test_events'].items()}
    )
    print(f"Score arrays saved to: {scores_path}")


def main():
    """Main naive baseline pipeline."""
    print("=" * 70)
    print("Naive Statistical Threshold Baseline")
    print("=" * 70)
    
    ensure_dirs()
    results_dir = os.path.join(RESULTS_DIR, 'baselines')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data = load_nbm_v2_data()
    
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
    
    # Plot
    plot_naive_results(results, results_dir)
    
    # Save
    save_naive_results(detector, results, results_dir)
    
    print("\n" + "=" * 70)
    print("Naive Baseline Complete!")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  Train score: {results['train']['mean']:.4f}")
    print(f"  Val 95th percentile: {results['val']['p95']:.4f}")
    print(f"\nAnomaly detection:")
    anomaly_count = sum(1 for d in results['test_events'].values() if d['label'] == 'anomaly')
    normal_count = sum(1 for d in results['test_events'].values() if d['label'] == 'normal')
    print(f"  Anomaly events: {anomaly_count}")
    print(f"  Normal events: {normal_count}")


if __name__ == "__main__":
    main()
