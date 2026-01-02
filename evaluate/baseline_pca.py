"""
Baseline Model 1: PCA-based Anomaly Detection

Approach:
- Train PCA on normal operation data
- Use reconstruction error as anomaly score
- Simple, classical unsupervised approach
"""

import os
import sys
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs, RANDOM_SEED


def load_nbm_v2_data():
    """Load NBM V2 preprocessed data."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_v2')
    
    print("Loading NBM V2 data for baseline...")
    X_train = np.load(os.path.join(nbm_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(nbm_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(nbm_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(nbm_dir, 'y_val.npy'))
    
    # OPTIMIZATION: Use last timestep only (prediction target)
    # This reduces samples from 15M to 7.8K while still being representative
    # Shape: (n_sequences, n_features) instead of (n_sequences * window_size, n_features)
    X_train_last = y_train  # Target is last timestep
    X_val_last = y_val
    
    print(f"  Train samples: {X_train_last.shape} (using last timestep only)")
    print(f"  Val samples: {X_val_last.shape}")
    
    metadata = joblib.load(os.path.join(nbm_dir, 'nbm_metadata_v2.pkl'))
    
    # Load test data
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    test_events = {}
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            y_test = data['y']  # Use prediction targets (last timesteps)
            
            test_events[event_id] = {
                'X': y_test,
                'label': str(data['label'])
            }
    
    print(f"  Test events loaded: {len(test_events)}")
    
    return {
        'X_train': X_train_last,
        'X_val': X_val_last,
        'test_events': test_events,
        'metadata': metadata
    }


def train_pca_baseline(X_train, n_components=0.95):
    """
    Train PCA model on normal data.
    
    Args:
        X_train: Training data (normal only)
        n_components: Number of components or variance ratio to keep
    
    Returns:
        Fitted PCA model
    """
    print(f"\nTraining PCA baseline...")
    print(f"  Variance to preserve: {n_components}")
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(X_train)
    
    print(f"  Components selected: {pca.n_components_}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return pca


def compute_reconstruction_error(pca, X):
    """
    Compute reconstruction error for PCA.
    
    Error = ||X - X_reconstructed||^2
    """
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # MSE per sample
    errors = np.mean((X - X_reconstructed) ** 2, axis=1)
    
    return errors


def evaluate_pca_baseline(pca, X_train, X_val, test_events):
    """Evaluate PCA baseline on all datasets."""
    print("\nEvaluating PCA baseline...")
    
    results = {}
    
    # Compute errors
    print("  Computing reconstruction errors...")
    train_errors = compute_reconstruction_error(pca, X_train)
    val_errors = compute_reconstruction_error(pca, X_val)
    
    results['train'] = {
        'errors': train_errors,
        'mean': float(np.mean(train_errors)),
        'std': float(np.std(train_errors)),
        'p95': float(np.percentile(train_errors, 95)),
        'p99': float(np.percentile(train_errors, 99))
    }
    
    results['val'] = {
        'errors': val_errors,
        'mean': float(np.mean(val_errors)),
        'std': float(np.std(val_errors)),
        'p95': float(np.percentile(val_errors, 95)),
        'p99': float(np.percentile(val_errors, 99))
    }
    
    print(f"    Train error: {results['train']['mean']:.6f} ± {results['train']['std']:.6f}")
    print(f"    Val error: {results['val']['mean']:.6f} ± {results['val']['std']:.6f}")
    
    # Test events
    results['test_events'] = {}
    for event_id, data in test_events.items():
        errors = compute_reconstruction_error(pca, data['X'])
        results['test_events'][event_id] = {
            'label': data['label'],
            'errors': errors,
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'p95': float(np.percentile(errors, 95)),
            'max': float(np.max(errors))
        }
    
    return results


def plot_pca_results(results, save_dir):
    """Plot PCA baseline results."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Error distribution (train/val)
    axes[0, 0].hist(results['train']['errors'], bins=100, alpha=0.6, label='Train', density=True)
    axes[0, 0].hist(results['val']['errors'], bins=100, alpha=0.6, label='Val', density=True)
    axes[0, 0].axvline(results['val']['p95'], color='r', linestyle='--', label='95th percentile')
    axes[0, 0].set_xlabel('Reconstruction Error', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('PCA Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test events comparison
    event_ids = []
    event_means = []
    event_labels = []
    colors = []
    
    for event_id, data in sorted(results['test_events'].items()):
        event_ids.append(event_id)
        event_means.append(data['mean'])
        event_labels.append(data['label'])
        colors.append('red' if data['label'] == 'anomaly' else 'green')
    
    bars = axes[0, 1].bar(range(len(event_ids)), event_means, color=colors, alpha=0.7)
    axes[0, 1].axhline(results['val']['p95'], color='orange', linestyle='--', 
                       linewidth=2, label='Val 95th percentile')
    axes[0, 1].set_xlabel('Event Index', fontsize=12)
    axes[0, 1].set_ylabel('Mean Reconstruction Error', fontsize=12)
    axes[0, 1].set_title('PCA Error by Test Event', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Anomaly'),
                      Patch(facecolor='green', alpha=0.7, label='Normal')]
    axes[0, 1].legend(handles=legend_elements, loc='upper left')
    
    # 3. Error range plot per event
    positions = range(len(event_ids))
    for i, (event_id, data) in enumerate(sorted(results['test_events'].items())):
        errors = data['errors']
        axes[1, 0].violinplot([errors], positions=[i], widths=0.7,
                              showmeans=True, showmedians=True)
    
    axes[1, 0].axhline(results['val']['p95'], color='orange', linestyle='--', 
                       linewidth=2, label='Val 95th percentile')
    axes[1, 0].set_xlabel('Event Index', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction Error', fontsize=12)
    axes[1, 0].set_title('Error Distribution per Event (Violin Plot)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Anomaly vs Normal comparison
    anomaly_errors = []
    normal_errors = []
    
    for event_id, data in results['test_events'].items():
        if data['label'] == 'anomaly':
            anomaly_errors.extend(data['errors'])
        else:
            normal_errors.extend(data['errors'])
    
    axes[1, 1].hist(normal_errors, bins=100, alpha=0.6, label='Normal Events', 
                    color='green', density=True)
    axes[1, 1].hist(anomaly_errors, bins=100, alpha=0.6, label='Anomaly Events', 
                    color='red', density=True)
    axes[1, 1].axvline(results['val']['p95'], color='orange', linestyle='--', 
                       linewidth=2, label='95th threshold')
    axes[1, 1].set_xlabel('Reconstruction Error', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Normal vs Anomaly Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'pca_baseline_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_path}")
    plt.close()


def save_pca_results(pca, results, save_dir):
    """Save PCA model and results."""
    # Save PCA model
    model_path = os.path.join(MODELS_DIR, 'pca_baseline.pkl')
    joblib.dump(pca, model_path)
    print(f"\nPCA model saved to: {model_path}")
    
    # Save results (without raw errors to save space)
    results_summary = {
        'model_type': 'PCA_Baseline',
        'n_components': int(pca.n_components_),
        'explained_variance': float(pca.explained_variance_ratio_.sum()),
        'train': {k: v for k, v in results['train'].items() if k != 'errors'},
        'val': {k: v for k, v in results['val'].items() if k != 'errors'},
        'test_events': {
            event_id: {k: v for k, v in data.items() if k != 'errors'}
            for event_id, data in results['test_events'].items()
        }
    }
    
    results_path = os.path.join(save_dir, 'pca_baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Save error arrays
    errors_path = os.path.join(save_dir, 'pca_baseline_errors.npz')
    np.savez(
        errors_path,
        train_errors=results['train']['errors'],
        val_errors=results['val']['errors'],
        **{f'event_{eid}_errors': data['errors'] 
           for eid, data in results['test_events'].items()}
    )
    print(f"Error arrays saved to: {errors_path}")


def main():
    """Main PCA baseline pipeline."""
    print("=" * 70)
    print("PCA Baseline for Anomaly Detection")
    print("=" * 70)
    
    ensure_dirs()
    results_dir = os.path.join(RESULTS_DIR, 'baselines')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data = load_nbm_v2_data()
    
    # Train PCA (preserve 95% variance)
    pca = train_pca_baseline(data['X_train'], n_components=0.95)
    
    # Evaluate
    results = evaluate_pca_baseline(
        pca,
        data['X_train'],
        data['X_val'],
        data['test_events']
    )
    
    # Plot
    plot_pca_results(results, results_dir)
    
    # Save
    save_pca_results(pca, results, results_dir)
    
    print("\n" + "=" * 70)
    print("PCA Baseline Complete!")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  Components: {pca.n_components_}")
    print(f"  Train error: {results['train']['mean']:.6f}")
    print(f"  Val 95th percentile: {results['val']['p95']:.6f}")
    print(f"\nAnomaly detection:")
    anomaly_count = sum(1 for d in results['test_events'].values() if d['label'] == 'anomaly')
    normal_count = sum(1 for d in results['test_events'].values() if d['label'] == 'normal')
    print(f"  Anomaly events: {anomaly_count}")
    print(f"  Normal events: {normal_count}")


if __name__ == "__main__":
    main()
