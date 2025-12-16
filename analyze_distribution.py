"""
Analyze class distribution in train/val/test datasets.
"""

import numpy as np
import os

# Define path directly
WIND_FARM_A_PROCESSED = os.path.join("Dataset", "processed", "Wind Farm A")


def analyze_distribution(split_name, y):
    """Analyze and print class distribution."""
    n_total = len(y)
    n_positive = y.sum()
    n_negative = n_total - n_positive
    
    pct_positive = (n_positive / n_total) * 100 if n_total > 0 else 0
    pct_negative = (n_negative / n_total) * 100 if n_total > 0 else 0
    
    print(f"\n{split_name}:")
    print(f"  Total samples:     {n_total:,}")
    print(f"  Negative (0):      {n_negative:,}  ({pct_negative:.2f}%)")
    print(f"  Positive (1):      {n_positive:,}  ({pct_positive:.2f}%)")
    print(f"  Imbalance ratio:   1:{n_negative/n_positive:.1f}" if n_positive > 0 else "  Imbalance ratio:   N/A")
    
    return {
        'total': n_total,
        'negative': n_negative,
        'positive': n_positive,
        'pct_positive': pct_positive,
        'pct_negative': pct_negative
    }


def main():
    print("=" * 70)
    print("Class Distribution Analysis - SCADA Fault Prediction")
    print("=" * 70)
    
    # Load data
    print("\nLoading datasets...")
    y_train = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_train.npy"))
    y_val = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_val.npy"))
    y_test = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_test.npy"))
    
    # Analyze each split
    train_stats = analyze_distribution("TRAIN SET", y_train)
    val_stats = analyze_distribution("VALIDATION SET", y_val)
    test_stats = analyze_distribution("TEST SET", y_test)
    
    # Overall statistics
    total_samples = train_stats['total'] + val_stats['total'] + test_stats['total']
    total_positive = train_stats['positive'] + val_stats['positive'] + test_stats['positive']
    total_negative = train_stats['negative'] + val_stats['negative'] + test_stats['negative']
    
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total samples:     {total_samples:,}")
    print(f"Total negative:    {total_negative:,}  ({total_negative/total_samples*100:.2f}%)")
    print(f"Total positive:    {total_positive:,}  ({total_positive/total_samples*100:.2f}%)")
    print(f"Imbalance ratio:   1:{total_negative/total_positive:.1f}")
    
    print("\n" + "=" * 70)
    print("INSIGHTS")
    print("=" * 70)
    print(f"✓ Extremely imbalanced dataset ({total_positive/total_samples*100:.2f}% positive)")
    print(f"✓ For every 1 fault sample, there are ~{total_negative/total_positive:.0f} normal samples")
    print(f"✓ Traditional accuracy is misleading - predicting all 0's gives {total_negative/total_samples*100:.2f}% accuracy")
    print(f"✓ Better metrics: ROC-AUC, Precision-Recall AUC, F1-Score")
    
    # Expected baseline
    expected_precision_random = total_positive / total_samples
    print(f"\n✓ Random classifier expected precision: {expected_precision_random*100:.4f}%")
    print(f"✓ Your baseline precision (0.7%) is ~{0.007/expected_precision_random:.1f}x better than random")


if __name__ == "__main__":
    main()
