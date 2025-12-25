"""
Feature Importance Analysis using Pearson Correlation
Analyze which features are most correlated with anomaly events
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, RESULTS_DIR


def load_test_data():
    """Load test data from event duration test set."""
    test_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day_event_duration', 'test_by_event')
    
    anomaly_data = []
    normal_data = []
    
    print("Loading test events...")
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            
            label = str(data['label'])
            X = data['X']  # Shape: (n_sequences, window_size, n_features)
            
            # Extract last timestep from each sequence (most recent data)
            last_timesteps = X[:, -1, :]  # Shape: (n_sequences, n_features)
            
            if label == 'anomaly':
                anomaly_data.append(last_timesteps)
            else:
                normal_data.append(last_timesteps)
    
    # Concatenate all sequences
    anomaly_features = np.vstack(anomaly_data) if anomaly_data else np.array([])
    normal_features = np.vstack(normal_data) if normal_data else np.array([])
    
    print(f"  Anomaly sequences: {len(anomaly_features)}")
    print(f"  Normal sequences: {len(normal_features)}")
    
    return anomaly_features, normal_features


def load_feature_names():
    """Load feature names from metadata."""
    metadata_path = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day', 'nbm_metadata_7day.pkl')
    metadata = joblib.load(metadata_path)
    return metadata['feature_columns']


def compute_pearson_correlation(anomaly_features, normal_features, feature_names):
    """
    Compute Pearson correlation between each feature and anomaly label.
    
    Strategy:
    1. Combine anomaly and normal features
    2. Create binary labels (1 = anomaly, 0 = normal)
    3. Compute Pearson correlation for each feature with the label
    """
    # Combine data
    all_features = np.vstack([anomaly_features, normal_features])
    labels = np.concatenate([
        np.ones(len(anomaly_features)),
        np.zeros(len(normal_features))
    ])
    
    print(f"\nComputing Pearson correlation for {all_features.shape[1]} features...")
    
    correlations = []
    p_values = []
    
    for i, feature_name in enumerate(feature_names):
        feature_values = all_features[:, i]
        
        # Compute Pearson correlation
        corr, p_val = pearsonr(feature_values, labels)
        
        correlations.append({
            'feature': feature_name,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_val,
            'significant': p_val < 0.05
        })
    
    # Create DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    return corr_df


def compute_feature_statistics(anomaly_features, normal_features, feature_names):
    """
    Compute statistical differences between anomaly and normal features.
    """
    print("\nComputing feature statistics...")
    
    stats = []
    
    for i, feature_name in enumerate(feature_names):
        anomaly_vals = anomaly_features[:, i]
        normal_vals = normal_features[:, i]
        
        stats.append({
            'feature': feature_name,
            'anomaly_mean': np.mean(anomaly_vals),
            'anomaly_std': np.std(anomaly_vals),
            'normal_mean': np.mean(normal_vals),
            'normal_std': np.std(normal_vals),
            'mean_diff': np.mean(anomaly_vals) - np.mean(normal_vals),
            'abs_mean_diff': abs(np.mean(anomaly_vals) - np.mean(normal_vals))
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('abs_mean_diff', ascending=False)
    
    return stats_df


def plot_top_features(corr_df, top_n=20):
    """Plot top N features by Pearson correlation."""
    print(f"\nPlotting top {top_n} features...")
    
    # Create output directory
    output_dir = os.path.join(RESULTS_DIR, 'feature_importance')
    os.makedirs(output_dir, exist_ok=True)
    
    # Top features by absolute correlation
    top_features = corr_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by positive/negative correlation
    colors = ['red' if x > 0 else 'blue' for x in top_features['correlation']]
    
    bars = ax.barh(range(len(top_features)), top_features['correlation'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Pearson Correlation with Anomaly Label', fontsize=12)
    ax.set_title(f'Top {top_n} Features by Pearson Correlation', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Positive correlation (higher in anomalies)'),
        Patch(facecolor='blue', alpha=0.7, label='Negative correlation (lower in anomalies)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'top_features_pearson.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()


def plot_correlation_heatmap(corr_df, top_n=30):
    """Plot heatmap of top features."""
    print(f"\nPlotting correlation heatmap for top {top_n} features...")
    
    output_dir = os.path.join(RESULTS_DIR, 'feature_importance')
    
    top_features = corr_df.head(top_n)
    
    # Create a simple heatmap showing correlation values
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Prepare data for heatmap
    data = top_features[['correlation']].values
    
    sns.heatmap(
        data,
        yticklabels=top_features['feature'],
        xticklabels=['Correlation'],
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )
    
    ax.set_title(f'Top {top_n} Features - Pearson Correlation with Anomaly', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()


def save_results(corr_df, stats_df):
    """Save correlation and statistics results."""
    output_dir = os.path.join(RESULTS_DIR, 'feature_importance')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save correlation results
    corr_path = os.path.join(output_dir, 'pearson_correlation.csv')
    corr_df.to_csv(corr_path, index=False)
    print(f"\nCorrelation results saved to: {corr_path}")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'feature_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Feature statistics saved to: {stats_path}")
    
    # Save top features summary
    summary_path = os.path.join(output_dir, 'feature_importance_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS - PEARSON CORRELATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TOP 20 FEATURES BY ABSOLUTE CORRELATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Feature':<40} {'Correlation':<15} {'P-value':<12} {'Significant'}\n")
        f.write("-" * 80 + "\n")
        
        for idx, row in corr_df.head(20).iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            f.write(f"{idx+1:<6} {row['feature']:<40} {row['correlation']:>+.6f}     {row['p_value']:<12.6f} {sig}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("=" * 80 + "\n")
        f.write("* Positive correlation: Feature values tend to be HIGHER during anomalies\n")
        f.write("* Negative correlation: Feature values tend to be LOWER during anomalies\n")
        f.write("* Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n")
        
        # Count significant features
        n_significant = (corr_df['p_value'] < 0.05).sum()
        f.write(f"\nTotal significant features (p<0.05): {n_significant}/{len(corr_df)}\n")
        
        # Top positive and negative correlations
        top_positive = corr_df[corr_df['correlation'] > 0].head(5)
        top_negative = corr_df[corr_df['correlation'] < 0].head(5)
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 5 FEATURES WITH POSITIVE CORRELATION (Higher in Anomalies):\n")
        f.write("-" * 80 + "\n")
        for idx, row in top_positive.iterrows():
            f.write(f"  {row['feature']:<40} r={row['correlation']:>+.4f} (p={row['p_value']:.6f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 5 FEATURES WITH NEGATIVE CORRELATION (Lower in Anomalies):\n")
        f.write("-" * 80 + "\n")
        for idx, row in top_negative.iterrows():
            f.write(f"  {row['feature']:<40} r={row['correlation']:>+.4f} (p={row['p_value']:.6f})\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS - PEARSON CORRELATION")
    print("=" * 80)
    
    # Load data
    anomaly_features, normal_features = load_test_data()
    
    if len(anomaly_features) == 0 or len(normal_features) == 0:
        print("Error: No data loaded!")
        return
    
    # Load feature names
    feature_names = load_feature_names()
    print(f"\nTotal features: {len(feature_names)}")
    
    # Compute Pearson correlation
    corr_df = compute_pearson_correlation(anomaly_features, normal_features, feature_names)
    
    # Compute feature statistics
    stats_df = compute_feature_statistics(anomaly_features, normal_features, feature_names)
    
    # Merge correlation and statistics
    merged_df = corr_df.merge(stats_df, on='feature')
    
    # Display top features
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES BY PEARSON CORRELATION")
    print("=" * 80)
    print(f"{'Rank':<6} {'Feature':<40} {'Correlation':<15} {'P-value':<12}")
    print("-" * 80)
    
    for idx, row in corr_df.head(20).iterrows():
        print(f"{idx+1:<6} {row['feature']:<40} {row['correlation']:>+.6f}     {row['p_value']:<12.6f}")
    
    # Plot results
    plot_top_features(corr_df, top_n=20)
    plot_correlation_heatmap(corr_df, top_n=30)
    
    # Save results
    save_results(merged_df, stats_df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
