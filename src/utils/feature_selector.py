"""
Feature Selection Utility
Provides functions to select and filter top features based on Pearson correlation analysis
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


# Top 30 features from Pearson correlation analysis
TOP_30_FEATURES = [
    'sensor_5_avg_cos',
    'sensor_5_max_cos',
    'sensor_5_min_cos',
    'sensor_5_avg_sin',
    'sensor_5_max_sin',
    'sensor_5_min_sin',
    'sensor_12_avg',
    'sensor_11_avg',
    'sensor_10_avg',
    'sensor_9_avg',
    'sensor_52_max',
    'sensor_18_max',
    'sensor_18_avg',
    'sensor_52_avg',
    'sensor_18_min',
    'sensor_52_min',
    'sensor_14_avg',
    'sensor_42_avg_cos',
    'reactive_power_28_min',
    'sensor_33_avg',
    'reactive_power_27_max',
    'reactive_power_27_avg',
    'reactive_power_28_avg',
    'sensor_6_avg',
    'sensor_31_avg',
    'sensor_1_avg_cos',
    'sensor_20_avg',
    'sensor_8_avg',
    'sensor_35_avg',
    'sensor_36_avg'
]


def get_top_features(n=30):
    """
    Get top N features from Pearson correlation analysis.
    
    Args:
        n: Number of top features to return (default: 30)
    
    Returns:
        List of feature names
    """
    if n == 30:
        return TOP_30_FEATURES.copy()
    
    # Load from CSV for other values
    corr_file = os.path.join(RESULTS_DIR, 'feature_importance', 'pearson_correlation.csv')
    if os.path.exists(corr_file):
        df = pd.read_csv(corr_file)
        return df['feature'].head(n).tolist()
    
    # Fallback to hardcoded top 30
    return TOP_30_FEATURES[:n]


def get_feature_indices(all_features, selected_features):
    """
    Get indices of selected features in the full feature list.
    
    Args:
        all_features: List of all feature names (80 features)
        selected_features: List of selected feature names (e.g., top 30)
    
    Returns:
        List of indices
    """
    indices = []
    for feat in selected_features:
        if feat in all_features:
            indices.append(all_features.index(feat))
        else:
            print(f"Warning: Feature '{feat}' not found in all_features")
    
    return indices


def filter_features(data, all_features, selected_features):
    """
    Filter data to only include selected features.
    
    Args:
        data: numpy array of shape (n_samples, n_features) or (n_samples, seq_len, n_features)
        all_features: List of all feature names
        selected_features: List of selected feature names
    
    Returns:
        Filtered data with only selected features
    """
    indices = get_feature_indices(all_features, selected_features)
    
    if len(data.shape) == 2:
        # Shape: (n_samples, n_features)
        return data[:, indices]
    elif len(data.shape) == 3:
        # Shape: (n_samples, seq_len, n_features)
        return data[:, :, indices]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")


def load_top_features_from_csv(top_n=30):
    """
    Load top N features from Pearson correlation CSV file.
    
    Args:
        top_n: Number of top features to load
    
    Returns:
        DataFrame with top features and their correlations
    """
    corr_file = os.path.join(RESULTS_DIR, 'feature_importance', 'pearson_correlation.csv')
    
    if not os.path.exists(corr_file):
        raise FileNotFoundError(f"Correlation file not found: {corr_file}")
    
    df = pd.read_csv(corr_file)
    return df.head(top_n)


def print_feature_selection_info(selected_features=None):
    """Print information about selected features."""
    if selected_features is None:
        selected_features = TOP_30_FEATURES
    
    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION INFO")
    print(f"{'='*80}")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"\nSelected features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test the utility
    print("Testing Feature Selection Utility")
    print_feature_selection_info()
    
    # Load and display correlation info
    try:
        top_df = load_top_features_from_csv(30)
        print("\nTop 30 Features with Correlations:")
        print(top_df[['feature', 'correlation', 'p_value']].to_string(index=False))
    except FileNotFoundError as e:
        print(f"Warning: {e}")
