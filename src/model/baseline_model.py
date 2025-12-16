"""
Baseline Model for SCADA Fault Prediction.

This module implements a simple Logistic Regression baseline using
aggregated statistical features from each 48-hour window.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from sequences for baseline model.
    
    For each sequence, compute: mean, std, min, max for each feature.
    This reduces (n_samples, seq_len, n_features) -> (n_samples, n_features * 4)
    
    Args:
        X: Sequence array of shape (n_samples, seq_len, n_features)
        
    Returns:
        Feature array of shape (n_samples, n_features * 4)
    """
    n_samples, seq_len, n_features = X.shape
    
    # Compute statistics along the time axis (axis=1)
    mean_features = np.mean(X, axis=1)
    std_features = np.std(X, axis=1)
    min_features = np.min(X, axis=1)
    max_features = np.max(X, axis=1)
    
    # Concatenate all features
    features = np.concatenate([
        mean_features, std_features, min_features, max_features
    ], axis=1)
    
    return features


def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    
    X_train = np.load(os.path.join(WIND_FARM_A_PROCESSED, "X_train.npy"))
    X_val = np.load(os.path.join(WIND_FARM_A_PROCESSED, "X_val.npy"))
    X_test = np.load(os.path.join(WIND_FARM_A_PROCESSED, "X_test.npy"))
    
    y_train = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_train.npy"))
    y_val = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_val.npy"))
    y_test = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_test.npy"))
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """
    Train Logistic Regression baseline model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained LogisticRegression model
    """
    print("\nExtracting statistical features...")
    X_train_features = extract_statistical_features(X_train)
    X_val_features = extract_statistical_features(X_val)
    
    print(f"  Train features shape: {X_train_features.shape}")
    print(f"  Val features shape: {X_val_features.shape}")
    
    # Calculate class weights for imbalanced data
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    
    class_weight = {
        0: n_samples / (2 * n_neg),
        1: n_samples / (2 * n_pos)
    }
    print(f"\nClass weights: {class_weight}")
    
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_features, y_train)
    
    # Validate
    train_acc = model.score(X_train_features, y_train)
    val_acc = model.score(X_val_features, y_val)
    
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Val accuracy: {val_acc:.4f}")
    
    return model


def evaluate_model(model: LogisticRegression, X: np.ndarray, y: np.ndarray,
                   dataset_name: str = "Test") -> dict:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X: Feature sequences
        y: True labels
        dataset_name: Name for printing
        
    Returns:
        Dictionary of metrics
    """
    X_features = extract_statistical_features(X)
    y_pred = model.predict(X_features)
    y_prob = model.predict_proba(X_features)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix ({dataset_name}):")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    print(f"\nClassification Report ({dataset_name}):")
    print(classification_report(y, y_pred, target_names=['Normal', 'Fault']))
    
    return metrics


def main():
    """Main training and evaluation pipeline."""
    print("=" * 60)
    print("SCADA Fault Prediction - Baseline Model")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_dirs()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Train model
    model = train_baseline(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "baseline_logistic_regression.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save results
    results = {
        'model': 'Logistic Regression Baseline',
        'test_metrics': test_metrics
    }
    results_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Baseline training complete!")
    print("=" * 60)
    
    return model, test_metrics


if __name__ == "__main__":
    main()
