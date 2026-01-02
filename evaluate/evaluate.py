"""
Evaluate the saved LSTM model on test data.
"""

import os
import sys
import numpy as np
import json

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR


def load_data():
    """Load preprocessed test data."""
    print("Loading test data...")
    X_test = np.load(os.path.join(WIND_FARM_A_PROCESSED, "X_test.npy"))
    y_test = np.load(os.path.join(WIND_FARM_A_PROCESSED, "y_test.npy"))
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_test, y_test


def evaluate_model(model, X, y, dataset_name="Test"):
    """Evaluate model and return metrics."""
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else 0.0
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
    print("=" * 60)
    print("SCADA Fault Prediction - LSTM Evaluation")
    print("=" * 60)
    
    # Load test data
    X_test, y_test = load_data()
    
    # Load saved model
    model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Evaluate
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Save results
    results = {
        'model': 'LSTM (Best Checkpoint)',
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(RESULTS_DIR, "lstm_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
