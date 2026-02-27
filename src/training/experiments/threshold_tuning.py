"""
Threshold Tuning — training.experiments.threshold_tuning
Determines anomaly detection thresholds using a hybrid IQR + PR-based approach.

Run after training to find the optimal threshold before final test evaluation.

Usage:
    python -m src.training.experiments.threshold_tuning
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(data_split: str = 'test') -> dict:
    """Load per-event NPZ files from NBM_7day/{data_split}_by_event/."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    split_dir = os.path.join(nbm_dir, f'{data_split}_by_event')
    events = {}

    print(f"Loading {data_split} events from: {split_dir}")
    if not os.path.exists(split_dir):
        print(f"  ERROR: Directory not found: {split_dir}")
        return {}

    for filename in os.listdir(split_dir):
        if not filename.endswith('.npz'):
            continue
        try:
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(split_dir, filename), allow_pickle=True)
            if 'X' in data and 'y' in data:
                events[event_id] = {
                    'X': data['X'],
                    'y': data['y'],
                    'label': str(data['label']),
                }
            else:
                print(f"  Skipping {filename}: missing X/y keys")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    print(f"  Loaded {len(events)} {data_split} events")
    return events


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def smooth_mae(mae: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply simple moving average to MAE signal to reduce noise spikes."""
    if len(mae) <= window:
        return mae
    return np.convolve(mae, np.ones(window) / window, mode='same')


def compute_event_scores(model, events: dict) -> tuple:
    """
    Compute p95 MAE per event.

    Returns:
        Tuple of (scores_array, labels_array, detailed_list)
    """
    scores, labels, detailed = [], [], []

    for event_id, data in sorted(events.items()):
        X, y = data['X'], data['y']
        true_label = data['label']

        y_pred = model.predict(X, verbose=0, batch_size=256)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        mae = smooth_mae(mae, window=3)
        event_p95 = np.percentile(mae, 95)

        scores.append(event_p95)
        labels.append(1 if true_label == 'anomaly' else 0)
        detailed.append({
            'event_id': int(event_id),
            'true_label': true_label,
            'mae_mean': float(np.mean(mae)),
            'mae_p95': float(event_p95),
            'mae_max': float(np.max(mae)),
            'n_samples': int(len(mae)),
        })

    return np.array(scores), np.array(labels), detailed


# ---------------------------------------------------------------------------
# Threshold determination (Hybrid IQR + PR)
# ---------------------------------------------------------------------------

def determine_threshold(
    model,
    val_events: dict,
    train_events: dict,
    min_recall: float = 0.7,
    smoothing_window: int = 3,
    iqr_multiplier: float = 1.5,
) -> tuple:
    """
    Compute the anomaly detection threshold using a hybrid strategy:
      1. PR-curve: find threshold maximising F1 while recall >= min_recall
      2. IQR (normal only): upper fence from normal training events
      3. Hybrid: min(IQR_upper, PR_upper)

    Args:
        model: Trained Keras model
        val_events: Validation event dict from load_events('val')
        train_events: Training event dict from load_events('train')
        min_recall: Minimum recall constraint for PR threshold
        smoothing_window: MAE smoothing window size
        iqr_multiplier: IQR fence multiplier (default 1.5)

    Returns:
        Tuple of (upper_threshold, lower_threshold)
    """
    print("\n[Threshold] Computing validation scores...")
    val_scores, val_labels, val_detailed = compute_event_scores(model, val_events)

    # 1. PR-based threshold
    prec, rec, thresholds = precision_recall_curve(val_labels, val_scores)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-6)
    valid_idx = np.where(rec >= min_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
    else:
        best_idx = np.argmax(f1_scores)
        print(f"  [WARN] No PR threshold meets recall >= {min_recall}. Using max-F1 fallback.")

    pr_upper_th = thresholds[best_idx]
    best_prec, best_rec, best_f1 = prec[best_idx], rec[best_idx], f1_scores[best_idx]

    # 2. IQR from normal training events
    print("\n[Threshold] Computing IQR from normal training events...")
    train_scores, train_labels, _ = compute_event_scores(model, train_events)
    normal_scores = train_scores[train_labels == 0]

    if len(normal_scores) >= 4:
        Q1 = np.percentile(normal_scores, 25)
        Q3 = np.percentile(normal_scores, 75)
        IQR_val = Q3 - Q1
        iqr_upper = Q3 + iqr_multiplier * IQR_val
        iqr_lower = max(0.0, Q1 - iqr_multiplier * IQR_val)
        print(f"  Q1={Q1:.6f}, Q3={Q3:.6f}, IQR={IQR_val:.6f}")
        print(f"  IQR Upper: {iqr_upper:.6f}")
    else:
        iqr_upper = np.percentile(normal_scores, 99) if len(normal_scores) > 0 else pr_upper_th
        iqr_lower = 0.0
        print(f"  [WARN] Too few normal events for IQR. Using p99 fallback: {iqr_upper:.6f}")

    # 3. Hybrid
    final_upper = min(iqr_upper, pr_upper_th)
    final_lower = iqr_lower

    print("\n[Threshold] HYBRID THRESHOLD SELECTION")
    print(f"  IQR Upper  (normal-only): {iqr_upper:.6f}")
    print(f"  PR Upper   (rec>={min_recall}): {pr_upper_th:.6f}")
    print(f"  ==> Final Upper Threshold: {final_upper:.6f}")
    print(f"  PR @ chosen: Prec={best_prec:.4f}, Rec={best_rec:.4f}, F1={best_f1:.4f}")

    # Save
    output_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'val_detailed.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'method': 'IQR_NORMAL_PLUS_PR_HYBRID',
            'upper_threshold': float(final_upper),
            'lower_threshold': float(final_lower),
            'iqr_upper': float(iqr_upper),
            'pr_upper': float(pr_upper_th),
            'iqr_multiplier': iqr_multiplier,
            'min_recall_target': min_recall,
            'val_detailed': val_detailed,
        }, f, indent=2)
    print(f"\n[Threshold] Saved to: {output_path}")

    return final_upper, final_lower


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from tensorflow import keras
    print("=" * 80)
    print("NBM LSTM — Threshold Tuning")
    print("=" * 80)

    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_7day.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run training first: python -m src.training.scripts.train_nbm")
        return

    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")

    train_events = load_events('train')
    val_events   = load_events('val')

    if not val_events:
        print("Error: No val events found.")
        return

    upper_th, lower_th = determine_threshold(model, val_events, train_events, min_recall=0.7)
    print(f"\nFinal threshold — Upper: {upper_th:.6f} | Lower: {lower_th:.6f}")
    print("\nNext step: run src/evaluation/evaluate_nbm.py for test set evaluation.")


if __name__ == '__main__':
    main()
