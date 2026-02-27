"""
Tabular Loader — data_pipeline.loaders.tabular_loader
Converts SCADA per-event NPZ files into a flat 2D feature matrix
for tree-based models (XGBoost, Random Forest).

Tree models cannot use raw 3D sequences (N, window, features), so this
module flattens (N, W, F) → (N, W*F) and assembles train/test datasets
from the 7day/ directory produced by the LSTM data pipeline.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import WIND_FARM_A_PROCESSED


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D sequence array to 2D tabular matrix.

    Args:
        X: Shape (N, window, features)

    Returns:
        Shape (N, window * features)
    """
    n = X.shape[0]
    return X.reshape(n, -1).astype(np.float32)


def compute_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Compute per-feature summary statistics across the time window.
    Produces a richer tabular representation than raw flattening.

    For each feature, computes: mean, std, min, max, range, p25, p75
    Result shape: (N, n_features * 7)

    Args:
        X: Shape (N, window, features)

    Returns:
        Statistical summary 2D array (N, features*7)
    """
    mean  = X.mean(axis=1)
    std   = X.std(axis=1)
    mn    = X.min(axis=1)
    mx    = X.max(axis=1)
    rng   = mx - mn
    p25   = np.percentile(X, 25, axis=1)
    p75   = np.percentile(X, 75, axis=1)
    return np.concatenate([mean, std, mn, mx, rng, p25, p75], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-event NPZ loading
# ---------------------------------------------------------------------------

def load_event_npz(
    split: str,
    data_dir: Optional[str] = None,
    use_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Load all per-event NPZ files from {data_dir}/{split}_by_event/.
    Returns a flat (N, D) X matrix and binary label vector y.

    Args:
        split: 'train', 'val', 'test', or 'train_by_event' etc.
        data_dir: Root directory (defaults to WIND_FARM_A_PROCESSED/7day/)
        use_stats: If True, use statistical features instead of raw flatten

    Returns:
        Tuple of (X, y, event_ids, event_labels)
    """
    if data_dir is None:
        data_dir = os.path.join(WIND_FARM_A_PROCESSED, "7day")

    split_dir = os.path.join(data_dir, f"{split}_by_event")
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    X_parts, y_parts, event_ids, event_labels = [], [], [], []

    npz_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".npz")])
    if not npz_files:
        raise RuntimeError(f"No NPZ files found in {split_dir}")

    for fname in npz_files:
        try:
            event_id = int(fname.split("_")[1].split(".")[0])
            data = np.load(os.path.join(split_dir, fname), allow_pickle=True)
            X, y_arr = data["X"], data["y"]
            label = str(data["label"])

            if len(X) == 0:
                continue

            # Flatten (N, W, F) → (N, D)
            if X.ndim == 3:
                X_flat = compute_statistical_features(X) if use_stats else flatten_sequences(X)
            elif X.ndim == 2:
                X_flat = X.astype(np.float32)
            else:
                print(f"  Skipping {fname}: unexpected shape {X.shape}")
                continue

            # Binary labels: anomaly=1, normal=0
            if y_arr.ndim > 1:
                y_arr = y_arr.mean(axis=1)
            y_binary = (np.mean(y_arr) > 0).astype(np.float32)
            y_sample = np.full(len(X_flat), float(label == "anomaly"), dtype=np.float32)

            X_parts.append(X_flat)
            y_parts.append(y_sample)
            event_ids.append(event_id)
            event_labels.append(label)

        except Exception as e:
            print(f"  ERROR loading {fname}: {e}")
            continue

    if not X_parts:
        raise RuntimeError(f"No valid NPZ files could be loaded from {split_dir}")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    print(f"  [{split}] Loaded {len(npz_files)} events → X={X_all.shape}, "
          f"pos={y_all.sum():.0f} ({100*y_all.mean():.1f}%)")
    return X_all, y_all, event_ids, event_labels


def load_train_val_test(
    data_dir: Optional[str] = None,
    use_stats: bool = False,
) -> dict:
    """
    Load all three splits at once. Convenience wrapper for training scripts.

    Returns:
        Dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        test_event_ids, test_event_labels
    """
    if data_dir is None:
        data_dir = os.path.join(WIND_FARM_A_PROCESSED, "7day")

    print(f"\nLoading tabular data from: {data_dir}")

    X_train, y_train, _, _ = load_event_npz("train", data_dir, use_stats)
    X_val,   y_val,   _, _ = load_event_npz("val",   data_dir, use_stats)
    X_test,  y_test, test_ids, test_labels = load_event_npz("test", data_dir, use_stats)

    print(f"\nDataset summary:")
    print(f"  Input dim: {X_train.shape[1]}")
    print(f"  Train: {X_train.shape[0]:,} samples (pos={y_train.sum():.0f})")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples ({len(test_ids)} events)")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "test_event_ids":    test_ids,
        "test_event_labels": test_labels,
    }


# ---------------------------------------------------------------------------
# Class balance utility
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute neg/pos ratio for XGBoost scale_pos_weight."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        print("  [WARN] No positive samples in training set — scale_pos_weight = 1.0")
        return 1.0
    ratio = neg / pos
    print(f"  Class balance: {pos:.0f} pos / {neg:.0f} neg → scale_pos_weight={ratio:.2f}")
    return ratio


def event_level_labels(
    data_dir: Optional[str] = None,
    split: str = "test",
) -> Dict[int, str]:
    """
    Return {event_id: 'anomaly'|'normal'} dict for event-level evaluation.
    Reads labels directly from NPZ metadata rather than sample-level y.
    """
    if data_dir is None:
        data_dir = os.path.join(WIND_FARM_A_PROCESSED, "7day")

    split_dir = os.path.join(data_dir, f"{split}_by_event")
    labels = {}
    for fname in os.listdir(split_dir):
        if not fname.endswith(".npz"):
            continue
        try:
            event_id = int(fname.split("_")[1].split(".")[0])
            data = np.load(os.path.join(split_dir, fname), allow_pickle=True)
            labels[event_id] = str(data["label"])
        except Exception:
            pass
    return labels
