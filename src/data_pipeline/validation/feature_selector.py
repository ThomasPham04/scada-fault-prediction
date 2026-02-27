"""
Feature Selector — data_pipeline.validation.feature_selector
Model-based feature selection using a lightweight probe LSTM autoencoder.
Ranks features via permutation importance, error sensitivity, and group ablation.
"""

import numpy as np
import json
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import NBM_WINDOW_SIZE, NBM_STRIDE


# ---------------------------------------------------------------------------
# Probe Model helpers
# ---------------------------------------------------------------------------

def normalize_probe_data(X: np.ndarray) -> tuple:
    """Standardize probe sequences in-place. Returns (normalized, (mean, std))."""
    mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6
    return (X - mean) / std, (mean, std)


def subsample_for_probe(data: np.ndarray, max_samples: int = 3_000) -> np.ndarray:
    """Uniformly subsample rows to speed up probe training."""
    if len(data) <= max_samples:
        return data
    idx = np.linspace(0, len(data) - 1, max_samples, dtype=int)
    return data[idx]


def build_probe_LSTM(input_shape: tuple):
    """
    Build a lightweight LSTM seq2seq autoencoder as a probe model.
    Used for model-based feature importance estimation.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
    from tensorflow.keras.initializers import GlorotUniform, Orthogonal
    from tensorflow.keras.optimizers import SGD

    kernel_init = GlorotUniform(seed=42)
    recurrent_init = Orthogonal(seed=42)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape,
             kernel_initializer=kernel_init, recurrent_initializer=recurrent_init),
        LSTM(32, return_sequences=False,
             kernel_initializer=kernel_init, recurrent_initializer=recurrent_init),
        RepeatVector(input_shape[0]),
        LSTM(32, return_sequences=True,
             kernel_initializer=kernel_init, recurrent_initializer=recurrent_init),
        LSTM(64, return_sequences=True,
             kernel_initializer=kernel_init, recurrent_initializer=recurrent_init),
        TimeDistributed(Dense(input_shape[1], kernel_initializer=kernel_init)),
    ])
    model.compile(optimizer=SGD(learning_rate=1e-2, momentum=0.0), loss='mae')
    return model


# ---------------------------------------------------------------------------
# Importance metrics
# ---------------------------------------------------------------------------

def permutation_importance(model, X: np.ndarray, feature_cols: list, model_pred: np.ndarray) -> dict:
    """Measure importance via increase in reconstruction error when a feature is shuffled."""
    model_error = np.mean(np.abs(model_pred - X), axis=(0, 1))
    feat_importances = {}
    for i, feat in enumerate(feature_cols):
        rng = np.random.default_rng(42 + i)
        X_permuted = X.copy()
        perm_idx = rng.permutation(X.shape[1])
        X_permuted[:, perm_idx, i] = X[:, :, i]
        pred = model.predict(X_permuted, batch_size=256, verbose=0)
        permuted_error = np.mean(np.abs(pred - X_permuted), axis=(0, 1))
        feat_importances[feat] = float(np.mean(permuted_error - model_error))
    return feat_importances


def error_sensitivity(model_pred: np.ndarray, X: np.ndarray, feature_cols: list) -> dict:
    """Per-feature mean+std reconstruction error as a sensitivity score."""
    errors = np.abs(model_pred - X)
    sensitivity = {}
    for i, feat in enumerate(feature_cols):
        mean_err = errors[:, :, i].mean()
        std_err = errors[:, :, i].std()
        sensitivity[feat] = mean_err + std_err
    return sensitivity


def group_features(feature_cols: list) -> dict:
    """Group features by their name prefix (e.g. 'sensor', 'wind_speed', 'power')."""
    groups = {}
    for f in feature_cols:
        key = f.split('_')[0]
        groups.setdefault(key, []).append(f)
    return groups


def _to_scalar(x) -> float:
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(np.mean(x))
    return float(x)


def group_ablation_score(model, X: np.ndarray, groups: dict, feature_cols: list, model_pred: np.ndarray) -> dict:
    """Measure the increase in error when an entire feature group is zeroed out."""
    base_errors = np.mean(np.abs(model_pred - X))
    group_score = {}
    for g_name, feats in groups.items():
        idx = [i for i, f in enumerate(feature_cols) if f in feats]
        X_zero = X.copy()
        X_zero[:, :, idx] = 0
        err_zero = np.mean(np.abs(model.predict(X_zero) - X_zero))
        group_score[g_name] = err_zero - base_errors
    return group_score


def final_feature_selection_scores(perm: dict, sens: dict, group: dict, feature_cols: list) -> dict:
    """Combine permutation, sensitivity, and group ablation into a unified score."""
    final_scores = {}
    for f in feature_cols:
        g = f.split('_')[0]
        p = _to_scalar(perm.get(f, 0))
        s = _to_scalar(sens.get(f, 0))
        g_score = _to_scalar(group.get(g, 0))
        final_scores[f] = 0.5 * p + 0.3 * s + 0.2 * g_score
    return final_scores


def select_top_features(score_dict: dict, top_k_ratio: float) -> list:
    """Return the top-k% features sorted by descending score."""
    sorted_feats = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    k = int(len(sorted_feats) * top_k_ratio)
    return [f for f, _ in sorted_feats[:k]]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def model_based_feature_selection(
    X_train: np.ndarray,
    feature_cols: list,
    window_size: int = NBM_WINDOW_SIZE,
    stride: int = NBM_STRIDE,
    probe_epochs: int = 10,
    top_k_ratio: float = 0.4,
    batch_size: int = 64,
) -> tuple:
    """
    Run full model-based feature selection using a probe LSTM autoencoder.

    Steps:
      1. Subsample training data for speed
      2. Create probe sequences and normalise
      3. Train probe LSTM
      4. Compute permutation importance, error sensitivity, group ablation
      5. Combine into final scores and select top-k features

    Args:
        X_train: Raw training data (T, F)
        feature_cols: List of feature names (length F)
        window_size: Probe sequence length
        stride: Stride for probe sequences
        probe_epochs: Epochs to train the probe
        top_k_ratio: Fraction of features to keep (e.g. 0.4 = 40%)
        batch_size: Probe training batch size

    Returns:
        Tuple of (selected_features, scores_dict, top_k_ratio)
    """
    import tensorflow as tf
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()

    from data_pipeline.loaders.sequence_maker import create_sequences, create_probe_sequences

    print("[FS] Subsampling training data for probe model...")
    X_sub = subsample_for_probe(X_train, max_samples=12_000)

    print("[FS] Creating probe sequences...")
    X = create_probe_sequences(X_sub, window_size=window_size, stride=stride)
    X, _ = normalize_probe_data(X)

    print("[FS] Building probe LSTM model...")
    np.random.seed(42)
    tf.random.set_seed(42)
    model = build_probe_LSTM(X.shape[1:])
    model.fit(X, X, epochs=probe_epochs, batch_size=batch_size, verbose=1, shuffle=False)

    print("[FS] Computing predictions...")
    model_pred = model.predict(X)

    print("[FS] Computing permutation feature importance...")
    np.random.seed(42)
    perm = permutation_importance(model, X, feature_cols, model_pred)

    print("[FS] Computing error sensitivity...")
    sens = error_sensitivity(model_pred, X, feature_cols)

    print("[FS] Computing group ablation scores...")
    groups = group_features(feature_cols)
    group_score = group_ablation_score(model, X, groups, feature_cols, model_pred)

    print("[FS] Final feature scoring...")
    scores = final_feature_selection_scores(perm, sens, group_score, feature_cols)

    selected_features = select_top_features(scores, top_k_ratio)
    print(f"[FS] Selected top {len(selected_features)} features out of {len(feature_cols)}")

    return selected_features, scores, top_k_ratio


def export_feature_selection_json(
    selected_features: list,
    scores: dict,
    output_dir: str,
    top_k_ratio: float,
) -> None:
    """Save feature selection results as a JSON file for analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_data = {
        "meta": {
            "method": "model_based_feature_selection",
            "top_k_ratio": top_k_ratio,
            "num_total_features": len(scores),
            "num_selected_features": len(selected_features),
        },
        "selected_features": [{"feature": f, "score": float(scores[f])} for f in selected_features],
        "all_feature_scores": {k: float(v) for k, v in scores.items()},
    }

    out_path = output_dir / "model_based_feature_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"[FS] Feature selection results saved to: {out_path}")
