"""
Train Random Forest — training.scripts.train_random_forest
Entry-point for training the Random Forest binary classifier on SCADA data.

Usage:
    python -m src.training.scripts.train_random_forest
    python src/training/scripts/train_random_forest.py --n_estimators 500 --use_stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from config import MODELS_DIR, RESULTS_DIR, RANDOM_SEED, ensure_dirs
from models.architectures.random_forest import build_random_forest_model
from data_pipeline.loaders.tabular_loader import load_train_val_test


# ---------------------------------------------------------------------------
# Helpers (shared metric helpers mirrored from train_xgboost.py)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Random Forest on SCADA tabular features.")
    ap.add_argument("--n_estimators",       type=int,   default=300)
    ap.add_argument("--max_depth",          type=int,   default=None)
    ap.add_argument("--min_samples_split",  type=int,   default=2)
    ap.add_argument("--min_samples_leaf",   type=int,   default=1)
    ap.add_argument("--max_features",       type=str,   default="sqrt",
                    help="Features per split: 'sqrt', 'log2', or a float ratio.")
    ap.add_argument("--class_weight",       type=str,   default="balanced")
    ap.add_argument("--use_stats",          action="store_true",
                    help="Use statistical window features instead of raw flattening.")
    ap.add_argument("--seed",               type=int,   default=RANDOM_SEED)
    return ap.parse_args()


def evaluate_binary(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    far  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    acc  = (TP + TN) / max(TP + FP + TN + FN, 1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "far": far,
            "f1": f1, "TP": TP, "FP": FP, "TN": TN, "FN": FN, "threshold": threshold}


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Grid-search threshold to maximise F1 on a validation set."""
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 200):
        m = evaluate_binary(y_true, y_proba, t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return best_t


def plot_feature_importance(model, output_dir: str) -> None:
    """Save Random Forest feature importance bar chart."""
    try:
        import matplotlib.pyplot as plt
        importance = model.feature_importances_
        k = min(30, len(importance))
        top_idx = np.argsort(importance)[::-1][:k]

        plt.figure(figsize=(12, 6))
        plt.bar(range(k), importance[top_idx])
        plt.xlabel("Feature Index")
        plt.ylabel("Importance (Gini)")
        plt.title(f"Random Forest — Top {k} Feature Importances")
        plt.tight_layout()
        path = os.path.join(output_dir, "rf_feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Feature importance saved to: {path}")
    except Exception as e:
        print(f"[WARN] Could not plot feature importance: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    ensure_dirs()

    print("=" * 70)
    print("Random Forest Fault Detector — Training Pipeline")
    print("=" * 70)
    feature_mode = "statistical" if args.use_stats else "raw_flatten"
    print(f"Feature mode: {feature_mode}")

    # Load data
    data = load_train_val_test(use_stats=args.use_stats)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    # Parse max_features (could be float from CLI)
    max_features = args.max_features
    try:
        max_features = float(max_features)
        if max_features >= 1.0:
            max_features = int(max_features)
    except ValueError:
        pass  # keep as string 'sqrt' / 'log2'

    # Build model
    model = build_random_forest_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        class_weight=args.class_weight,
        random_state=args.seed,
    )

    # Fit (RF does not have native early stopping — train fully)
    print(f"\nFitting Random Forest ({args.n_estimators} trees)...")
    print("  [Note] Training may take 1–5 min with large flattened features.")
    model.fit(X_train, y_train)
    print("  Training complete.")

    # OOB estimate (if enabled)
    if hasattr(model, "oob_score_") and model.oob_score_:
        print(f"  OOB accuracy: {model.oob_score_:.4f}")

    # Threshold tuning on val set
    val_proba = model.predict_proba(X_val)[:, 1]
    best_th = find_best_threshold(y_val, val_proba)
    print(f"\nBest threshold (val): {best_th:.4f}")

    # Evaluate
    val_metrics  = evaluate_binary(y_val,  val_proba, best_th)
    test_proba   = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_binary(y_test, test_proba, best_th)

    print("\n" + "=" * 70)
    print(f"VAL  — Acc={val_metrics['accuracy']:.2%}  P={val_metrics['precision']:.2%}  "
          f"R={val_metrics['recall']:.2%}  FAR={val_metrics['far']:.2%}  F1={val_metrics['f1']:.4f}")
    print(f"TEST — Acc={test_metrics['accuracy']:.2%} P={test_metrics['precision']:.2%}  "
          f"R={test_metrics['recall']:.2%}  FAR={test_metrics['far']:.2%}  F1={test_metrics['f1']:.4f}")
    print("=" * 70)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "random_forest_7day.pkl")
    joblib.dump({"model": model, "threshold": best_th, "feature_mode": feature_mode}, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save results
    results_dir = os.path.join(RESULTS_DIR, "7day")
    os.makedirs(results_dir, exist_ok=True)
    plot_feature_importance(model, results_dir)

    results = {
        "model": "RandomForest_7day",
        "feature_mode": feature_mode,
        "best_threshold": best_th,
        "params": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": str(max_features),
            "class_weight": args.class_weight,
        },
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }
    result_path = os.path.join(results_dir, "random_forest_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {result_path}")
    print("\nDone! Next: src/evaluation/evaluate_tree.py to compare with LSTM/XGBoost.")


if __name__ == "__main__":
    main()
