"""
Train XGBoost — training.scripts.train_xgboost
Entry-point for training the XGBoost binary classifier on SCADA data.

Usage:
    python -m src.training.scripts.train_xgboost
    python src/training/scripts/train_xgboost.py --use_stats --n_estimators 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import joblib
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from config import MODELS_DIR, RESULTS_DIR, RANDOM_SEED, ensure_dirs
from models.architectures.xgboost_model import build_xgboost_model
from data_pipeline.loaders.tabular_loader import load_train_val_test, compute_scale_pos_weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train XGBoost on SCADA tabular features.")
    ap.add_argument("--n_estimators",        type=int,   default=400)
    ap.add_argument("--max_depth",           type=int,   default=6)
    ap.add_argument("--learning_rate",       type=float, default=0.05)
    ap.add_argument("--subsample",           type=float, default=0.8)
    ap.add_argument("--colsample_bytree",    type=float, default=0.8)
    ap.add_argument("--min_child_weight",    type=float, default=1.0)
    ap.add_argument("--gamma",               type=float, default=0.0)
    ap.add_argument("--reg_lambda",          type=float, default=1.0)
    ap.add_argument("--reg_alpha",           type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=30)
    ap.add_argument("--tree_method",         type=str,   default="hist")
    ap.add_argument("--use_gpu",             action="store_true")
    ap.add_argument("--use_stats",           action="store_true",
                    help="Use statistical window features instead of raw flattening.")
    ap.add_argument("--seed",                type=int,   default=RANDOM_SEED)
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
    """Grid-search threshold to maximise F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 200):
        m = evaluate_binary(y_true, y_proba, t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return best_t


def plot_feature_importance(model, output_dir: str) -> None:
    """Save XGBoost feature importance bar chart."""
    try:
        import matplotlib.pyplot as plt
        importance = model.feature_importances_
        k = min(30, len(importance))
        top_idx = np.argsort(importance)[::-1][:k]

        plt.figure(figsize=(12, 6))
        plt.bar(range(k), importance[top_idx])
        plt.xlabel("Feature Index")
        plt.ylabel("Importance (gain)")
        plt.title(f"XGBoost — Top {k} Feature Importances")
        plt.tight_layout()
        path = os.path.join(output_dir, "xgboost_feature_importance.png")
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
    print("XGBoost Fault Detector — Training Pipeline")
    print("=" * 70)
    feature_mode = "statistical" if args.use_stats else "raw_flatten"
    print(f"Feature mode: {feature_mode}")

    # Load data
    data = load_train_val_test(use_stats=args.use_stats)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    # Compute class balance
    spw = compute_scale_pos_weight(y_train)

    # Build model
    model = build_xgboost_model(
        scale_pos_weight=spw,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        tree_method=args.tree_method,
        use_gpu=args.use_gpu,
        random_state=args.seed,
    )

    # Early stopping callback
    callbacks = []
    if args.early_stopping_rounds > 0:
        try:
            callbacks.append(xgb.callback.EarlyStopping(
                rounds=args.early_stopping_rounds,
                save_best=True,
                data_name="validation_1",
                maximize=False,
            ))
        except AttributeError:
            print("[WARN] XGBoost version does not support EarlyStopping callback.")

    # Fit
    print(f"\nFitting XGBoost ({args.n_estimators} rounds, early stop={args.early_stopping_rounds})...")
    fit_kwargs = dict(
        X=X_train, y=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    try:
        model.fit(callbacks=callbacks, **fit_kwargs)
    except TypeError:
        model.fit(**fit_kwargs)

    # Threshold tuning on val set
    val_proba = model.predict_proba(X_val)[:, 1]
    best_th = find_best_threshold(y_val, val_proba)
    print(f"\nBest threshold (val): {best_th:.4f}")

    # Evaluate val and test
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
    model_path = os.path.join(MODELS_DIR, "xgboost_7day.pkl")
    joblib.dump({"model": model, "threshold": best_th, "feature_mode": feature_mode}, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save results
    results_dir = os.path.join(RESULTS_DIR, "7day")
    os.makedirs(results_dir, exist_ok=True)
    plot_feature_importance(model, results_dir)

    results = {
        "model": "XGBoost_7day",
        "feature_mode": feature_mode,
        "best_threshold": best_th,
        "params": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "learning_rate": args.learning_rate, "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree, "scale_pos_weight": float(spw),
        },
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }
    result_path = os.path.join(results_dir, "xgboost_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {result_path}")
    print("\nDone! Next: src/evaluation/evaluate_tree.py to compare with LSTM baseline.")


if __name__ == "__main__":
    main()
