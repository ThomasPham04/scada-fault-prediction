"""
Evaluate Tree Models — evaluation.evaluate_tree
Loads saved XGBoost or Random Forest models and evaluates them on test events.
Produces confusion matrix, metrics, feature importance, and model comparison.

Usage:
    python -m src.evaluation.evaluate_tree --model xgboost
    python -m src.evaluation.evaluate_tree --model random_forest
    python -m src.evaluation.evaluate_tree --model both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import joblib
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import MODELS_DIR, RESULTS_DIR, WIND_FARM_A_PROCESSED
from data_pipeline.loaders.tabular_loader import load_train_val_test, event_level_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate tree models on SCADA test events.")
    ap.add_argument("--model", type=str, default="both",
                    choices=["xgboost", "random_forest", "both"])
    ap.add_argument("--use_stats", action="store_true",
                    help="Use statistical feature mode (must match training mode).")
    return ap.parse_args()


def evaluate_binary(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Binary classification metrics at fixed threshold."""
    y_pred = (y_proba >= threshold).astype(int)
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
            "f1": f1, "TP": TP, "FP": FP, "TN": TN, "FN": FN}


# ---------------------------------------------------------------------------
# Per-event evaluation
# ---------------------------------------------------------------------------

def evaluate_event_level(
    model,
    threshold: float,
    feature_mode: str,
    model_name: str,
) -> dict:
    """
    Event-level detection: classify each event as anomaly/normal
    based on how many of its samples exceed the threshold.

    An event is flagged as anomaly if >15% of its sample predictions
    exceed the threshold (majority vote with a low bar to improve recall).
    """
    data_dir = os.path.join(WIND_FARM_A_PROCESSED, "7day")
    split_dir = os.path.join(data_dir, "test_by_event")
    use_stats = feature_mode == "statistical"

    from data_pipeline.loaders.tabular_loader import (
        flatten_sequences,
        compute_statistical_features,
    )

    TP = FP = TN = FN = 0
    results = []

    print(f"\n{'='*100}")
    print(f"{model_name} — Event-Level Detection (threshold={threshold:.4f})")
    print(f"{'='*100}")
    print(f"{'Event':<8} {'True':<12} {'Predicted':<12} {'Result':<10} "
          f"{'Anomaly %':<12} {'Mean Prob':<12}")
    print("-" * 100)

    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith(".npz"):
            continue
        try:
            event_id = int(fname.split("_")[1].split(".")[0])
            raw = np.load(os.path.join(split_dir, fname), allow_pickle=True)
            X = raw["X"]
            true_label = str(raw["label"])

            if X.ndim == 3:
                X_flat = compute_statistical_features(X) if use_stats else flatten_sequences(X)
            else:
                X_flat = X.astype(np.float32)

            proba = model.predict_proba(X_flat)[:, 1]
            mean_prob = float(proba.mean())
            anomaly_ratio = float((proba >= threshold).mean())

            detected = anomaly_ratio >= 0.15
            is_anomaly = true_label == "anomaly"

            if is_anomaly and detected:
                res, symbol = "TP", "[OK] TP"; TP += 1
            elif not is_anomaly and not detected:
                res, symbol = "TN", "[OK] TN"; TN += 1
            elif not is_anomaly and detected:
                res, symbol = "FP", "[X]  FP"; FP += 1
            else:
                res, symbol = "FN", "[X]  FN"; FN += 1

            print(f"{event_id:<8} {true_label:<12} {'anomaly' if detected else 'normal':<12} "
                  f"{symbol:<10} {anomaly_ratio:<12.1%} {mean_prob:<12.4f}")
            results.append({
                "event_id": event_id, "true_label": true_label,
                "detected": detected, "result_type": res,
                "anomaly_ratio": anomaly_ratio, "mean_prob": mean_prob,
            })
        except Exception as e:
            print(f"  ERR {fname}: {e}")

    total = TP + FP + TN + FN
    acc   = (TP + TN) / total if total > 0 else 0
    prec  = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec   = TP / (TP + FN) if (TP + FN) > 0 else 0
    far   = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print("=" * 100)
    print(f"\nCONFUSION MATRIX:")
    print(f"  Actual Anomaly  | TP={TP:3d} | FN={FN:3d}")
    print(f"  Actual Normal   | FP={FP:3d} | TN={TN:3d}")
    print(f"\nMETRICS:")
    print(f"  Accuracy:         {acc:.2%}")
    print(f"  Detection Rate:   {rec:.2%}")
    print(f"  False Alarm Rate: {far:.2%}")
    print(f"  Precision:        {prec:.2%}")
    print(f"  F1 Score:         {f1:.4f}")

    return {
        "accuracy": acc, "recall": rec, "precision": prec, "far": far, "f1": f1,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "events": results,
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(model_results: dict, output_dir: str) -> None:
    """
    Print a side-by-side comparison table and save a bar chart
    comparing XGBoost vs Random Forest vs LSTM (if LSTM results exist).
    """
    # Try to load LSTM results for comparison
    lstm_path = os.path.join(RESULTS_DIR, "7day", "lstm_test_evaluation.json")
    baseline_path = os.path.join(RESULTS_DIR, "baselines", "naive_test_evaluation.json")

    all_results = {}
    for model_key, res in model_results.items():
        all_results[model_key] = res

    if os.path.exists(lstm_path):
        with open(lstm_path) as f:
            lstm_data = json.load(f)
        all_results["LSTM"] = lstm_data.get("metrics", {})
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            naive_data = json.load(f)
        all_results["Naive"] = naive_data.get("metrics", {})

    metrics = ["accuracy", "recall", "precision", "far", "f1"]
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    header = f"{'Metric':<22}" + "".join(f"{m:^18}" for m in all_results)
    print(header)
    print("-" * 90)
    for metric in metrics:
        row = f"{metric:<22}"
        for m_res in all_results.values():
            val = m_res.get(metric, m_res.get("false_alarm_rate", 0) if metric == "far" else 0)
            row += f"{val:^18.2%}" if isinstance(val, float) else f"{'N/A':^18}"
        print(row)

    # Bar chart
    try:
        n_models = len(all_results)
        n_metrics = 4  # accuracy, recall, precision, f1
        display_metrics = ["recall", "precision", "far", "f1"]
        labels = display_metrics
        x = np.arange(len(labels))
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (model_key, res) in enumerate(all_results.items()):
            vals = []
            for m in display_metrics:
                v = res.get(m, res.get("false_alarm_rate", 0) if m == "far" else 0)
                vals.append(float(v) if v is not None else 0.0)
            ax.bar(x + i * width, vals, width, label=model_key, alpha=0.85)

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(["Detection Rate", "Precision", "False Alarm Rate", "F1"], fontsize=11)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison — SCADA Fault Detection", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nComparison chart saved to: {path}")
    except Exception as e:
        print(f"[WARN] Could not generate comparison chart: {e}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    print("=" * 100)
    print("Tree Model Evaluation — SCADA Fault Detection")
    print("=" * 100)

    results_dir = os.path.join(RESULTS_DIR, "7day")
    os.makedirs(results_dir, exist_ok=True)

    models_to_run = (
        ["xgboost", "random_forest"] if args.model == "both"
        else [args.model]
    )

    model_keys = {"xgboost": "xgboost_7day.pkl", "random_forest": "random_forest_7day.pkl"}
    all_event_results = {}

    for model_key in models_to_run:
        model_path = os.path.join(MODELS_DIR, model_keys[model_key])
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_key}: model not found at {model_path}")
            print(f"  Run: python -m src.training.scripts.train_{model_key}")
            continue

        saved = joblib.load(model_path)
        model = saved["model"]
        threshold = saved["threshold"]
        feature_mode = saved.get("feature_mode", "raw_flatten")
        use_stats = feature_mode == "statistical"

        print(f"\nLoaded {model_key}: threshold={threshold:.4f}, features={feature_mode}")

        model_name = "XGBoost" if "xgboost" in model_key else "Random Forest"
        event_res = evaluate_event_level(model, threshold, feature_mode, model_name)
        all_event_results[model_name] = event_res

        # Save per-model JSON
        out = {
            "model": model_name,
            "threshold": threshold,
            "feature_mode": feature_mode,
            "metrics": {k: event_res[k] for k in ("accuracy", "recall", "precision", "far", "f1",
                                                   "TP", "FP", "TN", "FN")},
            "events": event_res["events"],
        }
        fname = f"{'xgboost' if 'XGB' in model_name else 'random_forest'}_test_evaluation.json"
        with open(os.path.join(results_dir, fname), "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {os.path.join(results_dir, fname)}")

    if len(all_event_results) > 0:
        compare_models(all_event_results, results_dir)


if __name__ == "__main__":
    main()
