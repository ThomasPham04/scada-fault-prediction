"""Evaluation helpers for retained Random Forest and XGBoost baselines."""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from config import MODELS_DIR, PER_ASSET_PROCESSED_DIR, RESULTS_DIR, WIND_FARM_A_PROCESSED
from training.hyperparameters.tree_defaults import (
    DEFAULT_ADAPTIVE_STD_MULTIPLIER,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_EVENT_SCORE_MULTIPLIER,
    DEFAULT_MIN_OUTLIER_RATIO,
    DEFAULT_PER_ASSET_PERCENTILE,
)


class TreeEvaluator:
    """Evaluate XGBoost or Random Forest models on test events."""

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        per_asset_dir: str = PER_ASSET_PROCESSED_DIR,
    ) -> None:
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.per_asset_dir = per_asset_dir

    def evaluate_event_level(
        self,
        model,
        threshold: float,
        feature_mode: str,
        model_name: str,
    ) -> dict:
        """Classify each test event using a retained tree baseline."""
        from data_pipeline.loaders.tabular_loader import TabularLoader

        split_dir = os.path.join(WIND_FARM_A_PROCESSED, "global", "test_by_event")
        loader = TabularLoader(use_stats=(feature_mode == "statistical"))

        tp = fp = tn = fn = 0
        results = []

        print(f"\n{'=' * 100}")
        print(f"{model_name} - Event-Level Detection (threshold={threshold:.4f})")
        print(f"{'=' * 100}")
        print(
            f"{'Event':<8} {'True':<12} {'Predicted':<12} {'Result':<10} "
            f"{'Anomaly %':<12} {'Mean Prob':<12}"
        )
        print("-" * 100)

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".npz"):
                continue
            try:
                event_id = int(fname.split("_")[1].split(".")[0])
                raw = np.load(os.path.join(split_dir, fname), allow_pickle=True)
                x_values = raw["X"]
                true_label = str(raw["label"])

                x_flat = loader._transform(x_values) if x_values.ndim == 3 else x_values.astype(np.float32)
                proba = model.predict_proba(x_flat)[:, 1]
                mean_prob = float(proba.mean())
                anomaly_ratio = float((proba >= threshold).mean())
                detected = anomaly_ratio >= DEFAULT_MIN_OUTLIER_RATIO
                is_anomaly = true_label == "anomaly"

                if is_anomaly and detected:
                    result_type, symbol = "TP", "[OK] TP"
                    tp += 1
                elif not is_anomaly and not detected:
                    result_type, symbol = "TN", "[OK] TN"
                    tn += 1
                elif not is_anomaly and detected:
                    result_type, symbol = "FP", "[X]  FP"
                    fp += 1
                else:
                    result_type, symbol = "FN", "[X]  FN"
                    fn += 1

                print(
                    f"{event_id:<8} {true_label:<12} "
                    f"{'anomaly' if detected else 'normal':<12} {symbol:<10} "
                    f"{anomaly_ratio:<12.1%} {mean_prob:<12.4f}"
                )
                results.append(
                    {
                        "event_id": event_id,
                        "true_label": true_label,
                        "detected": detected,
                        "result_type": result_type,
                        "anomaly_ratio": anomaly_ratio,
                        "mean_prob": mean_prob,
                    }
                )
            except Exception as exc:
                print(f"  ERR {fname}: {exc}")

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print("=" * 100)
        print(f"\nCONFUSION MATRIX:  TP={tp} | FN={fn} | FP={fp} | TN={tn}")
        print(
            f"\nMETRICS:  Accuracy={accuracy:.2%}  Recall={recall:.2%}  "
            f"Precision={precision:.2%}  FAR={far:.2%}  F1={f1:.4f}"
        )

        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "far": far,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "events": results,
        }

    def evaluate_per_asset(self, model_name: str, use_stats: bool = False, use_adaptive: bool = False) -> None:
        """Evaluate retained per-asset tree models."""
        import joblib

        from data_pipeline.loaders.tabular_loader import TabularLoader

        loader = TabularLoader(use_stats=use_stats)
        transform = loader.compute_statistical_features if use_stats else loader.flatten_sequences
        prefix = model_name

        print("=" * 80)
        print(f"Per-Asset {prefix.upper()} Evaluation")
        print("=" * 80)

        asset_dirs = sorted(glob.glob(os.path.join(self.per_asset_dir, "asset_*")))
        if not asset_dirs:
            print(f"[ERROR] No per-asset data found at: {self.per_asset_dir}")
            return

        print(f"\nEvaluating {prefix.upper()} with adaptive={use_adaptive}")
        tp = fp = tn = fn = 0
        all_results = []
        for asset_dir in asset_dirs:
            asset_id = os.path.basename(asset_dir).replace("asset_", "")
            model_path = os.path.join(self.models_dir, f"{prefix}_asset_{asset_id}.pkl")
            if not os.path.exists(model_path):
                print(f"  [SKIP] Asset {asset_id}: model not found")
                continue
            bundle = joblib.load(model_path)
            model = bundle["model"]
            threshold = bundle.get("threshold", DEFAULT_DECISION_THRESHOLD)
            print(f"\n  Asset {asset_id} - model loaded (threshold={threshold:.4f})")

            test_dir = os.path.join(asset_dir, "test_by_event")
            event_files = sorted(glob.glob(os.path.join(test_dir, "event_*.npz")))
            for event_file in event_files:
                event_id = os.path.basename(event_file).replace("event_", "").replace(".npz", "")
                event_data = np.load(event_file, allow_pickle=True)
                x_sequences = event_data["X"]
                label = str(event_data["label"])
                y_true = 1 if label == "anomaly" else 0

                if len(x_sequences) == 0:
                    continue

                proba = model.predict_proba(transform(x_sequences))[:, 1]
                current_threshold = threshold
                if use_adaptive:
                    current_threshold += DEFAULT_ADAPTIVE_STD_MULTIPLIER * np.std(proba)

                event_score = float(np.percentile(proba, DEFAULT_PER_ASSET_PERCENTILE))
                outlier_ratio = float(np.mean(proba > current_threshold))
                event_pred = bool(
                    (outlier_ratio >= DEFAULT_MIN_OUTLIER_RATIO)
                    or (event_score > threshold * DEFAULT_EVENT_SCORE_MULTIPLIER)
                )
                correct = bool(event_pred == y_true)

                if y_true == 1 and event_pred:
                    tp += 1
                elif y_true == 1 and not event_pred:
                    fn += 1
                elif y_true == 0 and event_pred:
                    fp += 1
                else:
                    tn += 1

                print(
                    f"    Event {event_id:>4s} ({label:7s}): "
                    f"pred={'anomaly' if event_pred else 'normal ':7s}  "
                    f"{'OK' if correct else 'MISS'}  "
                    f"outlier_rate={outlier_ratio:.2%}"
                )
                all_results.append(
                    {
                        "asset_id": asset_id,
                        "event_id": event_id,
                        "label": label,
                        "pred": "anomaly" if event_pred else "normal",
                        "correct": correct,
                        "mean_proba": float(proba.mean()),
                    }
                )

        if not all_results:
            print("\nNo events evaluated.")
            return

        results_dir = os.path.join(self.results_dir, "per_asset")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{prefix}_per_asset_eval.json")
        with open(out_path, "w") as output_file:
            json.dump(all_results, output_file, indent=2)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

        print("\n" + "=" * 50)
        print(f"{prefix.upper()} PER-ASSET SUMMARY")
        print("=" * 50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f} ({tp + tn}/{tp + fp + tn + fn})")
        print("=" * 50)
