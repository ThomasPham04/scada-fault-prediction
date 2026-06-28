"""Evaluation helpers for retained Random Forest and XGBoost baselines."""

from __future__ import annotations

import os

import numpy as np

from config import MODELS_DIR, RESULTS_DIR, WIND_FARM_A_PROCESSED
from training.hyperparameters.tree_defaults import (
    DEFAULT_MIN_OUTLIER_RATIO,
)


class TreeEvaluator:
    """Evaluate XGBoost or Random Forest models on test events."""

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
    ) -> None:
        self.models_dir = models_dir
        self.results_dir = results_dir

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
