"""
Evaluator — evaluation.evaluator
LSTMEvaluator and TreeEvaluator classes that absorb the evaluation logic
previously scattered across evaluate_lstm.py and evaluate_tree.py.

Entry-point scripts are refactored to thin wrappers that instantiate these classes.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # → src/
sys.path.insert(0, ROOT)

from config import (
    MODELS_DIR,
    RESULTS_DIR,
    WIND_FARM_A_PROCESSED,
    PER_ASSET_PROCESSED_DIR,
)
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer


# ===========================================================================
# LSTM Evaluator
# ===========================================================================

class LSTMEvaluator:
    """
    Full evaluation of the LSTM anomaly detector.

    Supports both the global (7-day) model and per-asset per-turbine models.

    Args:
        models_dir: Directory containing trained .keras files.
        results_dir: Root directory for output JSON / plots.
        per_asset_dir: Root directory for per-asset NPZ data.
    """

    def __init__(
        self,
        models_dir: str  = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        per_asset_dir: str = PER_ASSET_PROCESSED_DIR,
    ) -> None:
        self.models_dir   = models_dir
        self.results_dir  = results_dir
        self.per_asset_dir = per_asset_dir
        self._vis         = Visualizer()

    # ------------------------------------------------------------------
    # Train / Val error (used during training)
    # ------------------------------------------------------------------

    def evaluate_train_val(self, model, X_train, y_train, X_val, y_val) -> dict:
        """
        Compute MSE and MAE per sample for train and val splits.

        Returns:
            Dict with 'train' and 'val' error stats.
        """
        print("\nEvaluating model on train/val splits...")
        errors = {}
        for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
            y_pred = model.predict(X, verbose=0)
            mse_ps = np.mean((y - y_pred) ** 2, axis=1)
            mae_ps = np.mean(np.abs(y - y_pred), axis=1)
            errors[name] = {
                "mse_per_sample": mse_ps,
                "mae_per_sample": mae_ps,
                "mean_mse": float(np.mean(mse_ps)),
                "std_mse":  float(np.std(mse_ps)),
                "mean_mae": float(np.mean(mae_ps)),
                "std_mae":  float(np.std(mae_ps)),
            }
            print(
                f"  {name.capitalize()}: "
                f"MSE={errors[name]['mean_mse']:.6f} ± {errors[name]['std_mse']:.6f}  |  "
                f"MAE={errors[name]['mean_mae']:.6f} ± {errors[name]['std_mae']:.6f}"
            )
        return errors

    def save_results(self, errors: dict, metadata: dict, output_dir: str) -> None:
        """Save train/val error stats and raw arrays to disk."""
        results = {
            "version": 2,
            "model_type": "LSTM_Prediction_V2",
            "window_size": metadata["window_size"],
            "n_features": metadata["n_features"],
            "split_strategy": metadata["split_strategy"],
            "train_error_stats": {k: errors["train"][k] for k in ("mean_mse", "std_mse", "mean_mae", "std_mae")},
            "val_error_stats":   {k: errors["val"][k]   for k in ("mean_mse", "std_mse", "mean_mae", "std_mae")},
        }
        path = os.path.join(output_dir, "lstm_global_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {path}")

        np.savez(
            os.path.join(output_dir, "lstm_global_errors.npz"),
            train_mse=errors["train"]["mse_per_sample"],
            val_mse=errors["val"]["mse_per_sample"],
            train_mae=errors["train"]["mae_per_sample"],
            val_mae=errors["val"]["mae_per_sample"],
        )

    # ------------------------------------------------------------------
    # Test evaluation (global model)
    # ------------------------------------------------------------------

    @staticmethod
    def _longest_run(mask: np.ndarray) -> int:
        """Return the longest consecutive True run."""
        max_run = run = 0
        for v in mask:
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return max_run

    def evaluate_test(
        self,
        model,
        test_events: dict,
        upper_threshold: float,
        lower_threshold: float,
        min_recall: float = 0.7,
        smoothing_window: int = 3,
    ) -> dict:
        """
        Evaluate the LSTM on test events using the adaptive hybrid threshold.

        Args:
            model: Trained Keras model.
            test_events: Event dict from load_events('test').
            upper_threshold: Global upper threshold.
            lower_threshold: Lower threshold (usually 0).
            min_recall: Recall constraint (informational).
            smoothing_window: MAE smoothing window size.

        Returns:
            Dict with accuracy, recall, precision, far, f1.
        """
        from training.experiments.threshold_tuning import smooth_mae

        print(f"\nEvaluating with thresholds: upper={upper_threshold:.6f}, lower={lower_threshold:.6f}")
        TP = FP = TN = FN = 0
        results, plot_records = [], []

        print("\n" + "=" * 140)
        print(
            f"{'Event':<8} {'True Label':<12} {'Predicted':<12} {'Result':<10} "
            f"{'Mean MAE':<12} {'p95 MAE':<12} {'Max MAE':<12} {'Samples':<10}"
        )
        print("-" * 140)

        for event_id, data in sorted(test_events.items()):
            X, y       = data["X"], data["y"]
            true_label = data["label"]

            y_pred = model.predict(X, verbose=0, batch_size=256)
            mae    = np.mean(np.abs(y - y_pred), axis=1)
            mae    = smooth_mae(mae, window=smoothing_window)

            event_std        = np.std(mae)
            adaptive_th      = upper_threshold + 0.45 * event_std
            event_p90        = np.percentile(mae, 90)
            event_p95        = np.percentile(mae, 95)
            severity_score   = 0.6 * event_p95 + 0.4 * event_p90
            outlier_ratio    = np.mean(mae > adaptive_th)
            min_ratio        = max(0.15, 5 / len(mae))
            above_mask       = mae > adaptive_th
            longest_run      = self._longest_run(above_mask)
            min_run          = max(2, int(0.2 * len(mae)))
            VERY_HIGH        = severity_score > (adaptive_th * 1.3)
            SHORT_EVENT      = len(mae) < 18

            if SHORT_EVENT:
                detected = severity_score > (upper_threshold * 0.9) or VERY_HIGH
            else:
                detected = (
                    (longest_run >= min_run and
                     (severity_score > adaptive_th or outlier_ratio >= min_ratio))
                    or VERY_HIGH
                )

            mean_mae  = float(np.mean(mae))
            max_mae   = float(np.max(mae))
            is_anomaly = (true_label == "anomaly")

            if is_anomaly and detected:
                res_type, symbol = "TP", "[OK] TP"; TP += 1
            elif not is_anomaly and not detected:
                res_type, symbol = "TN", "[OK] TN"; TN += 1
            elif not is_anomaly and detected:
                res_type, symbol = "FP", "[X]  FP"; FP += 1
            else:
                res_type, symbol = "FN", "[X]  FN"; FN += 1

            print(
                f"{event_id:<8} {true_label:<12} "
                f"{'anomaly' if detected else 'normal':<12} {symbol:<10} "
                f"{mean_mae:<12.4f} {event_p95:<12.4f} {max_mae:<12.4f} {len(mae):<10}"
            )

            results.append({
                "event_id": int(event_id), "true_label": true_label,
                "detected": bool(detected), "result_type": res_type,
                "mae_mean": mean_mae, "mae_p95": float(event_p95),
                "mae_max": max_mae, "n_samples": int(len(mae)),
            })
            plot_records.append({
                "event_id": event_id, "true_label": true_label,
                "detected": detected, "p95": float(event_p95),
                "mean": mean_mae, "adaptive_threshold": float(adaptive_th),
            })

        total     = TP + FP + TN + FN
        accuracy  = (TP + TN) / total if total > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        far       = FP / (FP + TN) if (FP + TN) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print("=" * 140)
        print(f"\nCONFUSION MATRIX:")
        print(f"  Actual Anomaly  | TP={TP:3d} | FN={FN:3d}")
        print(f"  Actual Normal   | FP={FP:3d} | TN={TN:3d}")
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Accuracy:         {accuracy:.2%}")
        print(f"  Detection Rate:   {recall:.2%}")
        print(f"  False Alarm Rate: {far:.2%}")
        print(f"  Precision:        {precision:.2%}")
        print(f"  F1 Score:         {f1:.4f}")

        output_path = os.path.join(self.results_dir, "global", "lstm_test_evaluation.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "model": "LSTM_global", "upper_threshold": float(upper_threshold),
                "metrics": {"accuracy": accuracy, "recall": recall,
                            "precision": precision, "far": far, "f1": f1},
                "confusion_matrix": {"TP": TP, "FN": FN, "FP": FP, "TN": TN},
                "events": results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        self._vis.plot_event_scores(plot_records, upper_threshold)

        return {"accuracy": accuracy, "recall": recall, "precision": precision, "far": far, "f1": f1}

    # ------------------------------------------------------------------
    # Per-asset evaluation
    # ------------------------------------------------------------------

    def evaluate_per_asset(self, asset_filter: list = None, use_adaptive: bool = False) -> None:
        """
        Per-asset evaluation: load each asset's LSTM and report per-event MAE.

        Args:
            asset_filter: Optional list of asset IDs to evaluate. If None, evaluates all.
        """
        import pandas as pd
        from tensorflow import keras
        
        global_anomaly_scores = []
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        print("=" * 100)
        print("LSTM — Per-Asset Test Set Evaluation")
        print("=" * 100)

        asset_dirs = sorted(glob.glob(os.path.join(self.per_asset_dir, "asset_*")))
        if not asset_dirs:
            print(f"[ERROR] No per-asset data found at: {self.per_asset_dir}")
            return

        print(f"\nEvaluating with adaptive={use_adaptive}")
        TP = FP = TN = FN = 0

        # Apply asset filter if specified
        if asset_filter is not None:
            filter_set = {str(a) for a in asset_filter}
            asset_dirs = [d for d in asset_dirs
                          if os.path.basename(d).replace("asset_", "") in filter_set]
            if not asset_dirs:
                print(f"[ERROR] No data found for requested assets: {asset_filter}")
                return
            print(f"Evaluating only assets: {asset_filter}")

        from training.experiments.threshold_tuning import smooth_mae

        all_results = []
        for asset_dir in asset_dirs:
            asset_id   = os.path.basename(asset_dir).replace("asset_", "")
            model_path = os.path.join(self.models_dir, f"lstm_asset_{asset_id}.keras")

            if not os.path.exists(model_path):
                print(f"\n  [SKIP] Asset {asset_id}: model not found at {model_path}")
                continue

            model = keras.models.load_model(model_path)
            print(f"\n  Asset {asset_id} — model loaded")

            # 1. Compute Threshold from Validation Data (95th percentile)
            val_x_path = os.path.join(asset_dir, "X_val.npy")
            val_y_path = os.path.join(asset_dir, "y_val.npy")
            
            if not os.path.exists(val_x_path) or not os.path.exists(val_y_path):
                print(f"  [WARN] Missing validation data for threshold tuning in {asset_dir}")
                threshold = 0.5  # fallback
            else:
                X_val = np.load(val_x_path)
                y_val = np.load(val_y_path)
                if len(X_val) > 0:
                    val_pred = model.predict(X_val, verbose=0, batch_size=128)
                    val_mae = np.mean(np.abs(y_val - val_pred), axis=1)
                    val_mae = smooth_mae(val_mae, window=3)
                    threshold = np.percentile(val_mae, 85)
                else:
                    threshold = 0.5
            
            print(f"  Asset {asset_id} — Auto Threshold (val p85) = {threshold:.5f}")

            # 2. Evaluate on Test Events
            test_dir    = os.path.join(asset_dir, "test_by_event")
            event_files = sorted(glob.glob(os.path.join(test_dir, "event_*.npz")))

            for ef in event_files:
                event_id = os.path.basename(ef).replace("event_", "").replace(".npz", "")
                npz      = np.load(ef, allow_pickle=True)
                X, y     = npz["X"], npz["y"]
                label    = str(npz["label"])

                if len(X) == 0:
                    continue

                ts_arr = npz.get("time_stamps", [])

                y_pred   = model.predict(X, verbose=0, batch_size=256)
                raw_mae  = np.mean(np.abs(y - y_pred), axis=1)
                mae      = smooth_mae(raw_mae, window=3)
                
                # Collect scores for exact CARE to Compare reproduction
                for i in range(len(mae)):
                    timestamp = ts_arr[i] if i < len(ts_arr) else f"unknown_evt{event_id}_{i}"
                    global_anomaly_scores.append((timestamp, float(mae[i])))

                mean_mae  = float(np.mean(mae))
                event_p85 = float(np.percentile(mae, 85))
                
                # Decision logic
                current_threshold = threshold
                if use_adaptive:
                    current_threshold += 0.45 * np.std(mae)

                outlier_ratio = float(np.mean(mae > current_threshold))
                detected      = bool((outlier_ratio >= 0.15) or (event_p85 > threshold * 1.2))
                
                is_anomaly = (label == "anomaly")
                correct    = bool(detected == is_anomaly)

                if is_anomaly and detected: TP += 1
                elif is_anomaly and not detected: FN += 1
                elif not is_anomaly and detected: FP += 1
                else: TN += 1

                print(f"    Event {event_id:>4s} ({label:7s}): "
                      f"pred={'anomaly' if detected else 'normal ':7s}  "
                      f"{'OK' if correct else 'MISS'}  "
                      f"outlier_rate={outlier_ratio:.1%}  "
                      f"mean_MAE={mean_mae:.5f}  p85_MAE={event_p85:.5f}")
                
                all_results.append({
                    "asset_id": asset_id, "event_id": event_id,
                    "label": label, "n_samples": len(X),
                    "mean_mae": mean_mae, "p85_mae": event_p85,
                    "threshold": float(current_threshold),
                    "outlier_ratio": outlier_ratio,
                    "detected": bool(detected),
                    "correct": correct
                })

        if not all_results:
            print("\nNo events evaluated.")
            return

        results_dir = os.path.join(self.results_dir, "per_asset")
        os.makedirs(results_dir, exist_ok=True)
        out_path     = os.path.join(results_dir, "lstm_per_asset_eval.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nPer-asset results saved: {out_path}")

        if global_anomaly_scores:
            df_scores = pd.DataFrame(global_anomaly_scores, columns=["time_stamp", "0"])
            # Save into the root results tracking folder requested by user
            out_csv = os.path.join(self.results_dir, "anomaly_scores.csv")
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            df_scores.to_csv(out_csv, index=False)
            print(f"Saved global anomaly scores: {out_csv}")

        anomaly_mae = [r["mean_mae"] for r in all_results if r["label"] == "anomaly"]
        normal_mae  = [r["mean_mae"] for r in all_results if r["label"] == "normal"]
        if anomaly_mae and normal_mae:
            print(f"Avg MAE  anomaly={np.mean(anomaly_mae):.5f}  normal={np.mean(normal_mae):.5f}")

        # Summary Metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        
        print("\n" + "=" * 50)
        print("LSTM PER-ASSET SUMMARY")
        print("=" * 50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f} ({TP+TN}/{TP+FP+TN+FN})")
        print("=" * 50)


# ===========================================================================
# Tree Evaluator  (XGBoost & Random Forest)
# ===========================================================================

class TreeEvaluator:
    """
    Evaluates XGBoost or Random Forest models on test events.

    Args:
        models_dir: Directory containing trained .pkl bundles.
        results_dir: Root directory for output JSON / plots.
        per_asset_dir: Root directory for per-asset NPZ data.
    """

    def __init__(
        self,
        models_dir: str  = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        per_asset_dir: str = PER_ASSET_PROCESSED_DIR,
    ) -> None:
        self.models_dir    = models_dir
        self.results_dir   = results_dir
        self.per_asset_dir = per_asset_dir
        self._metrics      = MetricsCalculator()
        self._vis          = Visualizer()

    def evaluate_event_level(
        self,
        model,
        threshold: float,
        feature_mode: str,
        model_name: str,
    ) -> dict:
        """
        Event-level detection: classify each event as anomaly/normal.

        An event is flagged anomaly if >15% of its sample predictions
        exceed the threshold (low bar to improve recall).

        Args:
            model: Fitted sklearn/XGBoost classifier.
            threshold: Classification threshold.
            feature_mode: 'statistical' or 'raw_flatten'.
            model_name: Human-readable model name for printing.

        Returns:
            Dict with TP, FP, TN, FN, metrics, and per-event results.
        """
        from data_pipeline.loaders.tabular_loader import TabularLoader

        data_dir  = os.path.join(WIND_FARM_A_PROCESSED, "global")
        split_dir = os.path.join(data_dir, "test_by_event")
        loader    = TabularLoader(use_stats=(feature_mode == "statistical"))

        TP = FP = TN = FN = 0
        results = []

        print(f"\n{'='*100}")
        print(f"{model_name} — Event-Level Detection (threshold={threshold:.4f})")
        print(f"{'='*100}")
        print(
            f"{'Event':<8} {'True':<12} {'Predicted':<12} {'Result':<10} "
            f"{'Anomaly %':<12} {'Mean Prob':<12}"
        )
        print("-" * 100)

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".npz"):
                continue
            try:
                event_id   = int(fname.split("_")[1].split(".")[0])
                raw        = np.load(os.path.join(split_dir, fname), allow_pickle=True)
                X          = raw["X"]
                true_label = str(raw["label"])

                X_flat = loader._transform(X) if X.ndim == 3 else X.astype(np.float32)
                proba  = model.predict_proba(X_flat)[:, 1]
                mean_prob     = float(proba.mean())
                anomaly_ratio = float((proba >= threshold).mean())
                detected      = anomaly_ratio >= 0.15
                is_anomaly    = true_label == "anomaly"

                if is_anomaly and detected:
                    res, symbol = "TP", "[OK] TP"; TP += 1
                elif not is_anomaly and not detected:
                    res, symbol = "TN", "[OK] TN"; TN += 1
                elif not is_anomaly and detected:
                    res, symbol = "FP", "[X]  FP"; FP += 1
                else:
                    res, symbol = "FN", "[X]  FN"; FN += 1

                print(
                    f"{event_id:<8} {true_label:<12} "
                    f"{'anomaly' if detected else 'normal':<12} {symbol:<10} "
                    f"{anomaly_ratio:<12.1%} {mean_prob:<12.4f}"
                )
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
        print(f"\nCONFUSION MATRIX:  TP={TP} | FN={FN} | FP={FP} | TN={TN}")
        print(f"\nMETRICS:  Accuracy={acc:.2%}  Recall={rec:.2%}  "
              f"Precision={prec:.2%}  FAR={far:.2%}  F1={f1:.4f}")

        return {
            "accuracy": acc, "recall": rec, "precision": prec, "far": far, "f1": f1,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN, "events": results,
        }

    def evaluate_per_asset(self, model_name: str, use_stats: bool = False, use_adaptive: bool = False) -> None:
        """
        Per-asset tree model evaluation.

        Args:
            model_name: 'xgboost' or 'random_forest'.
            use_stats: Whether the model was trained with statistical features.
        """
        import joblib
        from data_pipeline.loaders.tabular_loader import TabularLoader

        loader = TabularLoader(use_stats=use_stats)
        fn     = loader.compute_statistical_features if use_stats else loader.flatten_sequences
        prefix = model_name

        print("=" * 80)
        print(f"Per-Asset {prefix.upper()} Evaluation")
        print("=" * 80)

        asset_dirs = sorted(glob.glob(os.path.join(self.per_asset_dir, "asset_*")))
        if not asset_dirs:
            print(f"[ERROR] No per-asset data found at: {self.per_asset_dir}")
            return

        print(f"\nEvaluating {prefix.upper()} with adaptive={use_adaptive}")
        TP = FP = TN = FN = 0
        all_results = []
        for asset_dir in asset_dirs:
            asset_id   = os.path.basename(asset_dir).replace("asset_", "")
            model_path = os.path.join(self.models_dir, f"{prefix}_asset_{asset_id}.pkl")
            if not os.path.exists(model_path):
                print(f"  [SKIP] Asset {asset_id}: model not found")
                continue
            bundle    = joblib.load(model_path)
            model     = bundle["model"]
            threshold = bundle.get("threshold", 0.5)
            print(f"\n  Asset {asset_id} — model loaded (threshold={threshold:.4f})")

            test_dir    = os.path.join(asset_dir, "test_by_event")
            event_files = sorted(glob.glob(os.path.join(test_dir, "event_*.npz")))
            for ef in event_files:
                event_id   = os.path.basename(ef).replace("event_", "").replace(".npz", "")
                npz        = np.load(ef, allow_pickle=True)
                X_seq      = npz["X"]
                label      = str(npz["label"])
                y_true_bin = 1 if label == "anomaly" else 0

                if len(X_seq) == 0:
                    continue

                X_flat    = fn(X_seq)
                proba     = model.predict_proba(X_flat)[:, 1]
                
                # Decision logic
                current_threshold = threshold
                if use_adaptive:
                    current_threshold += 0.45 * np.std(proba)

                event_p85     = float(np.percentile(proba, 85))
                outlier_ratio = float(np.mean(proba > current_threshold))
                
                # Using same hybrid trigger as LSTM for consistency
                event_pred = bool((outlier_ratio >= 0.15) or (event_p85 > threshold * 1.2))
                correct    = bool(event_pred == y_true_bin)

                if y_true_bin == 1 and event_pred: TP += 1
                elif y_true_bin == 1 and not event_pred: FN += 1
                elif y_true_bin == 0 and event_pred: FP += 1
                else: TN += 1

                print(
                    f"    Event {event_id:>4s} ({label:7s}): "
                    f"pred={'anomaly' if event_pred else 'normal ':7s}  "
                    f"{'OK' if correct else 'MISS'}  "
                    f"outlier_rate={outlier_ratio:.2%}"
                )
                all_results.append({
                    "asset_id": asset_id, "event_id": event_id, "label": label,
                    "pred": "anomaly" if event_pred else "normal",
                    "correct": correct, "mean_proba": float(proba.mean()),
                })

        if not all_results:
            print("\nNo events evaluated.")
            return

        results_dir = os.path.join(self.results_dir, "per_asset")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{prefix}_per_asset_eval.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
            
        # Summary Metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0

        print("\n" + "=" * 50)
        print(f"{prefix.upper()} PER-ASSET SUMMARY")
        print("=" * 50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f} ({TP+TN}/{TP+FP+TN+FN})")
        print("=" * 50)
        print(
            f"Event-level accuracy: {accuracy:.2%}"
        )

    def compare_models(self, model_results: dict) -> None:
        """
        Print a comparison table and save a bar chart for all models.

        Args:
            model_results: Dict {model_name: metrics_dict}.
        """
        import json

        all_results = dict(model_results)

        lstm_path     = os.path.join(self.results_dir, "global", "lstm_test_evaluation.json")
        baseline_path = os.path.join(self.results_dir, "baselines", "naive_test_evaluation.json")

        if os.path.exists(lstm_path):
            with open(lstm_path) as f:
                all_results["LSTM"] = json.load(f).get("metrics", {})
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                all_results["Naive"] = json.load(f).get("metrics", {})

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

        results_dir = os.path.join(self.results_dir, "global")
        os.makedirs(results_dir, exist_ok=True)
        self._vis.compare_models_chart(all_results, results_dir)
