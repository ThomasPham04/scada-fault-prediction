"""
Evaluate LSTM — evaluation.evaluate_lstm
Full test-set evaluation for the LSTM anomaly detector.

Modes:
  --per_asset            Evaluate per-asset LSTM models (one per turbine)
  --per_asset --assets 0 Evaluate only specific assets
  (default)              Evaluate the global LSTM (lstm_7day.keras)

Usage:
    python -m src.evaluation.evaluate_lstm
    python -m src.evaluation.evaluate_lstm --per_asset
    python -m src.evaluation.evaluate_lstm --per_asset --assets 0 10
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # → src/
sys.path.insert(0, ROOT)

from evaluation.evaluator import LSTMEvaluator


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate LSTM SCADA model.")
    ap.add_argument("--per_asset", action="store_true",
                    help="Evaluate per-asset LSTM models instead of the global one.")
    ap.add_argument("--assets", type=int, nargs="+", default=None,
                    metavar="ID",
                    help="Asset IDs to evaluate (only with --per_asset). "
                         "If omitted, evaluates all assets.")
    args = ap.parse_args()

    evaluator = LSTMEvaluator()

    if args.per_asset:
        evaluator.evaluate_per_asset(asset_filter=args.assets)
    else:
        # Global LSTM path
        from tensorflow import keras
        from config import MODELS_DIR
        from training.experiments.threshold_tuning import load_events, determine_threshold

        print("=" * 100)
        print("LSTM (global) — Final Test Set Evaluation")
        print("=" * 100)
        model_path = os.path.join(MODELS_DIR, "lstm_7day.keras")
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Run: python -m src.training.scripts.train_lstm")
            return
        model       = keras.models.load_model(model_path)
        train_events = load_events("train")
        val_events   = load_events("val")
        test_events  = load_events("test")
        if not val_events or not test_events:
            print("Error: Missing val or test event files.")
            return
        upper_th, lower_th = determine_threshold(model, val_events, train_events, min_recall=0.7)
        evaluator.evaluate_test(model, test_events, upper_th, lower_th)


if __name__ == "__main__":
    main()
