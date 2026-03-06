"""
Evaluate Tree Models — evaluation.evaluate_tree
Per-asset XGBoost / Random Forest evaluation.
Loads each per-turbine model and evaluates it on its own test events.

Usage:
    python -m src.evaluation.evaluate_tree --model xgboost
    python -m src.evaluation.evaluate_tree --model random_forest
    python -m src.evaluation.evaluate_tree --model both
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # → src/
sys.path.insert(0, ROOT)

from evaluation.evaluator import TreeEvaluator


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate per-asset tree models on SCADA test events.")
    ap.add_argument("--model", type=str, default="both",
                    choices=["xgboost", "random_forest", "both"])
    ap.add_argument("--use_stats", action="store_true",
                    help="Use statistical feature mode (must match training mode).")
    return ap.parse_args()


def main() -> None:
    args      = parse_args()
    evaluator = TreeEvaluator()

    models_to_run = (
        ["xgboost", "random_forest"] if args.model == "both" else [args.model]
    )
    for mk in models_to_run:
        evaluator.evaluate_per_asset(mk, use_stats=args.use_stats)


if __name__ == "__main__":
    main()
