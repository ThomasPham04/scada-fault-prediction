"""
Train XGBoost — training.scripts.train_xgboost
Entry-point for training per-asset XGBoost classifiers on SCADA data.

Requires: python -m src.training.scripts.prepare_per_asset (run first)

Usage:
    python -m src.training.scripts.train_xgboost
    python -m src.training.scripts.train_xgboost --use_stats --n_estimators 600
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import MODELS_DIR, RANDOM_SEED, ensure_dirs
from training.trainer import TreeTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train per-asset XGBoost on SCADA tabular features.")
    ap.add_argument("--n_estimators",          type=int,   default=400)
    ap.add_argument("--max_depth",             type=int,   default=6)
    ap.add_argument("--learning_rate",         type=float, default=0.05)
    ap.add_argument("--subsample",             type=float, default=0.8)
    ap.add_argument("--colsample_bytree",      type=float, default=0.8)
    ap.add_argument("--min_child_weight",      type=float, default=1.0)
    ap.add_argument("--gamma",                 type=float, default=0.0)
    ap.add_argument("--reg_lambda",            type=float, default=1.0)
    ap.add_argument("--reg_alpha",             type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int,   default=30)
    ap.add_argument("--tree_method",           type=str,   default="hist")
    ap.add_argument("--use_gpu",               action="store_true")
    ap.add_argument("--use_stats",             action="store_true",
                    help="Use statistical window features instead of raw flattening.")
    ap.add_argument("--seed",                  type=int,   default=RANDOM_SEED)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    ensure_dirs()
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("XGBoost — Per-Asset Training")
    print("=" * 70)
    TreeTrainer(model_type="xgboost").run_per_asset(args)


if __name__ == "__main__":
    main()
