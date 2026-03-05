"""
Train Random Forest — training.scripts.train_random_forest
Entry-point for training per-asset Random Forest classifiers on SCADA data.

Requires: python -m src.training.scripts.prepare_per_asset (run first)

Usage:
    python -m src.training.scripts.train_random_forest
    python -m src.training.scripts.train_random_forest --n_estimators 500
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
    ap = argparse.ArgumentParser(description="Train per-asset Random Forest on SCADA tabular features.")
    ap.add_argument("--n_estimators",      type=int,   default=300)
    ap.add_argument("--max_depth",         type=int,   default=None)
    ap.add_argument("--min_samples_split", type=int,   default=2)
    ap.add_argument("--min_samples_leaf",  type=int,   default=1)
    ap.add_argument("--max_features",      type=str,   default="sqrt",
                    help="Features per split: 'sqrt', 'log2', or a float ratio.")
    ap.add_argument("--class_weight",      type=str,   default="balanced")
    ap.add_argument("--use_stats",         action="store_true",
                    help="Use statistical window features instead of raw flattening.")
    ap.add_argument("--seed",              type=int,   default=RANDOM_SEED)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    ensure_dirs()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Coerce max_features string → numeric if needed
    try:
        mf = float(args.max_features)
        args.max_features = int(mf) if mf >= 1.0 else mf
    except (ValueError, TypeError):
        pass

    print("=" * 70)
    print("Random Forest — Per-Asset Training")
    print("=" * 70)
    TreeTrainer(model_type="random_forest").run_per_asset(args)


if __name__ == "__main__":
    main()
