"""
Train LSTM — training.scripts.train_lstm
Entry-point for training the LSTM per-turbine model.

Requires: python -m src.training.scripts.prepare_per_asset (run first)

Usage:
    python -m src.training.scripts.train_lstm
    python -m src.training.scripts.train_lstm --assets 10 11
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import ensure_dirs, MODELS_DIR
from training.trainer import LSTMTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train per-asset LSTM for SCADA fault detection.")
    ap.add_argument("--assets", type=int, nargs="+", default=None,
                    metavar="ID",
                    help="Asset IDs to train. E.g. --assets 10 11. "
                         "If omitted, trains all assets.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("LSTM — Per-Asset Training Pipeline")
    print("=" * 70)
    LSTMTrainer().run_per_asset(asset_filter=args.assets)


if __name__ == "__main__":
    main()
