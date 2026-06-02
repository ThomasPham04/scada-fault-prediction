"""
Train sequence classifiers from combined CSV exports.

Requires sequence exports from:
    python src/main.py prepare --csv df_final.csv --feature-file final_features.csv

Usage:
    python -m training.scripts.train_sequence_models --windows 24
    python src/main.py train-sequences --windows 24
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from training.hyperparameters.classifier_defaults import (  # noqa: E402
    CLASSIFIER_LOSSES,
    DEFAULT_CLASSIFIER_BATCH_SIZE,
    DEFAULT_CLASSIFIER_DROPOUT,
    DEFAULT_CLASSIFIER_EPOCHS,
    DEFAULT_CLASSIFIER_FOCAL_ALPHA,
    DEFAULT_CLASSIFIER_FOCAL_GAMMA,
    DEFAULT_CLASSIFIER_L2,
    DEFAULT_CLASSIFIER_LEARNING_RATE,
    DEFAULT_CLASSIFIER_LOSS,
    DEFAULT_CLASSIFIER_MODELS,
    DEFAULT_SEQUENCE_WINDOWS,
)
from training.hyperparameters.sequence_defaults import RANDOM_SEED  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train sequence classifiers from exported classifier NPY bundles."
    )
    ap.add_argument(
        "--exports-dir",
        type=Path,
        default=Path(PROCESSED_DATA_DIR) / "sequence_exports",
        help="Root folder containing window_<H>h sequence exports.",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path(RESULTS_DIR) / "sequence_training_results",
        help="Output folder for trained models, metrics, and plots.",
    )
    ap.add_argument("--windows", type=int, nargs="+", default=DEFAULT_SEQUENCE_WINDOWS)
    ap.add_argument(
        "--classifier-models",
        type=str,
        nargs="+",
        default=DEFAULT_CLASSIFIER_MODELS,
        choices=DEFAULT_CLASSIFIER_MODELS,
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-save-predictions", action="store_true")
    ap.add_argument("--classifier-epochs", type=int, default=DEFAULT_CLASSIFIER_EPOCHS)
    ap.add_argument("--classifier-batch-size", type=int, default=DEFAULT_CLASSIFIER_BATCH_SIZE)
    ap.add_argument("--classifier-learning-rate", type=float, default=DEFAULT_CLASSIFIER_LEARNING_RATE)
    ap.add_argument("--classifier-dropout", type=float, default=DEFAULT_CLASSIFIER_DROPOUT)
    ap.add_argument("--classifier-l2", type=float, default=DEFAULT_CLASSIFIER_L2)
    ap.add_argument("--classifier-loss", type=str, default=DEFAULT_CLASSIFIER_LOSS, choices=CLASSIFIER_LOSSES)
    ap.add_argument("--classifier-focal-gamma", type=float, default=DEFAULT_CLASSIFIER_FOCAL_GAMMA)
    ap.add_argument("--classifier-focal-alpha", type=float, default=DEFAULT_CLASSIFIER_FOCAL_ALPHA)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from training.sequence_model_trainer import SequenceModelTrainer, SequenceTrainingConfig

    config = SequenceTrainingConfig(
        exports_dir=args.exports_dir,
        results_dir=args.results_dir,
        windows=args.windows,
        classifier_models=args.classifier_models,
        random_seed=args.seed,
        overwrite=args.overwrite,
        save_predictions=not args.no_save_predictions,
        classifier_epochs=args.classifier_epochs,
        classifier_batch_size=args.classifier_batch_size,
        classifier_learning_rate=args.classifier_learning_rate,
        classifier_dropout=args.classifier_dropout,
        classifier_l2=args.classifier_l2,
        classifier_loss=args.classifier_loss,
        classifier_focal_gamma=args.classifier_focal_gamma,
        classifier_focal_alpha=args.classifier_focal_alpha,
    )
    SequenceModelTrainer(config).run()


if __name__ == "__main__":
    main()
