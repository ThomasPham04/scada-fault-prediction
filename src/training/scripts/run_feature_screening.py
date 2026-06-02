"""
Run Feature Screening - training.scripts.run_feature_screening
Entry-point script for cross-dataset feature screening on SCADA CSV files.

Usage:
    # Split a combined CSV by sequence_id and screen across all events:
    python -m src.training.scripts.run_feature_screening \
        --combined-csv "Dataset/processed/combined_dataset.csv"

    # Screen pre-existing per-event CSV files directly:
    python -m src.training.scripts.run_feature_screening \
        --dataset-dir "path/to/per_event_splits"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from config import PROCESSED_DATA_DIR, RESULTS_DIR, WIND_FARM_A_DATASETS
from data_pipeline.preprocessing.feature_screening import FeatureScreening
from training.hyperparameters.feature_screening_defaults import (
    DEFAULT_ALPHA,
    DEFAULT_COMBINED_CSV_SEP,
    DEFAULT_FF_THRESHOLD,
    DEFAULT_MIN_MEAN_ABS_PB_R,
    DEFAULT_MIN_PB_SIG_RATIO,
    DEFAULT_MIN_PB_SIGN_CONSISTENCY,
    DEFAULT_SAMPLE_PER_FILE,
    DEFAULT_TARGET_COL,
    DROP_COLUMNS,
)


def split_combined_csv(combined_csv: str | Path, splits_dir: str | Path) -> Path:
    """
    Split a combined CSV into one file per sequence_id.

    Each output file is named event_<sequence_id>.csv and contains all rows
    for that event, including the label column.

    Returns:
        Path to the splits directory.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading combined CSV: {combined_csv}")
    df = pd.read_csv(combined_csv)
    n_events = df["sequence_id"].nunique()
    print(f"  {len(df):,} rows  |  {n_events} events (sequence_ids)  |  splitting...")

    for seq_id, group in df.groupby("sequence_id"):
        out_path = splits_dir / f"event_{seq_id}.csv"
        group.to_csv(out_path, index=False)

    print(f"  Saved {n_events} event CSV files to: {splits_dir}")
    return splits_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run feature screening using point-biserial and Spearman aggregation "
            "across multiple per-event CSV files."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--combined-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a combined CSV with a sequence_id column. "
            "Will be split into per-event files before screening."
        ),
    )
    source.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory of pre-split per-event CSV files.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Where to save per-event splits when --combined-csv is used. "
            "Defaults to Dataset/processed/per_event_splits/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory for screening results.",
    )
    parser.add_argument("--sep", type=str, default=DEFAULT_COMBINED_CSV_SEP)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--min-mean-abs-pb-r", type=float, default=DEFAULT_MIN_MEAN_ABS_PB_R)
    parser.add_argument("--min-pb-sign-consistency", type=float, default=DEFAULT_MIN_PB_SIGN_CONSISTENCY)
    parser.add_argument("--min-pb-sig-ratio", type=float, default=DEFAULT_MIN_PB_SIG_RATIO)
    parser.add_argument("--ff-threshold", type=float, default=DEFAULT_FF_THRESHOLD)
    parser.add_argument("--sample-per-file", type=int, default=DEFAULT_SAMPLE_PER_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.combined_csv:
        splits_dir = Path(
            args.splits_dir or os.path.join(PROCESSED_DATA_DIR, "per_event_splits")
        )
        dataset_dir = split_combined_csv(args.combined_csv, splits_dir)
        sep = ","
    else:
        dataset_dir = args.dataset_dir or WIND_FARM_A_DATASETS
        sep = args.sep

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "feature_screening_per_event")

    screening = FeatureScreening(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        csv_sep=sep,
        target_col=args.target_col,
        drop_columns=DROP_COLUMNS,
        alpha=args.alpha,
        min_mean_abs_pb_r=args.min_mean_abs_pb_r,
        min_pb_sign_consistency=args.min_pb_sign_consistency,
        min_pb_sig_ratio=args.min_pb_sig_ratio,
        ff_threshold=args.ff_threshold,
    )

    results = screening.run_pipeline(sample_per_file=args.sample_per_file)

    print("=" * 70)
    print("Feature Screening Complete")
    print("=" * 70)
    print(f"Events processed      : {len(results['per_dataset_results'])}")
    print(f"Selected features     : {len(results['selected'])}")
    print(f"Final (after redund.) : {len(results['final_features'])}")
    print(f"Output directory      : {output_dir}")


if __name__ == "__main__":
    main()
