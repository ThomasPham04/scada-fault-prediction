"""
main.py - SCADA Fault Prediction Pipeline
Single entry point for the combined-sequence training flow.

Usage examples:
    # 1. Prepare windowed sequence exports from one combined CSV
    python src/main.py prepare --csv df_final.csv --feature-file final_features.csv --window-hours 24

    # 2. Train sequence classifiers from those exports
    python src/main.py train-sequences --windows 24

    # 3. Train specific classifier models
    python src/main.py train-sequences --windows 24 --model lstm
"""

from __future__ import annotations

import argparse
import os
import sys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from training.hyperparameters.classifier_defaults import (
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
from training.hyperparameters.feature_screening_defaults import (
    DEFAULT_EDA_MAX_FEATURES,
    DEFAULT_EDA_MAX_MISSING_PCT,
    DEFAULT_EDA_MIN_CORR,
    DEFAULT_EDA_REDUNDANCY_THRESHOLD,
)
from training.hyperparameters.sequence_defaults import (
    DEFAULT_LABEL_MODE,
    DEFAULT_PREDICTION_VAL_RATIO,
    DEFAULT_SCALER_TYPE,
    DEFAULT_TOP_K_WINDOWS,
    DEFAULT_VALIDATION_SOURCE,
    RANDOM_SEED,
    STRIDE,
    TIME_RESOLUTION,
)


def prediction_horizon_steps_from_hours(horizon_hours: int | None, time_resolution_minutes: int) -> int | None:
    """Convert a horizon in hours to whole timesteps for the sequence pipeline."""
    if horizon_hours is None:
        return None
    if horizon_hours <= 0:
        raise SystemExit("--prediction-horizon-hours must be positive.")

    horizon_steps = horizon_hours * 60 / time_resolution_minutes
    if not horizon_steps.is_integer():
        raise SystemExit(
            "--prediction-horizon-hours must convert to a whole number of "
            f"{time_resolution_minutes}-minute timesteps."
        )
    return int(horizon_steps)


def run_prepare_care(args: argparse.Namespace) -> None:
    """Build combined CSV from raw CARE per-event files, then prepare sequence exports."""
    import os
    from config import PROCESSED_DATA_DIR, WIND_FARM_A_DIR, WIND_FARM_A_DATASETS
    from data_pipeline.preprocessing.build_combined_csv import CAREToCombinedCSV
    from data_pipeline.preprocessing.combined_sequence_pipeline import CombinedSequencePipeline

    stride = args.stride if args.stride is not None else STRIDE
    prediction_horizon_steps = prediction_horizon_steps_from_hours(
        args.prediction_horizon_hours,
        TIME_RESOLUTION,
    )
    farm_dir = args.farm_dir or WIND_FARM_A_DIR
    datasets_dir = args.datasets_dir or os.path.join(farm_dir, "datasets")
    if args.combined_csv_output:
        combined_csv = args.combined_csv_output
    else:
        farm_name = os.path.basename(os.path.normpath(farm_dir))
        combined_csv = os.path.join(PROCESSED_DATA_DIR, farm_name, "combined.csv")

    print(f"Step 1/2: Building combined CSV from CARE event CSVs...")
    builder = CAREToCombinedCSV(farm_dir=farm_dir, datasets_dir=datasets_dir)
    builder.build(output_path=combined_csv)

    print(f"\nStep 2/2: Preparing sequence exports from combined CSV...")
    pipeline = CombinedSequencePipeline(
        csv_path=combined_csv,
        feature_file=args.feature_file,
        output_dir=args.sequence_output_dir,
        selected_windows_hours=args.window_hours,
        window_candidates_hours=args.window_candidates_hours,
        top_k_windows=args.top_k_windows,
        stride_steps=stride,
        prediction_horizon_steps=prediction_horizon_steps,
        expected_feature_count=args.expected_feature_count,
        scaler_type=args.combined_scaler,
        validation_source=args.validation_source,
        prediction_val_ratio=args.prediction_val_ratio,
        label_mode=args.label_mode,
        run_window_search=not args.skip_window_search,
        random_seed=args.seed,
        skip_classifier=args.skip_classifier_export,
    )
    pipeline.run()


def run_prepare(args: argparse.Namespace) -> None:
    """Prepare classifier exports from a combined CSV."""
    stride = args.stride if args.stride is not None else STRIDE
    prediction_horizon_steps = prediction_horizon_steps_from_hours(
        args.prediction_horizon_hours,
        TIME_RESOLUTION,
    )
    if not args.csv:
        raise SystemExit(
            "prepare requires --csv. The CSV must contain time_stamp, asset_id, "
            "train_test, sequence_id, and label or status_type_id."
        )

    from data_pipeline.preprocessing.combined_sequence_pipeline import (
        CombinedSequencePipeline,
        looks_like_combined_sequence_csv,
    )

    if not looks_like_combined_sequence_csv(args.csv):
        raise SystemExit(
            "Unsupported CSV schema. This CLI now supports only the combined-sequence "
            "schema: time_stamp, asset_id, train_test, sequence_id, and label or "
            "status_type_id."
        )

    pipeline = CombinedSequencePipeline(
        csv_path=args.csv,
        feature_file=args.feature_file,
        output_dir=args.sequence_output_dir,
        selected_windows_hours=args.window_hours,
        window_candidates_hours=args.window_candidates_hours,
        top_k_windows=args.top_k_windows,
        stride_steps=stride,
        prediction_horizon_steps=prediction_horizon_steps,
        expected_feature_count=args.expected_feature_count,
        scaler_type=args.combined_scaler,
        validation_source=args.validation_source,
        prediction_val_ratio=args.prediction_val_ratio,
        label_mode=args.label_mode,
        run_window_search=not args.skip_window_search,
        random_seed=args.seed,
        skip_classifier=args.skip_classifier_export,
    )
    pipeline.run()


def run_train_sequences(args: argparse.Namespace) -> None:
    """Train global sequence classifiers."""
    from config import PROCESSED_DATA_DIR, RESULTS_DIR
    from training.sequence_model_trainer import SequenceModelTrainer, SequenceTrainingConfig

    exports_dir = args.exports_dir or os.path.join(PROCESSED_DATA_DIR, "sequence_exports")
    results_dir = args.results_dir or os.path.join(RESULTS_DIR, "sequence_training_results")

    _classifier_choices = set(DEFAULT_CLASSIFIER_MODELS)

    if args.model:
        unknown = [m for m in args.model if m not in _classifier_choices]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Choose from: "
                             f"{sorted(_classifier_choices)}")
        classifier_models = list(args.model)
    else:
        classifier_models = args.classifier_models

    config = SequenceTrainingConfig(
        exports_dir=exports_dir,
        results_dir=results_dir,
        windows=args.windows,
        classifier_models=classifier_models,
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


def add_label_mode_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--label-mode",
        type=str,
        default=DEFAULT_LABEL_MODE,
        choices=["future_horizon", "input_window"],
        help=(
            "Window label target. 'future_horizon' preserves the original "
            "prediction task: any fault after the input window within H steps. "
            "'input_window' labels a window positive when any input timestamp "
            "is positive."
        ),
    )


def add_combined_csv_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        metavar="PATH",
        help="Combined sequence CSV with asset_id and sequence_id boundaries.",
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Feature list CSV for combined-sequence exports, e.g. final_features.csv.",
    )
    parser.add_argument(
        "--sequence-output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output root for sequence exports.",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        nargs="+",
        default=None,
        metavar="H",
        help="Export these window sizes directly and skip window search.",
    )
    parser.add_argument(
        "--window-candidates-hours",
        type=int,
        nargs="+",
        default=None,
        metavar="H",
        help="Window sizes to test during window search.",
    )
    parser.add_argument(
        "--prediction-horizon-hours",
        type=int,
        default=None,
        metavar="H",
        help=(
            "Future prediction horizon in hours. Defaults to the problem config "
            "value when omitted."
        ),
    )
    parser.add_argument(
        "--top-k-windows",
        type=int,
        default=DEFAULT_TOP_K_WINDOWS,
        help="Number of best windows to export after window search.",
    )
    parser.add_argument(
        "--skip-window-search",
        action="store_true",
        help="Skip probe-model search and export the first candidate window only.",
    )
    parser.add_argument(
        "--expected-feature-count",
        type=int,
        default=None,
        help="Optional guard for the selected feature count.",
    )
    parser.add_argument(
        "--combined-scaler",
        type=str,
        default=DEFAULT_SCALER_TYPE,
        choices=["minmax", "standard"],
        help="Scaler for combined CSV exports.",
    )
    parser.add_argument(
        "--validation-source",
        type=str,
        default=DEFAULT_VALIDATION_SOURCE,
        choices=["train_tail", "prediction"],
        help=(
            "How to create the validation split. 'train_tail' uses the tail "
            "of each train segment. 'prediction' "
            "uses the first part of each prediction segment for validation and "
            "the rest for test."
        ),
    )
    parser.add_argument(
        "--prediction-val-ratio",
        type=float,
        default=DEFAULT_PREDICTION_VAL_RATIO,
        help=(
            "When --validation-source prediction is used, fraction of each "
            "asset_id + sequence_id prediction segment assigned to validation."
        ),
    )
    add_label_mode_flag(parser)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stride (step) between sliding-window sequence starts in timesteps. "
            "Defaults to the centralized sequence setting (STRIDE=6, i.e. 1-hour steps). "
            "Larger values reduce overlap and dataset size; e.g. --stride 36 gives "
            "~75%% overlap with a 144-step window."
        ),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--skip-classifier-export",
        action="store_true",
        default=False,
        help="Skip exporting classifier windows.",
    )


def add_sequence_training_flags(parser: argparse.ArgumentParser) -> None:
    classifier_choices = DEFAULT_CLASSIFIER_MODELS
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        choices=classifier_choices,
        metavar="MODEL",
        help=(
            f"Train only these classifier model(s): {', '.join(classifier_choices)}. "
            "Overrides --classifier-models."
        ),
    )
    parser.add_argument(
        "--exports-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Root folder containing window_<H>h sequence exports.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output folder for sequence training results.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_WINDOWS,
        metavar="H",
        help="Window sizes to train.",
    )
    parser.add_argument(
        "--classifier-models",
        type=str,
        nargs="+",
        default=classifier_choices,
        choices=classifier_choices,
        help="Supervised sequence classifiers to train.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even if metrics already exist.",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Skip verbose prediction CSV outputs.",
    )
    parser.add_argument("--classifier-epochs", type=int, default=DEFAULT_CLASSIFIER_EPOCHS)
    parser.add_argument("--classifier-batch-size", type=int, default=DEFAULT_CLASSIFIER_BATCH_SIZE)
    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=DEFAULT_CLASSIFIER_LEARNING_RATE,
        help="Adam learning rate for supervised classifier models.",
    )
    parser.add_argument(
        "--classifier-dropout",
        type=float,
        default=DEFAULT_CLASSIFIER_DROPOUT,
        help=(
            "Override classifier dropout rate for all classifier dropout layers. "
            "When omitted, each architecture uses its built-in defaults."
        ),
    )
    parser.add_argument(
        "--classifier-l2",
        type=float,
        default=DEFAULT_CLASSIFIER_L2,
        help="L2 regularization strength for classifier Conv/RNN/Dense kernels.",
    )
    parser.add_argument(
        "--classifier-loss",
        type=str,
        default=DEFAULT_CLASSIFIER_LOSS,
        choices=CLASSIFIER_LOSSES,
        help=(
            "Classifier loss. 'focal' uses BinaryFocalCrossentropy with class "
            "balancing enabled and disables external class_weight."
        ),
    )
    parser.add_argument(
        "--classifier-focal-gamma",
        type=float,
        default=DEFAULT_CLASSIFIER_FOCAL_GAMMA,
        help="Gamma for focal loss when --classifier-loss focal is used.",
    )
    parser.add_argument(
        "--classifier-focal-alpha",
        type=float,
        default=DEFAULT_CLASSIFIER_FOCAL_ALPHA,
        help="Positive-class alpha for focal loss when --classifier-loss focal is used.",
    )


def run_eda(args: argparse.Namespace) -> None:
    """Run exploratory data analysis on a combined CSV."""
    from pathlib import Path
    from config import RESULTS_DIR
    from data_pipeline.eda import EDAReport

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        farm_name = csv_path.parent.name or "default"
        output_dir = Path(RESULTS_DIR) / "eda" / farm_name

    EDAReport(
        csv_path=csv_path,
        output_dir=output_dir,
        feature_file=args.feature_file,
        max_features=args.max_features,
        sample_asset=args.sample_asset,
        select_features=args.select_features,
        min_corr=args.min_corr,
        max_missing_pct=args.max_missing_pct,
        redundancy_threshold=args.redundancy_threshold,
    ).run()


def add_eda_flags(parser: argparse.ArgumentParser) -> None:
    """Flags for the `eda` subcommand."""
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the combined CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Where to write EDA artifacts. Defaults to results/eda/<farm-name>/.",
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional CSV with a 'final_feature' column to restrict analysis to a subset.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_EDA_MAX_FEATURES,
        metavar="N",
        help="Cap on features shown in correlation heatmap and distribution boxplots.",
    )
    parser.add_argument(
        "--sample-asset",
        type=int,
        default=None,
        metavar="ID",
        help="Asset to use for the time-series overview plot. Defaults to first asset in the CSV.",
    )
    parser.add_argument(
        "--select-features",
        action="store_true",
        help="Run EDA-based feature selection after analysis.",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=DEFAULT_EDA_MIN_CORR,
        metavar="F",
        help="Minimum |Spearman rho| with label to keep a feature (default 0.02).",
    )
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=DEFAULT_EDA_MAX_MISSING_PCT,
        metavar="F",
        help="Drop features with missing %% above this (default 80).",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=DEFAULT_EDA_REDUNDANCY_THRESHOLD,
        metavar="F",
        help="Inter-feature |Spearman rho| above which the weaker feature is dropped (default 0.90).",
    )


def add_prepare_care_flags(parser: argparse.ArgumentParser) -> None:
    """Flags for prepare-care: raw CARE data → combined CSV → sequence exports."""
    parser.add_argument(
        "--farm-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to wind farm root directory containing event_info.csv. "
             "Defaults to Wind Farm A.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to datasets sub-directory. Defaults to <farm-dir>/datasets.",
    )
    parser.add_argument(
        "--combined-csv-output",
        type=str,
        default=None,
        metavar="PATH",
        help="Where to save the combined CSV. "
             "Defaults to Dataset/processed/<farm-name>/combined.csv",
    )
    # Reuse combined-csv flags for the sequence export step
    parser.add_argument("--feature-file", type=str, default=None, metavar="PATH")
    parser.add_argument("--sequence-output-dir", type=str, default=None, metavar="DIR")
    parser.add_argument("--window-hours", type=int, nargs="+", default=None, metavar="H")
    parser.add_argument("--window-candidates-hours", type=int, nargs="+", default=None, metavar="H")
    parser.add_argument(
        "--prediction-horizon-hours",
        type=int,
        default=None,
        metavar="H",
        help="Future prediction horizon in hours. Defaults to the problem config value when omitted.",
    )
    parser.add_argument("--top-k-windows", type=int, default=DEFAULT_TOP_K_WINDOWS)
    parser.add_argument("--skip-window-search", action="store_true")
    parser.add_argument("--expected-feature-count", type=int, default=None)
    parser.add_argument("--combined-scaler", type=str, default=DEFAULT_SCALER_TYPE, choices=["minmax", "standard"])
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stride (step) between sliding-window sequence starts in timesteps. "
            "Defaults to the centralized STRIDE setting (6 = 1-hour steps)."
        ),
    )
    parser.add_argument(
        "--validation-source",
        type=str,
        default=DEFAULT_VALIDATION_SOURCE,
        choices=["train_tail", "prediction"],
    )
    parser.add_argument("--prediction-val-ratio", type=float, default=DEFAULT_PREDICTION_VAL_RATIO)
    add_label_mode_flag(parser)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--skip-classifier-export",
        action="store_true",
        default=False,
        help="Skip exporting classifier windows.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SCADA Fault Prediction - combined-sequence pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical workflow (CARE dataset):\n"
            "  1. python src/main.py prepare-care\n"
            "  2. python src/main.py train-sequences --windows 24\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_care_parser = subparsers.add_parser(
        "prepare-care",
        help="Build combined CSV from raw CARE event files, then prepare sequence exports.",
    )
    add_prepare_care_flags(prepare_care_parser)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare sequence exports from an already-built combined CSV.",
    )
    add_combined_csv_flags(prepare_parser)

    train_sequences_parser = subparsers.add_parser(
        "train-sequences",
        help="Train sequence classifiers from exports.",
    )
    add_sequence_training_flags(train_sequences_parser)

    eda_parser = subparsers.add_parser(
        "eda",
        help="Run exploratory data analysis on a combined CSV.",
    )
    add_eda_flags(eda_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-care":
        run_prepare_care(args)
    elif args.command == "prepare":
        run_prepare(args)
    elif args.command == "train-sequences":
        run_train_sequences(args)
    elif args.command == "eda":
        run_eda(args)


if __name__ == "__main__":
    main()
