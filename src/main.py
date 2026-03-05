"""
main.py — SCADA Fault Prediction Pipeline
Single entry point for all pipeline stages.

Usage examples:
    # 1. Prepare per-asset data (full Wind Farm A dataset)
    python src/main.py prepare

    # 1b. Prepare from a single CSV file
    python src/main.py prepare --csv "D:/Data/turbine_01.csv"
    python src/main.py prepare --csv "D:/Data/turbine_01.csv" --asset-name turbine_01

    # 2. Train models
    python src/main.py train --model lstm
    python src/main.py train --model random_forest
    python src/main.py train --model xgboost
    python src/main.py train --model all          # lstm + rf + xgboost

    # 2b. Train from a CSV directly (prepare + train in one command)
    python src/main.py train --model lstm --csv "D:/Data/turbine_01.csv"

    # 3. Evaluate models
    python src/main.py evaluate --model lstm
    python src/main.py evaluate --model random_forest
    python src/main.py evaluate --model both      # xgboost + random_forest

    # 4. Run full pipeline end-to-end
    python src/main.py run --model lstm
    python src/main.py run --model all

    # Extra flags
    python src/main.py train --model xgboost --use_stats
    python src/main.py train --model lstm --assets 10 11
    python src/main.py evaluate --model random_forest --assets 10
"""

import argparse
import os
import sys

# Ensure src/ is on the path regardless of where you call this from
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline():
    from config import (
        WIND_FARM_A_DIR, WIND_FARM_A_DATASETS, PER_ASSET_PROCESSED_DIR,
        WINDOW_SIZE, STRIDE, VAL_SIZE,
    )
    from training.scripts.prepare_per_asset import PerAssetPipeline
    return PerAssetPipeline(
        farm_dir=WIND_FARM_A_DIR,
        datasets_dir=WIND_FARM_A_DATASETS,
        output_dir=PER_ASSET_PROCESSED_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        val_size=VAL_SIZE,
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_prepare(args) -> None:
    """Stage 1 — prepare per-asset data arrays."""
    pipeline = _make_pipeline()

    csv = getattr(args, "csv", None)
    if csv:
        # Single-CSV mode
        pipeline.run_from_csv(
            csv_path=csv,
            asset_name=getattr(args, "asset_name", None),
            label=getattr(args, "label", "normal"),
        )
    else:
        # Full Wind Farm A dataset
        print("=" * 70)
        print("STAGE 1 — Prepare Per-Asset Data")
        print("=" * 70)
        pipeline.run()


def run_train(args) -> None:
    """Stage 2 — train the requested model(s)."""
    from config import MODELS_DIR, ensure_dirs
    from training.trainer import LSTMTrainer, TreeTrainer

    csv = getattr(args, "csv", None)
    asset_name = None

    if csv:
        # Prepare from CSV first, then train only that asset
        pipeline   = _make_pipeline()
        asset_dir  = pipeline.run_from_csv(
            csv_path=csv,
            asset_name=getattr(args, "asset_name", None),
            label=getattr(args, "label", "normal"),
        )
        asset_name = os.path.basename(asset_dir).replace("asset_", "")

    ensure_dirs()
    os.makedirs(MODELS_DIR, exist_ok=True)
    model = args.model.lower()

    asset_filter = [asset_name] if asset_name else getattr(args, "assets", None)

    if model in ("lstm", "all"):
        print("\n" + "=" * 70)
        print("TRAINING — LSTM")
        print("=" * 70)
        LSTMTrainer().run_per_asset(asset_filter=asset_filter)

    if model in ("random_forest", "rf", "all"):
        print("\n" + "=" * 70)
        print("TRAINING — Random Forest")
        print("=" * 70)
        TreeTrainer(model_type="random_forest").run_per_asset(args)

    if model in ("xgboost", "xgb", "all"):
        print("\n" + "=" * 70)
        print("TRAINING — XGBoost")
        print("=" * 70)
        TreeTrainer(model_type="xgboost").run_per_asset(args)


def run_evaluate(args) -> None:
    """Stage 3 — evaluate the requested model(s)."""
    from evaluation.evaluator import LSTMEvaluator, TreeEvaluator

    model = args.model.lower()
    
    csv = getattr(args, "csv", None)
    asset_name = None
    if csv:
        asset_name = getattr(args, "asset_name", None) or os.path.splitext(os.path.basename(csv))[0]
    
    asset_filter = [asset_name] if asset_name else getattr(args, "assets", None)

    if model in ("lstm", "all"):
        print("\n" + "=" * 70)
        print("EVALUATION — LSTM")
        print("=" * 70)
        LSTMEvaluator().evaluate_per_asset(asset_filter=asset_filter)

    tree_models = []
    if model in ("random_forest", "rf", "both", "all"):
        tree_models.append("random_forest")
    if model in ("xgboost", "xgb", "both", "all"):
        tree_models.append("xgboost")

    for mk in tree_models:
        print("\n" + "=" * 70)
        print(f"EVALUATION — {mk.upper()}")
        print("=" * 70)
        TreeEvaluator().evaluate_per_asset(mk, use_stats=args.use_stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SCADA Fault Prediction — unified pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- shared CSV flags (reused by prepare and train) ----
    def add_csv_flags(p):
        p.add_argument("--csv", type=str, default=None, metavar="PATH",
                       help="Path to a raw SCADA CSV file. "
                            "If given, processes that file instead of the full dataset.")
        p.add_argument("--asset-name", type=str, default=None, dest="asset_name",
                       metavar="NAME",
                       help="Asset folder name for the CSV output (default: CSV filename stem).")
        p.add_argument("--label", type=str, default="normal",
                       choices=["normal", "anomaly"],
                       help="Label for the held-out test portion when using --csv (default: normal).")

    # ---- shared model/asset flags ----
    def add_common(p):
        p.add_argument("--model", type=str, default="lstm",
                       choices=["lstm", "random_forest", "rf", "xgboost", "xgb", "both", "all"],
                       help="Which model(s) to use.")
        p.add_argument("--assets", type=str, nargs="+", default=None,
                       metavar="ID",
                       help="Restrict to specific asset IDs/names. E.g. --assets 10 11")
        p.add_argument("--use_stats", action="store_true",
                       help="Use statistical window features for tree models.")

    # ---- shared hyperparameter flags ----
    def add_hparams(p):
        p.add_argument("--n_estimators",          type=int,   default=400)
        p.add_argument("--max_depth",             type=int,   default=6)
        p.add_argument("--learning_rate",         type=float, default=0.05)
        p.add_argument("--subsample",             type=float, default=0.8)
        p.add_argument("--colsample_bytree",      type=float, default=0.8)
        p.add_argument("--min_child_weight",      type=float, default=1.0)
        p.add_argument("--gamma",                 type=float, default=0.0)
        p.add_argument("--reg_lambda",            type=float, default=1.0)
        p.add_argument("--reg_alpha",             type=float, default=0.0)
        p.add_argument("--early_stopping_rounds", type=int,   default=30)
        p.add_argument("--tree_method",           type=str,   default="hist")
        p.add_argument("--use_gpu",               action="store_true")
        p.add_argument("--min_samples_split",     type=int,   default=2)
        p.add_argument("--min_samples_leaf",      type=int,   default=1)
        p.add_argument("--max_features",          type=str,   default="sqrt")
        p.add_argument("--class_weight",          type=str,   default="balanced")
        p.add_argument("--seed",                  type=int,   default=42)

    # ---- prepare ----
    prep_p = sub.add_parser("prepare",
                             help="Prepare per-asset training data. "
                                  "Use --csv to process a single file.")
    add_csv_flags(prep_p)

    # ---- train ----
    train_p = sub.add_parser("train",
                              help="Train model(s). Use --csv to prepare + train from a single file.")
    add_common(train_p)
    add_csv_flags(train_p)
    add_hparams(train_p)

    # ---- evaluate ----
    eval_p = sub.add_parser("evaluate", help="Evaluate trained model(s) on test events.")
    add_common(eval_p)
    add_csv_flags(eval_p)

    # ---- run (prepare + train + evaluate) ----
    run_p = sub.add_parser("run", help="Full pipeline: prepare → train → evaluate.")
    add_common(run_p)
    add_csv_flags(run_p)
    add_hparams(run_p)
    run_p.add_argument("--skip_prepare", action="store_true",
                       help="Skip data preparation (use if already prepared).")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "prepare":
        run_prepare(args)

    elif args.command == "train":
        run_train(args)

    elif args.command == "evaluate":
        run_evaluate(args)

    elif args.command == "run":
        if not args.skip_prepare:
            run_prepare(args)
        run_train(args)
        run_evaluate(args)


if __name__ == "__main__":
    main()
