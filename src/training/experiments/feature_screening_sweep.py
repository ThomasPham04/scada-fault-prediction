"""
Feature Screening Sweep - training.experiments.feature_screening_sweep
Runs a grid sweep over feature-screening thresholds and saves per-run and
cross-run artifacts suitable for paper writing.

Usage:
    python -m src.training.experiments.feature_screening_sweep
    python -m src.training.experiments.feature_screening_sweep --ff-threshold-grid 0.8,0.85,0.9
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import RESULTS_DIR, WIND_FARM_A_DATASETS
from data_pipeline.preprocessing.feature_screening import FeatureScreening
from training.hyperparameters.feature_screening_defaults import (
    DEFAULT_ALPHA,
    DEFAULT_CSV_SEP,
    DEFAULT_FF_THRESHOLD_GRID,
    DEFAULT_MIN_MEAN_ABS_PB_R_GRID,
    DEFAULT_MIN_PB_SIG_RATIO_GRID,
    DEFAULT_MIN_PB_SIGN_CONSISTENCY_GRID,
    DEFAULT_SAMPLE_PER_FILE,
    DEFAULT_TARGET_COL,
    DROP_COLUMNS,
)


def parse_float_grid(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def threshold_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def build_run_name(
    min_mean_abs_pb_r: float,
    min_pb_sign_consistency: float,
    min_pb_sig_ratio: float,
    ff_threshold: float,
) -> str:
    return (
        f"pb_{threshold_tag(min_mean_abs_pb_r)}"
        f"__sign_{threshold_tag(min_pb_sign_consistency)}"
        f"__sig_{threshold_tag(min_pb_sig_ratio)}"
        f"__ff_{threshold_tag(ff_threshold)}"
    )


def safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.mean())
    return None if math.isnan(value) else value


def safe_median(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.median())
    return None if math.isnan(value) else value


def write_feature_list(path: Path, features: list[str]) -> None:
    path.write_text("\n".join(features), encoding="utf-8")


def build_run_summary(
    run_name: str,
    run_dir: Path,
    results: dict,
    params: dict,
) -> dict:
    agg = results["agg"]
    selected = results["selected"]
    final_features = list(results["final_features"])
    final_df = agg.loc[final_features].copy() if final_features else agg.iloc[0:0].copy()

    summary = {
        "run_name": run_name,
        "status": "ok",
        "output_dir": str(run_dir),
        "dataset_dir": params["dataset_dir"],
        "csv_sep": params["csv_sep"],
        "target_col": params["target_col"],
        "alpha": params["alpha"],
        "min_mean_abs_pb_r": params["min_mean_abs_pb_r"],
        "min_pb_sign_consistency": params["min_pb_sign_consistency"],
        "min_pb_sig_ratio": params["min_pb_sig_ratio"],
        "ff_threshold": params["ff_threshold"],
        "sample_per_file": params["sample_per_file"],
        "n_datasets_processed": len(results["per_dataset_results"]),
        "n_datasets_skipped": len(results["skipped_datasets"]),
        "n_features_aggregated": int(len(agg)),
        "n_selected_features": int(len(selected)),
        "n_final_features": int(len(final_features)),
        "n_redundant_removed": int(len(results["redundant_removed"])),
        "selected_mean_abs_pb_r_mean": safe_mean(selected["mean_abs_pb_r"]) if "mean_abs_pb_r" in selected else None,
        "selected_mean_abs_pb_r_median": safe_median(selected["mean_abs_pb_r"]) if "mean_abs_pb_r" in selected else None,
        "final_mean_abs_pb_r_mean": safe_mean(final_df["mean_abs_pb_r"]) if "mean_abs_pb_r" in final_df else None,
        "final_mean_abs_pb_r_median": safe_median(final_df["mean_abs_pb_r"]) if "mean_abs_pb_r" in final_df else None,
        "selected_features": selected.index.tolist(),
        "final_features": final_features,
        "redundant_removed": list(results["redundant_removed"]),
        "processed_datasets": sorted(results["per_dataset_results"].keys()),
        "skipped_datasets": results["skipped_datasets"],
        "top_10_final_features": final_features[:10],
        "top_20_selected_features": selected.index.tolist()[:20],
        "aggregated_metrics_csv": str(run_dir / "aggregated_feature_target_metrics.csv"),
        "selected_features_csv": str(run_dir / "selected_features.csv"),
        "final_selected_features_csv": str(run_dir / "final_selected_features.csv"),
        "feature_feature_corr_csv": str(run_dir / "feature_feature_corr.csv"),
        "skipped_datasets_csv": str(run_dir / "skipped_datasets.csv"),
        "selected_features_txt": str(run_dir / "selected_features_list.txt"),
        "final_features_txt": str(run_dir / "final_features_list.txt"),
        "bar_mean_abs_pb_r_png": str(run_dir / "bar_mean_abs_pb_r.png"),
        "bar_pb_sign_consistency_png": str(run_dir / "bar_pb_sign_consistency.png"),
        "boxplot_cross_dataset_pb_r_png": str(run_dir / "boxplot_cross_dataset_pb_r.png"),
        "boxplot_cross_dataset_spearman_png": str(run_dir / "boxplot_cross_dataset_spearman.png"),
        "heatmap_ff_corr_png": str(run_dir / "heatmap_ff_corr.png"),
    }
    return summary


def save_feature_frequency(path: Path, counter: Counter, n_runs: int, label: str) -> None:
    rows = [
        {
            "feature": feature,
            f"{label}_count": int(count),
            f"{label}_ratio": float(count / n_runs) if n_runs else 0.0,
        }
        for feature, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep feature-screening thresholds and save paper-ready summaries.",
    )
    parser.add_argument("--dataset-dir", type=str, default=WIND_FARM_A_DATASETS)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "feature_screening_sweep"),
    )
    parser.add_argument("--sep", type=str, default=DEFAULT_CSV_SEP)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--sample-per-file", type=int, default=DEFAULT_SAMPLE_PER_FILE)
    parser.add_argument("--min-mean-abs-pb-r-grid", type=str, default=DEFAULT_MIN_MEAN_ABS_PB_R_GRID)
    parser.add_argument("--min-pb-sign-consistency-grid", type=str, default=DEFAULT_MIN_PB_SIGN_CONSISTENCY_GRID)
    parser.add_argument("--min-pb-sig-ratio-grid", type=str, default=DEFAULT_MIN_PB_SIG_RATIO_GRID)
    parser.add_argument("--ff-threshold-grid", type=str, default=DEFAULT_FF_THRESHOLD_GRID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    pb_grid = parse_float_grid(args.min_mean_abs_pb_r_grid)
    sign_grid = parse_float_grid(args.min_pb_sign_consistency_grid)
    sig_ratio_grid = parse_float_grid(args.min_pb_sig_ratio_grid)
    ff_grid = parse_float_grid(args.ff_threshold_grid)

    sweep_config = {
        "dataset_dir": args.dataset_dir,
        "output_dir": str(output_dir),
        "csv_sep": args.sep,
        "target_col": args.target_col,
        "alpha": args.alpha,
        "sample_per_file": args.sample_per_file,
        "drop_columns": sorted(DROP_COLUMNS),
        "grids": {
            "min_mean_abs_pb_r": pb_grid,
            "min_pb_sign_consistency": sign_grid,
            "min_pb_sig_ratio": sig_ratio_grid,
            "ff_threshold": ff_grid,
        },
    }
    (output_dir / "sweep_config.json").write_text(json.dumps(sweep_config, indent=2), encoding="utf-8")

    selected_counter: Counter = Counter()
    final_counter: Counter = Counter()
    run_summaries: list[dict] = []

    combinations = list(itertools.product(pb_grid, sign_grid, sig_ratio_grid, ff_grid))
    print("=" * 80)
    print("Feature Screening Threshold Sweep")
    print("=" * 80)
    print(f"Total combinations: {len(combinations)}")

    for idx, (pb_thr, sign_thr, sig_thr, ff_thr) in enumerate(combinations, start=1):
        run_name = build_run_name(pb_thr, sign_thr, sig_thr, ff_thr)
        run_dir = runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx:03d}/{len(combinations):03d}] {run_name}")

        params = {
            "dataset_dir": args.dataset_dir,
            "csv_sep": args.sep,
            "target_col": args.target_col,
            "alpha": args.alpha,
            "min_mean_abs_pb_r": pb_thr,
            "min_pb_sign_consistency": sign_thr,
            "min_pb_sig_ratio": sig_thr,
            "ff_threshold": ff_thr,
            "sample_per_file": args.sample_per_file,
        }

        try:
            screening = FeatureScreening(
                dataset_dir=args.dataset_dir,
                output_dir=run_dir,
                csv_sep=args.sep,
                target_col=args.target_col,
                drop_columns=DROP_COLUMNS,
                alpha=args.alpha,
                min_mean_abs_pb_r=pb_thr,
                min_pb_sign_consistency=sign_thr,
                min_pb_sig_ratio=sig_thr,
                ff_threshold=ff_thr,
            )
            results = screening.run_pipeline(sample_per_file=args.sample_per_file)

            selected_features = results["selected"].index.tolist()
            final_features = list(results["final_features"])
            selected_counter.update(selected_features)
            final_counter.update(final_features)

            write_feature_list(run_dir / "selected_features_list.txt", selected_features)
            write_feature_list(run_dir / "final_features_list.txt", final_features)

            summary = build_run_summary(run_name, run_dir, results, params)
            (run_dir / "paper_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            run_summaries.append(summary)
        except Exception as exc:
            error_summary = {
                "run_name": run_name,
                "status": "failed",
                "output_dir": str(run_dir),
                "dataset_dir": args.dataset_dir,
                "csv_sep": args.sep,
                "target_col": args.target_col,
                "alpha": args.alpha,
                "min_mean_abs_pb_r": pb_thr,
                "min_pb_sign_consistency": sign_thr,
                "min_pb_sig_ratio": sig_thr,
                "ff_threshold": ff_thr,
                "sample_per_file": args.sample_per_file,
                "error": str(exc),
            }
            (run_dir / "paper_summary.json").write_text(json.dumps(error_summary, indent=2), encoding="utf-8")
            run_summaries.append(error_summary)
            print(f"  FAILED: {exc}")

    summary_df = pd.DataFrame(run_summaries)
    summary_csv = output_dir / "sweep_summary.csv"
    summary_json = output_dir / "sweep_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(run_summaries, indent=2), encoding="utf-8")

    ok_df = summary_df[summary_df["status"] == "ok"].copy() if not summary_df.empty else summary_df.copy()
    if not ok_df.empty:
        ok_df = ok_df.sort_values(
            by=[
                "min_mean_abs_pb_r",
                "min_pb_sign_consistency",
                "min_pb_sig_ratio",
                "ff_threshold",
            ],
            ascending=[True, True, True, True],
        )
        ok_df.to_csv(output_dir / "sweep_summary_sorted.csv", index=False)

    save_feature_frequency(output_dir / "selected_feature_frequency.csv", selected_counter, len(combinations), "selected")
    save_feature_frequency(output_dir / "final_feature_frequency.csv", final_counter, len(combinations), "final")

    print("=" * 80)
    print("Sweep complete")
    print(f"Runs saved to: {runs_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
