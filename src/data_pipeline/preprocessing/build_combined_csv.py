"""
build_combined_csv.py — data_pipeline.preprocessing.build_combined_csv

Converts raw CARE per-event CSVs into a single combined CSV ready for
CombinedSequencePipeline.

For each event in event_info.csv:
  - Load the event CSV
  - Apply feature engineering (angle sin/cos encoding)
  - Add asset_id  (from event_info)
  - Add sequence_id = event_id
  - Add label column via GroundTruth (0=normal, 1=fault window rows)

Output columns (comma-separated CSV):
  time_stamp, asset_id, sequence_id, train_test, status_type_id, label,
  <engineered feature columns>

Usage:
    # Wind Farm A with defaults
    python -m src.data_pipeline.preprocessing.build_combined_csv

    # Specify farm directory and output path
    python -m src.data_pipeline.preprocessing.build_combined_csv ^
        --farm-dir "Dataset/raw/Wind Farm A" ^
        --output "Dataset/processed/Wind Farm A/combined.csv"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add src/ to sys.path so config and sub-packages resolve correctly
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import PROCESSED_DATA_DIR, WIND_FARM_A_DIR, WIND_FARM_A_DATASETS
from data_pipeline.loaders.event_loader import EventLoader
from data_pipeline.preprocessing.feature_engineering import FeatureEngineer
from data_pipeline.preprocessing.ground_truth import GroundTruth

_GT_REQUIRED = {"event_id", "event_label", "event_start_id", "event_end_id"}


class CAREToCombinedCSV:
    """
    Reads all CARE event CSVs for a wind farm, applies feature engineering,
    and concatenates them into one combined CSV for CombinedSequencePipeline.

    Args:
        farm_dir: Path to wind farm root directory (contains event_info.csv).
        datasets_dir: Path to datasets sub-directory (contains <event_id>.csv).
    """

    def __init__(self, farm_dir: str, datasets_dir: str) -> None:
        self.farm_dir = farm_dir
        self.datasets_dir = datasets_dir
        self._loader = EventLoader(farm_dir=farm_dir, datasets_dir=datasets_dir)
        self._engineer = FeatureEngineer()

    def build(self, output_path: str | Path) -> Path:
        """
        Build the combined CSV and save it to output_path.

        Returns:
            Absolute path to the saved combined CSV.
        """
        output_path = Path(output_path)
        event_info = self._loader.load_event_info()
        asset_col = "asset_id" if "asset_id" in event_info.columns else "asset"

        has_gt = _GT_REQUIRED.issubset(event_info.columns)
        gt = GroundTruth(event_info) if has_gt else None
        if not has_gt:
            print(
                "[WARN] event_info is missing event_start_id / event_end_id. "
                "label column will be all zeros — evaluation will be wrong."
            )

        print(f"\n{'='*70}")
        print(f"Building combined CSV from {len(event_info)} events")
        print(f"Output: {output_path}")
        print(f"{'='*70}")

        parts: list[pd.DataFrame] = []
        feature_cols: list[str] | None = None

        for _, event in tqdm(event_info.iterrows(), total=len(event_info)):
            event_id = int(event["event_id"])
            try:
                df = self._loader.load_event_data(event_id)
                if asset_col in event.index:
                    asset_id = int(event[asset_col])
                elif "asset_id" in df.columns:
                    asset_id = int(df["asset_id"].iloc[0])
                else:
                    raise ValueError(
                        f"Cannot determine asset_id for event {event_id}: "
                        "not in event_info and not in event CSV."
                    )

                # Feature engineering: angle → sin/cos
                df = self._engineer.engineer_angle_features(df)

                if feature_cols is None:
                    feature_cols = self._engineer.get_feature_columns(df)
                    print(f"\n  Feature engineering complete: {len(feature_cols)} features")

                # Per-row fault labels from GroundTruth
                if gt is not None:
                    labels = gt.make_labels(df, event_id).astype(np.int8)
                else:
                    labels = pd.Series(
                        np.zeros(len(df), dtype=np.int8),
                        index=df.index,
                        name="label",
                    )

                # Select only the columns needed for the combined CSV
                keep = ["time_stamp", "train_test", "status_type_id"] + [
                    c for c in feature_cols if c in df.columns
                ]
                out = df[keep].copy()
                out["asset_id"] = asset_id
                out["sequence_id"] = event_id
                out["label"] = labels.values

                parts.append(out)
                n_fault = int(labels.sum())
                print(
                    f"  Event {event_id:3d} (asset {asset_id}) "
                    f"{str(event['event_label']):7s}: "
                    f"{len(df):6,} rows, {n_fault:,} fault rows"
                )

            except Exception as exc:
                print(f"  Event {event_id}: ERROR — {exc}")
                continue

        if not parts:
            raise RuntimeError("No events processed — check farm_dir and datasets_dir paths.")

        combined = pd.concat(parts, ignore_index=True)

        # Canonical column order: metadata first, then features
        meta_cols = ["time_stamp", "asset_id", "sequence_id", "train_test", "status_type_id", "label"]
        meta_cols = [c for c in meta_cols if c in combined.columns]
        feat_cols = [c for c in combined.columns if c not in set(meta_cols)]
        combined = combined[meta_cols + feat_cols]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

        n_fault = int(combined["label"].sum())
        n_train = int((combined["train_test"] == "train").sum())
        print(f"\n{'='*70}")
        print(f"Combined CSV saved: {output_path}")
        print(f"  Total rows : {len(combined):,}")
        print(f"  train rows : {n_train:,}")
        print(f"  prediction : {len(combined) - n_train:,}")
        print(f"  Fault rows : {n_fault:,}  (label=1)")
        print(f"  Features   : {len(feat_cols)}")
        print(f"{'='*70}")

        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined CSV from CARE per-event datasets "
            "for input into CombinedSequencePipeline."
        )
    )
    parser.add_argument(
        "--farm-dir",
        type=str,
        default=WIND_FARM_A_DIR,
        metavar="DIR",
        help="Path to wind farm root directory containing event_info.csv.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to datasets sub-directory. Defaults to <farm-dir>/datasets.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output path for the combined CSV. "
            "Defaults to Dataset/processed/<farm-name>/combined.csv"
        ),
    )
    args = parser.parse_args()

    farm_dir = args.farm_dir
    datasets_dir = args.datasets_dir or os.path.join(farm_dir, "datasets")

    if args.output:
        output_path = args.output
    else:
        farm_name = os.path.basename(os.path.normpath(farm_dir))
        output_path = os.path.join(PROCESSED_DATA_DIR, farm_name, "combined.csv")

    builder = CAREToCombinedCSV(farm_dir=farm_dir, datasets_dir=datasets_dir)
    builder.build(output_path=output_path)


if __name__ == "__main__":
    main()
