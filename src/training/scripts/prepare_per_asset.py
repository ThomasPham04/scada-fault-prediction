"""
Prepare Per-Asset Data — training.scripts.prepare_per_asset
Entry-point script for preparing per-turbine (per-asset) training data.

Follows the ref project's approach: train one model per asset (turbine),
each on its own historical training data, with its own StandardScaler.

Output structure:
    Dataset/processed/Wind Farm A/per_asset/asset_{id}/
        X_train.npy, X_val.npy, y_train.npy, y_val.npy
        test_by_event/event_{id}.npz
        scaler_asset_{id}.pkl
        metadata.pkl

Usage:
    python -m src.training.scripts.prepare_per_asset
"""

import os
import sys
import numpy as np
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → src/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    WIND_FARM_A_DIR,
    WIND_FARM_A_DATASETS,
    PER_ASSET_PROCESSED_DIR,
    WINDOW_SIZE,
    STRIDE,
    VAL_SIZE,
)
from data_pipeline.loaders.event_loader import EventLoader
from data_pipeline.loaders.sequence_maker import SequenceMaker
from data_pipeline.preprocessing.splitter import DataSplitter
from data_pipeline.preprocessing.normalizer import AssetNormalizer


class PerAssetPipeline:
    """
    Orchestrates the full per-turbine data preparation pipeline.

    For each asset:
      1. Load & combine training data from all its events.
      2. Load test (prediction) data per event.
      3. Temporal train/val split.
      4. Create sliding-window sequences.
      5. Fit a per-asset StandardScaler and transform all splits.
      6. Save arrays, scaler, and metadata to disk.

    Args:
        farm_dir: Path to Wind Farm A root directory.
        datasets_dir: Path to the raw CSV datasets directory.
        output_dir: Root directory for per-asset output.
        window_size: Sliding window length in timesteps.
        stride: Step between sequence start positions.
        val_size: Fraction of training data reserved for validation.
    """

    def __init__(
        self,
        farm_dir: str,
        datasets_dir: str,
        output_dir: str,
        window_size: int,
        stride: int,
        val_size: float,
    ) -> None:
        self.farm_dir     = farm_dir
        self.datasets_dir = datasets_dir
        self.output_dir   = output_dir
        self.window_size  = window_size
        self.stride       = stride
        self.val_size     = val_size

        self._event_loader = EventLoader(farm_dir=farm_dir, datasets_dir=datasets_dir)
        self._seq_maker    = SequenceMaker(window_size=window_size, stride=stride)

    def _save_test_events(self, asset_dir: str, test_data_scaled: dict) -> None:
        """Save each test event as an individual NPZ file for event-level evaluation."""
        test_dir = os.path.join(asset_dir, "test_by_event")
        os.makedirs(test_dir, exist_ok=True)
        for event_id, data in test_data_scaled.items():
            path = os.path.join(test_dir, f"event_{event_id}.npz")
            np.savez_compressed(
                path,
                X=data["X"],
                y=data["y"],
                label=data["label"],
                time_stamps=data.get("time_stamps", []),
                asset_id=data.get("asset_id", -1),
            )
        print(f"  Saved {len(test_data_scaled)} test events → {test_dir}")

    def run(self) -> dict:
        """
        Execute the full per-asset pipeline for all assets.

        Returns:
            Summary dict {asset_id: {train_seq, val_seq, test_events, n_features}}.
        """
        print("=" * 70)
        print("Per-Asset Data Preparation Pipeline")
        print("=" * 70)
        print(f"Output root: {self.output_dir}")

        event_info = self._event_loader.load_event_info()
        splitter   = DataSplitter(datasets_dir=self.datasets_dir, event_info=event_info)
        asset_groups = splitter.group_events_by_asset()

        summary = {}

        for asset_id, event_ids in asset_groups.items():
            print(f"\n{'='*70}")
            print(f"Asset {asset_id}  ({len(event_ids)} events: {event_ids})")
            print(f"{'='*70}")

            asset_dir = os.path.join(self.output_dir, f"asset_{asset_id}")
            os.makedirs(asset_dir, exist_ok=True)

            # --- Train data ---
            train_raw, feature_cols, contributions = splitter.process_asset_train(
                asset_id, event_ids
            )
            if train_raw is None:
                print(f"  [SKIP] Asset {asset_id}: no training data, skipping.")
                continue

            # --- Test data ---
            test_data = splitter.process_asset_test(asset_id, event_ids, feature_cols)

            # --- Train/Val temporal split ---
            train_split, val_split = splitter.temporal_split_train_val(
                train_raw, val_ratio=self.val_size
            )

            # --- Sequence creation ---
            X_train, y_train = self._seq_maker.create_sequences(train_split)
            X_val,   y_val   = self._seq_maker.create_sequences(val_split)

            if len(X_train) == 0:
                print(f"  [SKIP] Asset {asset_id}: no sequences generated (window too large?).")
                continue

            print(f"\n  Sequences — Train: {X_train.shape}, Val: {X_val.shape}")

            # --- Normalization (per-asset scaler) ---
            normalizer = AssetNormalizer(output_dir=asset_dir, seq_len=self.window_size)
            X_train_sc, X_val_sc, y_train_sc, y_val_sc, test_scaled = \
                normalizer.normalize_asset(
                    asset_id=asset_id,
                    X_train=X_train, X_val=X_val,
                    y_train=y_train, y_val=y_val,
                    test_data_dict=test_data,
                )

            # --- Save train/val arrays ---
            np.save(os.path.join(asset_dir, "X_train.npy"), X_train_sc)
            np.save(os.path.join(asset_dir, "X_val.npy"),   X_val_sc)
            np.save(os.path.join(asset_dir, "y_train.npy"), y_train_sc)
            np.save(os.path.join(asset_dir, "y_val.npy"),   y_val_sc)

            # --- Save test events ---
            self._save_test_events(asset_dir, test_scaled)

            # --- Save metadata ---
            metadata = {
                "asset_id": asset_id,
                "event_ids": event_ids,
                "feature_cols": feature_cols,
                "n_features": X_train_sc.shape[2],
                "window_size": self.window_size,
                "stride": self.stride,
                "n_train_seq": len(X_train_sc),
                "n_val_seq":   len(X_val_sc),
                "n_test_events": len(test_scaled),
                "event_contributions": contributions,
            }
            joblib.dump(metadata, os.path.join(asset_dir, "metadata.pkl"))

            summary[asset_id] = {
                "train_seq":   len(X_train_sc),
                "val_seq":     len(X_val_sc),
                "test_events": len(test_scaled),
                "n_features":  X_train_sc.shape[2],
            }
            print(f"\n  Asset {asset_id} saved → {asset_dir}")

        # --- Final summary table ---
        print(f"\n{'='*70}")
        print("PREPARATION COMPLETE")
        print(f"{'='*70}")
        print(f"{'Asset':<10} {'Train Seq':>12} {'Val Seq':>10} {'Test Events':>13} {'Features':>10}")
        print("-" * 57)
        for aid, s in summary.items():
            print(
                f"{aid:<10} {s['train_seq']:>12,} {s['val_seq']:>10,} "
                f"{s['test_events']:>13} {s['n_features']:>10}"
            )
        print(f"\nAll per-asset data saved under: {self.output_dir}")
        print("Next: python -m src.training.scripts.train_lstm --per_asset")

        return summary

    def run_from_csv(self, csv_path: str, asset_name: str = None, label: str = "normal") -> str:
        """
        Prepare data from a single raw SCADA CSV file and save it as a
        standalone asset directory — ready for training immediately after.

        No event_info.csv required.  The whole CSV is treated as training
        data; an 85/15 temporal train/val split is applied automatically.
        A small held-out portion at the end is kept as a synthetic test event.

        Args:
            csv_path:   Absolute path to the raw SCADA CSV file.
            asset_name: Name used for the output folder (default: CSV stem).
            label:      Event label for the synthetic test event —
                        'normal' or 'anomaly' (default: 'normal').

        Returns:
            Path to the created asset directory.
        """
        import pandas as pd
        from data_pipeline.preprocessing.feature_engineering import FeatureEngineer

        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        if asset_name is None:
            asset_name = os.path.splitext(os.path.basename(csv_path))[0]
            
        # Try to auto-detect the label from event_info.csv if the asset_name is a number
        if asset_name.isdigit():
            try:
                event_df = self._event_loader.load_event_info()
                event_id_int = int(asset_name)
                # Check if this event ID exists in the DataFrame
                row = event_df[event_df['event_id'] == event_id_int]
                if not row.empty:
                    label = row.iloc[0]['event_label']
            except Exception as e:
                print(f"  [WARN] Could not auto-detect label from event_info.csv: {e}")

        print("=" * 70)
        print(f"CSV Pipeline — {asset_name} (Label: {label})")
        print(f"  Source : {csv_path}")
        print("=" * 70)

        # 1. Load CSV — auto-detect delimiter (supports comma or semicolon)
        df = pd.read_csv(csv_path, sep=None, engine="python")
        print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

        # 2. Feature engineering (same transforms as the normal pipeline)
        fe           = FeatureEngineer()
        df           = fe.engineer_angle_features(df)
        df           = fe.drop_counter_features(df)
        feature_cols = fe.get_feature_columns(df)
        arr          = fe.preprocess_features(df, feature_cols).astype("float32")
        print(f"  After feature engineering: {arr.shape[1]} features")

        # 3. Temporal split — last 10 % becomes a synthetic test event
        test_frac   = 0.10
        test_cut    = int(len(arr) * (1 - test_frac))
        train_full  = arr[:test_cut]
        test_arr    = arr[test_cut:]

        # 4. Train / Val split from the training portion
        val_cut     = int(len(train_full) * (1 - self.val_size))
        train_split = train_full[:val_cut]
        val_split   = train_full[val_cut:]
        print(f"  Split  → train {len(train_split):,} | val {len(val_split):,} | test {len(test_arr):,}")

        # 5. Sequence creation (train + val only; normalizer does test sequencing internally)
        X_train, y_train = self._seq_maker.create_sequences(train_split)
        X_val,   y_val   = self._seq_maker.create_sequences(val_split)

        if len(X_train) == 0:
            raise RuntimeError(
                f"No training sequences generated — CSV has fewer than "
                f"{self.window_size} rows after feature engineering."
            )
        print(f"  Sequences — Train: {X_train.shape}, Val: {X_val.shape}")

        # 6. Normalisation (fit on train only)
        #    normalizer.normalize_asset expects test_data_dict entries to have
        #    keys: features (raw array), label, event_start, event_end
        asset_dir  = os.path.join(self.output_dir, f"asset_{asset_name}")
        os.makedirs(asset_dir, exist_ok=True)

        normalizer = AssetNormalizer(output_dir=asset_dir, seq_len=self.window_size)
        test_dict  = {asset_name: {
            "features":    test_arr,
            "time_stamps": df["time_stamp"].astype(str).tolist()[test_cut:],
            "label":       label,
            "event_start": 0,
            "event_end":   len(test_arr) - 1,
        }}
        X_tr_sc, X_val_sc, y_tr_sc, y_val_sc, test_scaled = normalizer.normalize_asset(
            asset_id=asset_name,
            X_train=X_train, X_val=X_val,
            y_train=y_train, y_val=y_val,
            test_data_dict=test_dict,
        )

        # 7. Save arrays
        np.save(os.path.join(asset_dir, "X_train.npy"), X_tr_sc)
        np.save(os.path.join(asset_dir, "X_val.npy"),   X_val_sc)
        np.save(os.path.join(asset_dir, "y_train.npy"), y_tr_sc)
        np.save(os.path.join(asset_dir, "y_val.npy"),   y_val_sc)
        self._save_test_events(asset_dir, test_scaled)

        # 8. Metadata
        metadata = {
            "asset_id":    asset_name,
            "source_csv":  csv_path,
            "label":       label,
            "feature_cols": feature_cols,
            "n_features":  X_tr_sc.shape[2],
            "window_size": self.window_size,
            "stride":      self.stride,
            "n_train_seq": len(X_tr_sc),
            "n_val_seq":   len(X_val_sc),
            "n_test_events": len(test_scaled),
        }
        joblib.dump(metadata, os.path.join(asset_dir, "metadata.pkl"))

        print(f"\n  Asset '{asset_name}' saved → {asset_dir}")
        print(f"  Next: python src/main.py train --model lstm --assets {asset_name}")
        return asset_dir

def main() -> None:
    pipeline = PerAssetPipeline(
        farm_dir=WIND_FARM_A_DIR,
        datasets_dir=WIND_FARM_A_DATASETS,
        output_dir=PER_ASSET_PROCESSED_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        val_size=VAL_SIZE,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
