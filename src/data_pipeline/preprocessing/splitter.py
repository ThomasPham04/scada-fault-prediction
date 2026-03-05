"""
DataSplitter — data_pipeline.preprocessing.splitter
Processes all 22 events into train/test splits following the CARE paper methodology.

Supports two modes:
  - Global: all events combined → one model  (original)
  - Per-asset: events grouped by turbine → one model per asset  (new)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import WINDOW_SIZE
from data_pipeline.loaders.event_loader import EventLoader
from data_pipeline.preprocessing.feature_engineering import FeatureEngineer


class DataSplitter:
    """
    Splits and prepares SCADA event data into train/test arrays
    following the CARE to Compare paper methodology.

    Args:
        datasets_dir: Path to the raw datasets directory
                      (CSV files named <event_id>.csv).
        event_info: DataFrame returned by EventLoader.load_event_info().
    """

    def __init__(self, datasets_dir: str, event_info: pd.DataFrame) -> None:
        self.datasets_dir = datasets_dir
        self.event_info   = event_info
        self._loader      = EventLoader(farm_dir="", datasets_dir=datasets_dir)
        self._engineer    = FeatureEngineer()

    # ------------------------------------------------------------------
    # Global pipeline helpers
    # ------------------------------------------------------------------

    def process_all_events_train(self, farm_dir: str = "") -> tuple:
        """
        Process TRAIN data from ALL events (CARE methodology).

        For each event: load train rows → engineer features → concatenate
        into one large training array.

        Returns:
            Tuple of (combined_train_array, feature_cols, event_contributions_dict).
        """
        all_train_data = []
        feature_cols   = None
        event_contributions = {}

        print(f"\n{'='*70}")
        print(f"Processing TRAIN data from ALL {len(self.event_info)} events...")
        print(f"{'='*70}")

        for _, event in tqdm(self.event_info.iterrows(), total=len(self.event_info)):
            event_id    = event["event_id"]
            event_label = event["event_label"]

            try:
                df = self._loader.load_event_data(event_id)
                df_train = df[df["train_test"] == "train"].copy()

                if len(df_train) == 0:
                    print(f"  Event {event_id} ({event_label}): No train data")
                    continue

                if len(df_train) < WINDOW_SIZE + 1:
                    print(
                        f"  Event {event_id} ({event_label}): Insufficient normal data "
                        f"({len(df_train)} timesteps, need {WINDOW_SIZE + 1})"
                    )
                    continue

                df_train = self._engineer.engineer_angle_features(df_train)
                df_train = self._engineer.drop_counter_features(df_train)

                if feature_cols is None:
                    feature_cols = self._engineer.get_feature_columns(df_train)
                    print(f"  Using {len(feature_cols)} features after engineering")

                features = self._engineer.preprocess_features(df_train, feature_cols)
                all_train_data.append(features)
                event_contributions[event_id] = {
                    "label":             event_label,
                    "original_train_len": len(df_train),
                    "filtered_len":       len(df_train),
                    "features_shape":     features.shape,
                }
                print(
                    f"  Event {event_id} ({event_label:7s}): "
                    f"{len(df_train):5d} timesteps"
                )

            except Exception as e:
                print(f"  Event {event_id}: ERROR - {e}")
                continue

        if not all_train_data:
            raise ValueError("No training data generated from any event!")

        combined = np.concatenate(all_train_data, axis=0)
        print(f"\n{'='*70}")
        print(
            f"Combined training data: {combined.shape[0]} timesteps "
            f"from {len(all_train_data)} events"
        )
        print(f"{'='*70}")

        return combined, feature_cols, event_contributions

    def process_all_events_test(
        self,
        farm_dir: str = "",
        feature_cols: list = None,
    ) -> dict:
        """
        Process TEST (prediction) data from ALL events.

        Takes the train_test == 'prediction' portion.
        Keeps ALL status types (including anomalies — do NOT filter).

        Args:
            farm_dir: Unused (kept for API compatibility).
            feature_cols: Feature columns determined during train processing.

        Returns:
            Dict of {event_id: {'features', 'label', 'n_timesteps',
                                'event_start', 'event_end'}}.
        """
        test_data = {}

        print(f"\n{'='*70}")
        print(f"Processing TEST (prediction) data from ALL {len(self.event_info)} events...")
        print(f"  NOTE: Keeping ALL status types (including anomalies)")
        print(f"{'='*70}")

        for _, event in tqdm(self.event_info.iterrows(), total=len(self.event_info)):
            event_id    = event["event_id"]
            event_label = event["event_label"]

            try:
                df = self._loader.load_event_data(event_id)
                df_pred = df[df["train_test"] == "prediction"].copy()

                if len(df_pred) == 0:
                    print(f"  Event {event_id} ({event_label}): No prediction data")
                    continue

                if len(df_pred) < WINDOW_SIZE + 1:
                    print(
                        f"  Event {event_id} ({event_label}): Insufficient prediction data "
                        f"({len(df_pred)} timesteps, need {WINDOW_SIZE + 1})"
                    )
                    continue

                df_pred  = self._engineer.engineer_angle_features(df_pred)
                df_pred  = self._engineer.drop_counter_features(df_pred)
                features = self._engineer.preprocess_features(df_pred, feature_cols)

                test_data[event_id] = {
                    "features":    features,
                    "label":       event_label,
                    "n_timesteps": len(df_pred),
                    "event_start": event["event_start"],
                    "event_end":   event["event_end"],
                }
                print(f"  Event {event_id} ({event_label:7s}): {len(df_pred)} timesteps")

            except Exception as e:
                print(f"  Event {event_id}: ERROR - {e}")
                continue

        print(f"\n{'='*70}")
        print(f"Processed {len(test_data)} events for testing")
        print(f"{'='*70}")

        return test_data

    def temporal_split_train_val(
        self,
        train_data: np.ndarray,
        val_ratio: float = 0.15,
    ) -> tuple:
        """
        Temporal train/val split — takes the LAST val_ratio% as validation.

        Better than random split for time-series: preserves temporal ordering
        and avoids leakage from future into the past.

        Args:
            train_data: Combined training array (n_timesteps, n_features).
            val_ratio: Fraction reserved for validation (default 15%).

        Returns:
            Tuple of (train_split, val_split).
        """
        n_total    = len(train_data)
        val_size   = int(n_total * val_ratio)
        train_size = n_total - val_size

        train_split = train_data[:train_size]
        val_split   = train_data[train_size:]

        print(f"\nTemporal train/val split:")
        print(f"  Train: {len(train_split)} timesteps ({len(train_split) / n_total * 100:.1f}%)")
        print(f"  Val:   {len(val_split)} timesteps ({len(val_split) / n_total * 100:.1f}%)")

        return train_split, val_split

    # ------------------------------------------------------------------
    # Per-asset helpers
    # ------------------------------------------------------------------

    def group_events_by_asset(self) -> dict:
        """
        Group event IDs by their asset (turbine) ID.

        Returns:
            Dict of {asset_id: [event_id, ...]} ordered by asset_id.
        """
        ei = self.event_info
        asset_col = "asset_id" if "asset_id" in ei.columns else "asset"
        grouped = (
            ei.groupby(asset_col)["event_id"]
            .apply(list)
            .to_dict()
        )
        asset_ids = sorted(grouped.keys())
        print(f"\nFound {len(asset_ids)} assets: {asset_ids}")
        for aid in asset_ids:
            events  = grouped[aid]
            labels  = ei.set_index("event_id").loc[events, "event_label"].tolist()
            n_anom  = sum(1 for l in labels if l == "anomaly")
            print(
                f"  Asset {aid}: {len(events)} events "
                f"({n_anom} anomaly, {len(events) - n_anom} normal)"
            )
        return {aid: grouped[aid] for aid in asset_ids}

    def process_asset_train(
        self,
        asset_id: int,
        event_ids: list,
    ) -> tuple:
        """
        Process TRAIN data for a single asset — combines the training portions
        of all events belonging to that asset.

        Args:
            asset_id: Turbine/asset ID being processed.
            event_ids: List of event IDs for this asset.

        Returns:
            Tuple of (combined_train_array, feature_cols, contributions_dict).
            Returns (None, None, {}) if no usable training data is found.
        """
        all_train_data = []
        feature_cols   = None
        contributions  = {}
        ei = self.event_info.set_index("event_id")

        print(f"\n  Asset {asset_id} — processing {len(event_ids)} events for TRAIN")
        for event_id in event_ids:
            event_label = ei.loc[event_id, "event_label"]
            try:
                df = self._loader.load_event_data(event_id)
                df_train = df[df["train_test"] == "train"].copy()

                if len(df_train) == 0 or len(df_train) < WINDOW_SIZE + 1:
                    print(f"    Event {event_id}: skipped (only {len(df_train)} rows)")
                    continue

                df_train = self._engineer.engineer_angle_features(df_train)
                df_train = self._engineer.drop_counter_features(df_train)

                if feature_cols is None:
                    feature_cols = self._engineer.get_feature_columns(df_train)

                features = self._engineer.preprocess_features(df_train, feature_cols)
                all_train_data.append(features)
                contributions[event_id] = {
                    "label":       event_label,
                    "n_timesteps": len(df_train),
                    "shape":       features.shape,
                }
                print(f"    Event {event_id} ({event_label:7s}): {len(df_train)} train timesteps")

            except Exception as e:
                print(f"    Event {event_id}: ERROR — {e}")

        if not all_train_data:
            print(f"  Asset {asset_id}: no usable training data!")
            return None, None, {}

        combined = np.concatenate(all_train_data, axis=0)
        print(f"  Asset {asset_id}: total train array → {combined.shape}")
        return combined, feature_cols, contributions

    def process_asset_test(
        self,
        asset_id: int,
        event_ids: list,
        feature_cols: list,
    ) -> dict:
        """
        Process TEST (prediction) data for a single asset.

        Args:
            asset_id: Turbine/asset ID being processed.
            event_ids: List of event IDs for this asset.
            feature_cols: Feature columns from process_asset_train().

        Returns:
            Dict of {event_id: {'features', 'label', 'n_timesteps',
                                'event_start', 'event_end', 'asset_id'}}.
        """
        test_data = {}
        ei = self.event_info.set_index("event_id")

        print(f"\n  Asset {asset_id} — processing {len(event_ids)} events for TEST")
        for event_id in event_ids:
            event_label = ei.loc[event_id, "event_label"]
            try:
                df = self._loader.load_event_data(event_id)
                df_pred = df[df["train_test"] == "prediction"].copy()

                if len(df_pred) == 0 or len(df_pred) < WINDOW_SIZE + 1:
                    print(f"    Event {event_id}: skipped (only {len(df_pred)} rows)")
                    continue

                df_pred  = self._engineer.engineer_angle_features(df_pred)
                df_pred  = self._engineer.drop_counter_features(df_pred)
                features = self._engineer.preprocess_features(df_pred, feature_cols)

                test_data[event_id] = {
                    "features":    features,
                    "time_stamps": df_pred["time_stamp"].astype(str).tolist(),
                    "label":       event_label,
                    "n_timesteps": len(df_pred),
                    "event_start": ei.loc[event_id, "event_start"],
                    "event_end":   ei.loc[event_id, "event_end"],
                    "asset_id":    asset_id,
                }
                print(f"    Event {event_id} ({event_label:7s}): {len(df_pred)} test timesteps")

            except Exception as e:
                print(f"    Event {event_id}: ERROR — {e}")

        return test_data


# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases
# ---------------------------------------------------------------------------

def group_events_by_asset(event_info: pd.DataFrame) -> dict:
    """Legacy alias."""
    return DataSplitter(datasets_dir="", event_info=event_info).group_events_by_asset()


def process_asset_train(
    asset_id: int,
    event_ids: list,
    event_info: pd.DataFrame,
    datasets_dir: str,
) -> tuple:
    """Legacy alias."""
    return DataSplitter(datasets_dir=datasets_dir, event_info=event_info).process_asset_train(
        asset_id, event_ids
    )


def process_asset_test(
    asset_id: int,
    event_ids: list,
    event_info: pd.DataFrame,
    datasets_dir: str,
    feature_cols: list,
) -> dict:
    """Legacy alias."""
    return DataSplitter(datasets_dir=datasets_dir, event_info=event_info).process_asset_test(
        asset_id, event_ids, feature_cols
    )


def temporal_split_train_val(train_data: np.ndarray, val_ratio: float = 0.15) -> tuple:
    """Legacy alias."""
    return DataSplitter(datasets_dir="", event_info=pd.DataFrame()).temporal_split_train_val(
        train_data, val_ratio
    )


def process_all_events_train(
    farm_dir: str,
    datasets_dir: str,
    event_info: pd.DataFrame,
) -> tuple:
    """Legacy alias."""
    return DataSplitter(datasets_dir=datasets_dir, event_info=event_info).process_all_events_train(farm_dir)


def process_all_events_test(
    farm_dir: str,
    datasets_dir: str,
    event_info: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """Legacy alias."""
    return DataSplitter(datasets_dir=datasets_dir, event_info=event_info).process_all_events_test(
        farm_dir, feature_cols
    )
