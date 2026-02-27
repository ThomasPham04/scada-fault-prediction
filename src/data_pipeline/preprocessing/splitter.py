"""
Event Splitter — data_pipeline.preprocessing.splitter
Processes all 22 events into train/test splits following the CARE paper methodology.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import (
    NBM_WINDOW_SIZE,
)
from data_pipeline.loaders import load_event_data
from data_pipeline.preprocessing.feature_engineering import (
    engineer_angle_features,
    drop_counter_features,
    get_feature_columns,
    preprocess_features,
)


def process_all_events_train(
    farm_dir: str,
    datasets_dir: str,
    event_info: pd.DataFrame,
) -> tuple:
    """
    Process TRAIN data from ALL events using CARE methodology.

    For each event:
      - Combine all into one large training array

    Args:
        farm_dir: Path to Wind Farm A directory
        datasets_dir: Path to datasets CSV directory
        event_info: DataFrame from load_event_info()

    Returns:
        Tuple of (combined_train_array, feature_cols, event_contributions_dict)
    """
    all_train_data = []
    feature_cols = None
    event_contributions = {}

    print(f"\n{'='*70}")
    print(f"Processing TRAIN data from ALL {len(event_info)} events...")
    print(f"{'='*70}")

    for _, event in tqdm(event_info.iterrows(), total=len(event_info)):
        event_id = event['event_id']
        event_label = event['event_label']

        try:
            df = load_event_data(event_id, datasets_dir)
            df_train = df[df['train_test'] == 'train'].copy()

            if len(df_train) == 0:
                print(f"  Event {event_id} ({event_label}): No train data")
                continue

            df_normal = df_train.copy()

            if len(df_normal) < NBM_WINDOW_SIZE + 1:
                print(
                    f"  Event {event_id} ({event_label}): Insufficient normal data "
                    f"({len(df_normal)} timesteps, need {NBM_WINDOW_SIZE + 1})"
                )
                continue

            df_normal = engineer_angle_features(df_normal)
            df_normal = drop_counter_features(df_normal)

            if feature_cols is None:
                feature_cols = get_feature_columns(df_normal)
                print(f"  Using {len(feature_cols)} features after engineering")

            features = preprocess_features(df_normal, feature_cols)
            all_train_data.append(features)
            event_contributions[event_id] = {
                'label': event_label,
                'original_train_len': len(df_train),
                'filtered_len': len(df_normal),
                'features_shape': features.shape,
            }

            print(
                f"  Event {event_id} ({event_label:7s}): "
                f"{len(df_train):5d} → {len(df_normal):5d} timesteps "
                f"({len(df_normal) / len(df_train) * 100:.1f}% retained)"
            )

        except Exception as e:
            print(f"  Event {event_id}: ERROR - {e}")
            continue

    if len(all_train_data) == 0:
        raise ValueError("No training data generated from any event!")

    combined_train = np.concatenate(all_train_data, axis=0)

    print(f"\n{'='*70}")
    print(
        f"Combined training data: {combined_train.shape[0]} timesteps "
        f"from {len(all_train_data)} events"
    )
    print(f"{'='*70}")

    return combined_train, feature_cols, event_contributions


def process_all_events_test(
    farm_dir: str,
    datasets_dir: str,
    event_info: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Process TEST (prediction period) data from ALL events.

    - Takes train_test == 'prediction' portion
    - Keeps ALL status types (including anomalies — do NOT filter)

    Args:
        farm_dir: Path to Wind Farm A directory
        datasets_dir: Path to datasets CSV directory
        event_info: DataFrame from load_event_info()
        feature_cols: Feature columns determined during train processing

    Returns:
        Dict of {event_id: {'features', 'label', 'n_timesteps', 'event_start', 'event_end'}}
    """
    test_data_by_event = {}

    print(f"\n{'='*70}")
    print(f"Processing TEST (prediction) data from ALL {len(event_info)} events...")
    print(f"  NOTE: Keeping ALL status types (including anomalies)")
    print(f"{'='*70}")

    for _, event in tqdm(event_info.iterrows(), total=len(event_info)):
        event_id = event['event_id']
        event_label = event['event_label']

        try:
            df = load_event_data(event_id, datasets_dir)
            df_pred = df[df['train_test'] == 'prediction'].copy()

            if len(df_pred) == 0:
                print(f"  Event {event_id} ({event_label}): No prediction data")
                continue

            if len(df_pred) < NBM_WINDOW_SIZE + 1:
                print(
                    f"  Event {event_id} ({event_label}): Insufficient prediction data "
                    f"({len(df_pred)} timesteps, need {NBM_WINDOW_SIZE + 1})"
                )
                continue

            df_pred = engineer_angle_features(df_pred)
            df_pred = drop_counter_features(df_pred)
            features = preprocess_features(df_pred, feature_cols)

            test_data_by_event[event_id] = {
                'features': features,
                'label': event_label,
                'n_timesteps': len(df_pred),
                'event_start': event['event_start'],
                'event_end': event['event_end'],
            }

            print(f"  Event {event_id} ({event_label:7s}): {len(df_pred)} timesteps")

        except Exception as e:
            print(f"  Event {event_id}: ERROR - {e}")
            continue

    print(f"\n{'='*70}")
    print(f"Processed {len(test_data_by_event)} events for testing")
    print(f"{'='*70}")

    return test_data_by_event


def temporal_split_train_val(train_data: np.ndarray, val_ratio: float = 0.15) -> tuple:
    """
    Temporal train/val split — takes the LAST val_ratio% as validation.

    Better than random split for time-series data: preserves temporal
    ordering and avoids leakage from future into the past.

    Args:
        train_data: Combined training array (n_timesteps, n_features)
        val_ratio: Fraction reserved for validation (default 15%)

    Returns:
        Tuple of (train_split, val_split)
    """
    n_total = len(train_data)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size

    train_split = train_data[:train_size]
    val_split = train_data[train_size:]

    print(f"\nTemporal train/val split:")
    print(f"  Train: {len(train_split)} timesteps ({len(train_split) / n_total * 100:.1f}%)")
    print(f"  Val:   {len(val_split)} timesteps ({len(val_split) / n_total * 100:.1f}%)")

    return train_split, val_split
