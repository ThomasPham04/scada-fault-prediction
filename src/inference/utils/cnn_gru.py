"""
CNN-GRU inference utilities for sequence-based fault classification.

This module supports two common paths:
1. Predict directly from already-built window tensors.
2. Predict from a raw / processed pandas DataFrame by:
   - selecting the final feature columns
   - optionally applying angle engineering
   - filling missing values
   - optionally applying saved scaler(s)
   - building sliding windows
   - running the saved Keras classifier

Example:
    from inference.utils import run_cnn_gru_inference_from_dataframe
    import pandas as pd

    df = pd.read_csv("results/results/df_final.csv")
    outputs = run_cnn_gru_inference_from_dataframe(
        df=df[df["asset_id"].eq(10)].copy(),
        scaler_dir="sequence_exports/window_36h/classifier/scalers",
        threshold=0.90,
    )

    print(outputs["window_predictions"].head())
    print(outputs["event_predictions"].head())
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODELS_DIR, RESULTS_DIR  # noqa: E402
from data_pipeline.preprocessing.feature_engineering import FeatureEngineer  # noqa: E402
from training.hyperparameters.inference_defaults import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_STRIDE_STEPS,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_HOURS,
)
from training.hyperparameters.sequence_defaults import TIME_RESOLUTION  # noqa: E402


DEFAULT_MODEL_PATH = Path(MODELS_DIR) / "CNN_GRU" / "model.keras"
DEFAULT_FEATURE_FILE = Path(RESULTS_DIR) / "results" / "final_features.csv"
_FEATURE_ENGINEER = FeatureEngineer()


def window_steps_from_hours(
    window_hours: int,
    time_resolution_minutes: int = TIME_RESOLUTION,
) -> int:
    """Convert a window size in hours to sequence timesteps."""
    return int(window_hours * 60 / time_resolution_minutes)


def load_feature_columns(feature_file: str | Path = DEFAULT_FEATURE_FILE) -> list[str]:
    """
    Load the ordered feature list used by the sequence classifier.

    The expected file is the project's selected-feature CSV, typically with
    a `final_feature` column.
    """
    feature_path = Path(feature_file)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    feature_df = pd.read_csv(feature_path)
    if feature_df.empty:
        raise ValueError(f"Feature file is empty: {feature_path}")

    if "final_feature" in feature_df.columns:
        series = feature_df["final_feature"]
    else:
        series = feature_df.iloc[:, 0]

    feature_cols = [str(value) for value in series.dropna().tolist()]
    if not feature_cols:
        raise ValueError(f"No usable feature names found in: {feature_path}")

    return feature_cols


def load_cnn_gru_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the saved CNN-GRU classifier."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def load_threshold(
    threshold: float | None = None,
    threshold_path: str | Path | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """
    Resolve the score threshold used to convert probabilities into labels.

    Priority:
    1. Explicit `threshold`
    2. JSON file containing `selected_threshold` or `threshold`
    3. `default_threshold`
    """
    if threshold is not None:
        return float(threshold)

    if threshold_path is not None:
        threshold_path = Path(threshold_path)
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
        with threshold_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "selected_threshold" in payload:
            return float(payload["selected_threshold"])
        if "threshold" in payload:
            return float(payload["threshold"])
        raise ValueError(
            f"Threshold file does not contain `selected_threshold` or `threshold`: {threshold_path}"
        )

    return float(default_threshold)


def load_scaler_bundle(
    scaler_path: str | Path | None = None,
    scaler_dir: str | Path | None = None,
):
    """
    Load either one scaler or a per-asset scaler dictionary.

    Returns:
        scaler: single sklearn-style scaler or None
        scaler_map: dict[int, scaler] or None
    """
    if scaler_path and scaler_dir:
        raise ValueError("Use either `scaler_path` or `scaler_dir`, not both.")

    if scaler_path is not None:
        scaler_path = Path(scaler_path)
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        return joblib.load(scaler_path), None

    if scaler_dir is not None:
        scaler_dir = Path(scaler_dir)
        if not scaler_dir.exists():
            raise FileNotFoundError(f"Scaler directory not found: {scaler_dir}")

        scaler_map = {}
        for scaler_file in sorted(scaler_dir.glob("asset_*.pkl")):
            try:
                asset_id = int(scaler_file.stem.replace("asset_", ""))
            except ValueError:
                continue
            scaler_map[asset_id] = joblib.load(scaler_file)

        if not scaler_map:
            raise ValueError(f"No per-asset scaler files found in: {scaler_dir}")
        return None, scaler_map

    return None, None


def _fill_feature_gaps(
    frame: pd.DataFrame,
    feature_cols: list[str],
    group_cols: Iterable[str],
) -> pd.DataFrame:
    """Forward/backward-fill feature gaps inside each logical sequence group."""
    frame = frame.copy()
    usable_group_cols = [col for col in group_cols if col in frame.columns]

    if usable_group_cols:
        frame[feature_cols] = (
            frame.groupby(usable_group_cols, group_keys=False)[feature_cols]
            .apply(lambda part: part.ffill().bfill().fillna(0.0))
        )
    else:
        frame[feature_cols] = frame[feature_cols].ffill().bfill().fillna(0.0)

    frame[feature_cols] = frame[feature_cols].astype(np.float32)
    return frame


def _apply_scaling(
    frame: pd.DataFrame,
    feature_cols: list[str],
    scaler=None,
    scaler_map: dict[int, object] | None = None,
) -> pd.DataFrame:
    """Apply either one global scaler or a per-asset scaler map."""
    frame = frame.copy()

    if scaler is not None:
        frame.loc[:, feature_cols] = scaler.transform(
            frame[feature_cols].to_numpy(dtype=np.float32)
        )
        return frame

    if scaler_map is None:
        return frame

    if "asset_id" not in frame.columns:
        raise ValueError("Per-asset scaling requires an `asset_id` column.")

    transformed_parts = []
    for asset_id, asset_rows in frame.groupby("asset_id", sort=False):
        asset_id = int(asset_id)
        if asset_id not in scaler_map:
            raise KeyError(f"No scaler found for asset_id={asset_id}.")
        asset_rows = asset_rows.copy()
        asset_rows.loc[:, feature_cols] = scaler_map[asset_id].transform(
            asset_rows[feature_cols].to_numpy(dtype=np.float32)
        )
        transformed_parts.append(asset_rows)

    transformed = pd.concat(transformed_parts, axis=0)
    return transformed.sort_index(kind="mergesort")


def prepare_inference_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    feature_file: str | Path = DEFAULT_FEATURE_FILE,
    time_col: str = "time_stamp",
    group_cols: tuple[str, ...] = ("asset_id", "sequence_id"),
    scaler_path: str | Path | None = None,
    scaler_dir: str | Path | None = None,
    scaler=None,
    scaler_map: dict[int, object] | None = None,
) -> pd.DataFrame:
    """
    Prepare a DataFrame for sequence-model inference.

    Notes:
    - If engineered sin/cos columns are missing but raw angle columns exist,
      angle engineering is applied automatically.
    - Scaling is optional in code, but for reliable predictions you should use
      the same scaler(s) that were used during training.
    """
    if feature_cols is None:
        feature_cols = load_feature_columns(feature_file)

    frame = df.copy()

    if time_col in frame.columns:
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        sort_cols = [col for col in list(group_cols) + [time_col] if col in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    missing_features = [col for col in feature_cols if col not in frame.columns]
    if missing_features:
        frame = _FEATURE_ENGINEER.engineer_angle_features(frame)
        missing_features = [col for col in feature_cols if col not in frame.columns]

    if missing_features:
        raise ValueError(
            "The following required feature columns are missing after preprocessing: "
            + ", ".join(missing_features)
        )

    frame = _fill_feature_gaps(frame, feature_cols, group_cols)

    loaded_scaler, loaded_scaler_map = load_scaler_bundle(
        scaler_path=scaler_path,
        scaler_dir=scaler_dir,
    )

    scaler = scaler if scaler is not None else loaded_scaler
    scaler_map = scaler_map if scaler_map is not None else loaded_scaler_map
    frame = _apply_scaling(frame, feature_cols, scaler=scaler, scaler_map=scaler_map)

    return frame


def build_sequence_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_steps: int,
    stride_steps: int = DEFAULT_STRIDE_STEPS,
    group_cols: tuple[str, ...] = ("asset_id", "sequence_id"),
    time_col: str = "time_stamp",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Create sliding sequence windows and aligned metadata from a DataFrame.
    """
    window_rows = []
    meta_rows = []
    feature_count = len(feature_cols)
    usable_group_cols = [col for col in group_cols if col in df.columns]

    if usable_group_cols:
        grouped_iter = df.groupby(usable_group_cols, sort=False)
    else:
        grouped_iter = [(("full_frame",), df)]

    for group_key, group in grouped_iter:
        group = group.copy()
        if time_col in group.columns:
            group = group.sort_values(time_col, kind="mergesort")

        if len(group) < window_steps:
            continue

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        features = group[feature_cols].to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=np.int8) if "label" in group.columns else None

        for start in range(0, len(group) - window_steps + 1, stride_steps):
            end = start + window_steps
            window_rows.append(features[start:end])

            start_row = group.iloc[start]
            end_row = group.iloc[end - 1]

            meta = {
                "window_index": len(meta_rows),
                "window_steps": int(window_steps),
                "start_row_index": int(group.index[start]),
                "end_row_index": int(group.index[end - 1]),
            }

            for idx, col in enumerate(usable_group_cols):
                meta[col] = group_key[idx]

            if time_col in group.columns:
                meta["start_time"] = start_row[time_col]
                meta["end_time"] = end_row[time_col]

            if labels is not None:
                meta["label_last"] = int(labels[end - 1])
                meta["label_any"] = int(labels[start:end].max())

            meta_rows.append(meta)

    if not window_rows:
        empty_x = np.empty((0, window_steps, feature_count), dtype=np.float32)
        return empty_x, pd.DataFrame(columns=["window_index", "window_steps"])

    X = np.asarray(window_rows, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df


def run_cnn_gru_inference_from_array(
    X: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
    threshold_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Run inference on an already-built 3D sequence array.
    """
    model = load_cnn_gru_model(model_path)
    score_threshold = load_threshold(threshold=threshold, threshold_path=threshold_path)

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(
            f"`X` must be a 3D array shaped (n_windows, timesteps, n_features), got {X.shape}."
        )

    scores = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    predictions = (scores >= score_threshold).astype(np.int8)

    return pd.DataFrame(
        {
            "window_index": np.arange(len(scores), dtype=int),
            "score": scores.astype(np.float32),
            "predicted_label": predictions,
            "threshold": float(score_threshold),
        }
    )


def aggregate_event_predictions(
    window_predictions: pd.DataFrame,
    threshold: float,
    event_cols: tuple[str, ...] = ("asset_id", "sequence_id"),
) -> pd.DataFrame:
    """
    Aggregate window scores to event-level predictions using max score.
    """
    usable_event_cols = [col for col in event_cols if col in window_predictions.columns]
    if not usable_event_cols:
        return pd.DataFrame()

    agg_map = {"score": "max"}
    if "label_last" in window_predictions.columns:
        agg_map["label_last"] = "max"
    if "label_any" in window_predictions.columns:
        agg_map["label_any"] = "max"
    if "start_time" in window_predictions.columns:
        agg_map["start_time"] = "min"
    if "end_time" in window_predictions.columns:
        agg_map["end_time"] = "max"

    event_df = (
        window_predictions.groupby(usable_event_cols, as_index=False)
        .agg(agg_map)
        .rename(columns={"score": "event_score"})
    )
    event_df["predicted_label"] = (
        event_df["event_score"].to_numpy(dtype=np.float32) >= float(threshold)
    ).astype(np.int8)
    event_df["threshold"] = float(threshold)
    return event_df


def run_cnn_gru_inference_from_dataframe(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
    threshold_path: str | Path | None = None,
    feature_cols: list[str] | None = None,
    feature_file: str | Path = DEFAULT_FEATURE_FILE,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    window_steps: int | None = None,
    stride_steps: int = DEFAULT_STRIDE_STEPS,
    group_cols: tuple[str, ...] = ("asset_id", "sequence_id"),
    time_col: str = "time_stamp",
    scaler_path: str | Path | None = None,
    scaler_dir: str | Path | None = None,
    scaler=None,
    scaler_map: dict[int, object] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    """
    Full DataFrame-to-prediction inference pipeline for the saved CNN-GRU model.
    """
    if feature_cols is None:
        feature_cols = load_feature_columns(feature_file)

    prepared_df = prepare_inference_dataframe(
        df=df,
        feature_cols=feature_cols,
        feature_file=feature_file,
        time_col=time_col,
        group_cols=group_cols,
        scaler_path=scaler_path,
        scaler_dir=scaler_dir,
        scaler=scaler,
        scaler_map=scaler_map,
    )

    if window_steps is None:
        window_steps = window_steps_from_hours(window_hours)

    X, meta_df = build_sequence_windows(
        df=prepared_df,
        feature_cols=feature_cols,
        window_steps=window_steps,
        stride_steps=stride_steps,
        group_cols=group_cols,
        time_col=time_col,
    )

    prediction_df = run_cnn_gru_inference_from_array(
        X=X,
        model_path=model_path,
        threshold=threshold,
        threshold_path=threshold_path,
        batch_size=batch_size,
    )

    if not meta_df.empty:
        prediction_columns = [col for col in prediction_df.columns if col != "window_index"]
        window_predictions = pd.concat(
            [
                meta_df.reset_index(drop=True),
                prediction_df[prediction_columns].reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        window_predictions = prediction_df.copy()

    resolved_threshold = float(window_predictions["threshold"].iloc[0]) if not window_predictions.empty else load_threshold(
        threshold=threshold,
        threshold_path=threshold_path,
    )
    event_predictions = aggregate_event_predictions(
        window_predictions=window_predictions,
        threshold=resolved_threshold,
    )

    return {
        "prepared_df": prepared_df,
        "feature_cols": feature_cols,
        "window_steps": int(window_steps),
        "stride_steps": int(stride_steps),
        "X": X,
        "window_predictions": window_predictions,
        "event_predictions": event_predictions,
        "threshold": resolved_threshold,
    }
