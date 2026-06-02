"""
Combined CSV sequence preparation.

This module prepares sequence exports for the main project from one large CSV
that was built by concatenating many event CSV files while preserving
asset_id and sequence_id.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from training.hyperparameters.sequence_defaults import (
    DEFAULT_LABEL_MODE,
    DEFAULT_PREDICTION_VAL_RATIO,
    DEFAULT_PROBE_BATCH_SIZE,
    DEFAULT_PROBE_CALLBACK_MODE,
    DEFAULT_PROBE_CALLBACK_MONITOR,
    DEFAULT_PROBE_EARLY_STOPPING_PATIENCE,
    DEFAULT_PROBE_EPOCHS,
    DEFAULT_PROBE_GRU_UNITS,
    DEFAULT_PROBE_INITIAL_THRESHOLD,
    DEFAULT_PROBE_LEARNING_RATE,
    DEFAULT_PROBE_LOSS,
    DEFAULT_PROBE_MAX_TRAIN_WINDOWS,
    DEFAULT_PROBE_THRESHOLD_START,
    DEFAULT_PROBE_THRESHOLD_STEP,
    DEFAULT_PROBE_THRESHOLD_STOP,
    DEFAULT_SCALER_TYPE,
    DEFAULT_SELECTED_WINDOWS_HOURS,
    DEFAULT_TOP_K_WINDOWS,
    DEFAULT_VALIDATION_SOURCE,
    DEFAULT_WINDOW_CANDIDATES_HOURS,
    NORMAL_STATUS,
    PREDICTION_HORIZON_STEPS as DEFAULT_PREDICTION_HORIZON_STEPS,
    RANDOM_SEED,
    STRIDE as DEFAULT_STRIDE_STEPS,
    TIME_RESOLUTION,
    VAL_SIZE,
)

VALIDATION_SOURCES = {"train_tail", "prediction"}
LABEL_MODE_FUTURE_HORIZON = DEFAULT_LABEL_MODE
LABEL_MODE_INPUT_WINDOW = "input_window"
LABEL_MODE_ALIASES = {
    "future_horizon": LABEL_MODE_FUTURE_HORIZON,
    "prediction": LABEL_MODE_FUTURE_HORIZON,
    "horizon": LABEL_MODE_FUTURE_HORIZON,
    "input_window": LABEL_MODE_INPUT_WINDOW,
    "window": LABEL_MODE_INPUT_WINDOW,
    "window_max": LABEL_MODE_INPUT_WINDOW,
    "detection_window": LABEL_MODE_INPUT_WINDOW,
}
LABEL_DEFINITIONS = {
    LABEL_MODE_FUTURE_HORIZON: "future_horizon_any_positive_after_input_window",
    LABEL_MODE_INPUT_WINDOW: "input_window_any_positive_within_window",
}

METADATA_COLUMNS = {
    "time_stamp",
    "asset_id",
    "train_test",
    "status_type_id",
    "sequence_id",
    "label",
    "data_split",
}


def looks_like_combined_sequence_csv(csv_path: str | Path) -> bool:
    """Return True when a CSV has the schema needed for combined sequence export."""
    try:
        columns = set(pd.read_csv(csv_path, nrows=0, sep=None, engine="python").columns)
    except Exception:
        return False

    required = {"time_stamp", "asset_id", "train_test", "sequence_id"}
    has_label_source = "label" in columns or "status_type_id" in columns
    return required.issubset(columns) and has_label_source


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_metadata(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def empty_split_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns + ["data_split"])


class CombinedSequencePipeline:
    """
    Prepare classifier sequence exports from one combined CSV.

    The input CSV must preserve the original logical boundaries with asset_id
    and sequence_id. Windows are always built inside each (asset_id,
    sequence_id) group, never across concatenated event boundaries.
    """

    def __init__(
        self,
        csv_path: str | Path,
        feature_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        window_candidates_hours: Sequence[int] | None = None,
        selected_windows_hours: Sequence[int] | None = None,
        top_k_windows: int = DEFAULT_TOP_K_WINDOWS,
        stride_steps: int = DEFAULT_STRIDE_STEPS,
        prediction_horizon_steps: int | None = DEFAULT_PREDICTION_HORIZON_STEPS,
        val_ratio: float = VAL_SIZE,
        normal_statuses: Iterable[int] = NORMAL_STATUS,
        time_resolution_minutes: int = TIME_RESOLUTION,
        expected_feature_count: int | None = None,
        scaler_type: str = DEFAULT_SCALER_TYPE,
        validation_source: str = DEFAULT_VALIDATION_SOURCE,
        prediction_val_ratio: float = DEFAULT_PREDICTION_VAL_RATIO,
        label_mode: str = LABEL_MODE_FUTURE_HORIZON,
        run_window_search: bool = True,
        probe_epochs: int = DEFAULT_PROBE_EPOCHS,
        probe_batch_size: int = DEFAULT_PROBE_BATCH_SIZE,
        probe_max_train_windows: int = DEFAULT_PROBE_MAX_TRAIN_WINDOWS,
        random_seed: int = RANDOM_SEED,
        skip_classifier: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.feature_file = Path(feature_file) if feature_file else None
        self.output_dir = Path(output_dir) if output_dir else Path(PROCESSED_DATA_DIR) / "sequence_exports"
        self.window_candidates_hours = list(window_candidates_hours or DEFAULT_WINDOW_CANDIDATES_HOURS)
        self.selected_windows_hours = list(
            DEFAULT_SELECTED_WINDOWS_HOURS if selected_windows_hours is None else selected_windows_hours
        )
        self.top_k_windows = top_k_windows
        self.stride_steps = stride_steps
        self.prediction_horizon_steps = int(
            DEFAULT_PREDICTION_HORIZON_STEPS
            if prediction_horizon_steps is None
            else prediction_horizon_steps
        )
        self.val_ratio = val_ratio
        self.normal_statuses = set(int(x) for x in normal_statuses)
        self.time_resolution_minutes = time_resolution_minutes
        self.expected_feature_count = expected_feature_count
        self.scaler_type = scaler_type.lower()
        self.validation_source = validation_source.lower()
        self.prediction_val_ratio = float(prediction_val_ratio)
        self.label_mode = self._normalize_label_mode(label_mode)
        self.run_window_search = run_window_search
        self.probe_epochs = probe_epochs
        self.probe_batch_size = probe_batch_size
        self.probe_max_train_windows = probe_max_train_windows
        self.random_seed = random_seed
        self.skip_classifier = skip_classifier

        if self.scaler_type not in ("minmax", "standard"):
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        if self.validation_source not in VALIDATION_SOURCES:
            raise ValueError(
                f"validation_source must be one of {sorted(VALIDATION_SOURCES)}"
            )
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1.")
        if not 0.0 < self.prediction_val_ratio < 1.0:
            raise ValueError("prediction_val_ratio must be between 0 and 1.")
        if self.prediction_horizon_steps <= 0:
            raise ValueError("prediction_horizon_steps must be positive")

    @staticmethod
    def _normalize_label_mode(label_mode: str) -> str:
        normalized = str(label_mode).strip().lower()
        if normalized not in LABEL_MODE_ALIASES:
            raise ValueError(
                "label_mode must be one of "
                f"{sorted(LABEL_MODE_ALIASES)}"
            )
        return LABEL_MODE_ALIASES[normalized]

    @property
    def label_definition(self) -> str:
        return LABEL_DEFINITIONS[self.label_mode]

    @property
    def uses_future_horizon(self) -> bool:
        return self.label_mode == LABEL_MODE_FUTURE_HORIZON

    def window_steps_from_hours(self, window_hours: int) -> int:
        return int((window_hours * 60) / self.time_resolution_minutes)

    def _load_feature_columns(self, df: pd.DataFrame) -> list[str]:
        if self.feature_file is not None:
            if not self.feature_file.exists():
                raise FileNotFoundError(f"Feature file not found: {self.feature_file}")
            feature_df = pd.read_csv(self.feature_file)
            if feature_df.empty:
                raise ValueError(f"Feature file is empty: {self.feature_file}")

            feature_col_name = "final_feature" if "final_feature" in feature_df.columns else feature_df.columns[0]
            feature_cols = (
                feature_df[feature_col_name]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
        else:
            feature_cols = [
                col for col in df.columns
                if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
            ]

        if not feature_cols:
            raise ValueError("No feature columns were found for combined sequence export.")

        if self.expected_feature_count is not None and len(feature_cols) != self.expected_feature_count:
            raise ValueError(
                f"Expected {self.expected_feature_count} features, found {len(feature_cols)}."
            )

        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing selected features in combined CSV: {missing_features}")

        return feature_cols

    def load_inputs(self) -> tuple[pd.DataFrame, list[str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Combined CSV not found: {self.csv_path}")

        # Combined exports in this project are comma-separated; use the C
        # engine first because these files can be >1GB. Fall back to delimiter
        # detection only if the file was parsed as a single column.
        df = pd.read_csv(self.csv_path)
        if len(df.columns) == 1:
            df = pd.read_csv(self.csv_path, sep=None, engine="python")
        feature_cols = self._load_feature_columns(df)
        return df, feature_cols

    def prepare_base_dataframe(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        required_cols = {"time_stamp", "asset_id", "train_test", "sequence_id"}
        missing_required = sorted(required_cols - set(df.columns))
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        if "label" not in df.columns and "status_type_id" not in df.columns:
            raise ValueError("Combined CSV must contain either 'label' or 'status_type_id'.")

        prepared = df.copy()
        prepared["time_stamp"] = pd.to_datetime(prepared["time_stamp"], errors="coerce")
        if prepared["time_stamp"].isna().any():
            bad_count = int(prepared["time_stamp"].isna().sum())
            raise ValueError(f"Found {bad_count} invalid time_stamp values.")

        prepared["train_test"] = prepared["train_test"].astype(str).str.strip().str.lower()
        valid_train_test = {"train", "prediction"}
        invalid_values = sorted(set(prepared["train_test"]) - valid_train_test)
        if invalid_values:
            raise ValueError(f"Unexpected train_test values: {invalid_values}")

        for col in feature_cols:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

        if "status_type_id" in prepared.columns:
            prepared["status_type_id"] = (
                pd.to_numeric(prepared["status_type_id"], errors="coerce")
                .fillna(-1)
                .astype(int)
            )
        else:
            prepared["status_type_id"] = -1

        if "label" in prepared.columns:
            prepared["label"] = (
                pd.to_numeric(prepared["label"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            prepared["label"] = (prepared["label"] > 0).astype(np.int8)
        else:
            raise ValueError(
                "Combined CSV has no 'label' column. "
                "In the CARE dataset, fault events have status_type_id=0, so "
                "status-based labeling incorrectly marks all faults as normal. "
                "Rebuild the combined CSV using CAREToCombinedCSV (build_combined_csv.py) "
                "which adds GroundTruth event-boundary labels."
            )

        return prepared.sort_values(
            ["asset_id", "sequence_id", "time_stamp"],
            kind="mergesort",
        ).reset_index(drop=True)

    def split_train_val_test_by_sequence(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_parts = []
        val_parts = []
        test_parts = []

        for (_, _), group in df.groupby(["asset_id", "sequence_id"], sort=False):
            group = group.sort_values("time_stamp", kind="mergesort")
            group_train = group[group["train_test"] == "train"].copy()
            group_test = group[group["train_test"] == "prediction"].copy()

            if self.validation_source == "train_tail" and not group_train.empty:
                split_idx = 1 if len(group_train) == 1 else int(len(group_train) * (1.0 - self.val_ratio))
                fit_part = group_train.iloc[:split_idx].copy()
                val_part = group_train.iloc[split_idx:].copy()

                fit_part["data_split"] = "train"
                val_part["data_split"] = "val"
                train_parts.append(fit_part)
                if not val_part.empty:
                    val_parts.append(val_part)

            elif self.validation_source == "prediction" and not group_train.empty:
                group_train["data_split"] = "train"
                train_parts.append(group_train)

            if self.validation_source == "prediction" and not group_test.empty:
                if len(group_test) == 1:
                    val_part = group_test.iloc[0:0]
                    test_part = group_test.copy()
                else:
                    split_idx = int(len(group_test) * self.prediction_val_ratio)
                    split_idx = max(1, min(len(group_test) - 1, split_idx))
                    val_part = group_test.iloc[:split_idx].copy()
                    test_part = group_test.iloc[split_idx:].copy()

                val_part["data_split"] = "val"
                test_part["data_split"] = "test"
                if not val_part.empty:
                    val_parts.append(val_part)
                if not test_part.empty:
                    test_parts.append(test_part)
            elif not group_test.empty:
                group_test["data_split"] = "test"
                test_parts.append(group_test)

        base_columns = df.columns.tolist()
        train_df = pd.concat(train_parts, ignore_index=True) if train_parts else empty_split_frame(base_columns)
        val_df = pd.concat(val_parts, ignore_index=True) if val_parts else empty_split_frame(base_columns)
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else empty_split_frame(base_columns)
        return train_df, val_df, test_df

    @staticmethod
    def summarize_split_rows(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        summary = {}
        for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            rows = len(split_df)
            positive_rows = int(split_df["label"].sum()) if rows and "label" in split_df.columns else 0
            has_group_cols = {"asset_id", "sequence_id"}.issubset(split_df.columns)
            summary[split_name] = {
                "rows": rows,
                "positive_rows": positive_rows,
                "negative_rows": rows - positive_rows,
                "positive_rate": float(positive_rows / rows) if rows else 0.0,
                "assets": int(split_df["asset_id"].nunique()) if rows and "asset_id" in split_df.columns else 0,
                "sequences": (
                    split_df.groupby(["asset_id", "sequence_id"], sort=False).ngroups
                    if rows and has_group_cols
                    else 0
                ),
            }
        return summary

    @staticmethod
    def fill_feature_gaps(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        filled = df.sort_values(
            ["asset_id", "sequence_id", "time_stamp"],
            kind="mergesort",
        ).copy()
        filled[feature_cols] = (
            filled.groupby(["asset_id", "sequence_id"], group_keys=False)[feature_cols]
            .apply(lambda part: part.ffill().bfill().fillna(0.0))
        )
        filled[feature_cols] = filled[feature_cols].astype(np.float32)
        return filled

    def _new_scaler(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        return MinMaxScaler() if self.scaler_type == "minmax" else StandardScaler()

    def fit_asset_scalers(self, train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
        prepared_train = self.fill_feature_gaps(train_df, feature_cols)
        scalers = {}

        for asset_id, asset_rows in prepared_train.groupby("asset_id", sort=False):
            # Fit scaler only on normal production rows so that maintenance/service
            # sensor readings don't skew the scaling range.
            normal_rows = asset_rows[asset_rows["status_type_id"].isin(self.normal_statuses)]
            if normal_rows.empty:
                normal_rows = asset_rows
            scaler = self._new_scaler()
            scaler.fit(normal_rows[feature_cols].to_numpy(dtype=np.float32))
            scalers[asset_id] = scaler

        if not scalers:
            raise ValueError("No per-asset scalers could be fitted from the train split.")
        return scalers

    def transform_by_asset(self, df: pd.DataFrame, feature_cols: list[str], scalers: dict) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        transformed = self.fill_feature_gaps(df, feature_cols)
        for asset_id, asset_rows in transformed.groupby("asset_id", sort=False):
            if asset_id not in scalers:
                raise KeyError(f"No scaler fitted for asset_id={asset_id}")

            transformed_values = scalers[asset_id].transform(
                asset_rows[feature_cols].to_numpy(dtype=np.float32)
            ).astype(np.float32)
            transformed.loc[asset_rows.index, feature_cols] = transformed_values

        return transformed

    def build_windows(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        window_steps: int,
        group_cols=("asset_id", "sequence_id"),
        split_name: str = "",
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        feature_count = len(feature_cols)
        meta_columns = [
            "split",
            "asset_id",
            "sequence_id",
            "start_time",
            "end_time",
            "horizon_start_time",
            "horizon_end_time",
            "target_label",
            "future_label",
            "first_future_anomaly_time",
            "last_input_label",
            "last_input_status_type_id",
            "last_label",
            "last_status_type_id",
            "window_steps",
            "horizon_steps",
        ]

        if df.empty:
            empty_x = np.empty((0, window_steps, feature_count), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int8)
            return empty_x, empty_y, pd.DataFrame(columns=meta_columns)

        X_list = []
        y_list = []
        meta_rows = []

        for _, group in df.groupby(list(group_cols), sort=False):
            group = group.sort_values("time_stamp", kind="mergesort").reset_index(drop=True)
            horizon_steps = self.prediction_horizon_steps if self.uses_future_horizon else 0
            required_steps = window_steps + horizon_steps
            if len(group) < required_steps:
                continue

            features = group[feature_cols].to_numpy(dtype=np.float32)
            labels = group["label"].to_numpy(dtype=np.int8)
            statuses = group["status_type_id"].to_numpy(dtype=int)
            timestamps = group["time_stamp"].astype(str).to_numpy()

            max_start = len(group) - required_steps + 1
            for start in range(0, max_start, self.stride_steps):
                end = start + window_steps
                last_input_time = timestamps[end - 1]
                last_input_label = int(labels[end - 1])
                last_input_status = int(statuses[end - 1])
                if self.uses_future_horizon:
                    horizon_end = end + horizon_steps
                    future_labels = labels[end:horizon_end]
                    target_label = int(future_labels.max())
                    positive_offsets = np.flatnonzero(future_labels == 1)
                    first_future_anomaly_time = (
                        timestamps[end + int(positive_offsets[0])]
                        if len(positive_offsets)
                        else ""
                    )
                    horizon_start_time = timestamps[end]
                    horizon_end_time = timestamps[horizon_end - 1]
                else:
                    target_label = int(labels[start:end].max())
                    first_future_anomaly_time = ""
                    horizon_start_time = last_input_time
                    horizon_end_time = last_input_time
                X_list.append(features[start:end])
                y_list.append(target_label)
                meta_rows.append(
                    {
                        "split": split_name,
                        "asset_id": group.iloc[end - 1]["asset_id"],
                        "sequence_id": group.iloc[end - 1]["sequence_id"],
                        "start_time": timestamps[start],
                        "end_time": last_input_time,
                        "horizon_start_time": horizon_start_time,
                        "horizon_end_time": horizon_end_time,
                        "target_label": target_label,
                        "future_label": target_label,
                        "first_future_anomaly_time": first_future_anomaly_time,
                        "last_input_label": last_input_label,
                        "last_input_status_type_id": last_input_status,
                        # Backward-compatible aliases for older training helpers.
                        "last_label": target_label,
                        "last_status_type_id": last_input_status,
                        "window_steps": window_steps,
                        "horizon_steps": horizon_steps,
                    }
                )

        if not X_list:
            empty_x = np.empty((0, window_steps, feature_count), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int8)
            return empty_x, empty_y, pd.DataFrame(columns=meta_columns)

        return np.stack(X_list).astype(np.float32), np.asarray(y_list, dtype=np.int8), pd.DataFrame(meta_rows)

    @staticmethod
    def compute_class_weights(y: np.ndarray) -> dict:
        counts = np.bincount(y.astype(int), minlength=2)
        present = counts > 0
        if present.sum() < 2:
            return {0: 1.0, 1: 1.0}

        total = counts.sum()
        return {
            class_id: float(total / (present.sum() * class_count))
            for class_id, class_count in enumerate(counts)
            if class_count > 0
        }

    def sample_probe_training_windows(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(X) <= self.probe_max_train_windows:
            return X, y

        rng = np.random.default_rng(self.random_seed)
        selected_indices = []
        unique_classes, class_counts = np.unique(y, return_counts=True)
        for class_id, class_count in zip(unique_classes, class_counts):
            class_indices = np.where(y == class_id)[0]
            target_count = max(1, int(round(self.probe_max_train_windows * (class_count / len(y)))))
            target_count = min(target_count, len(class_indices))
            selected_indices.append(rng.choice(class_indices, size=target_count, replace=False))

        selected_indices = np.concatenate(selected_indices)
        if len(selected_indices) > self.probe_max_train_windows:
            selected_indices = rng.choice(selected_indices, size=self.probe_max_train_windows, replace=False)

        selected_indices = np.sort(selected_indices)
        return X[selected_indices], y[selected_indices]

    @staticmethod
    def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))

    @staticmethod
    def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def compute_event_f1(meta_df: pd.DataFrame, scores: np.ndarray) -> tuple[float, float, int]:
        from sklearn.metrics import f1_score

        if meta_df.empty or len(scores) == 0:
            return 0.0, 0.5, 0

        label_col = "target_label" if "target_label" in meta_df.columns else "last_label"
        event_df = meta_df[["asset_id", "sequence_id", label_col]].copy()
        event_df.rename(columns={label_col: "true_label"}, inplace=True)
        event_df["score"] = scores
        event_df = (
            event_df.groupby(["asset_id", "sequence_id"], as_index=False)
            .agg(true_label=("true_label", "max"), max_score=("score", "max"))
        )

        best_f1 = 0.0
        best_threshold = DEFAULT_PROBE_INITIAL_THRESHOLD
        for threshold in np.arange(
            DEFAULT_PROBE_THRESHOLD_START,
            DEFAULT_PROBE_THRESHOLD_STOP,
            DEFAULT_PROBE_THRESHOLD_STEP,
        ):
            predictions = (event_df["max_score"] >= threshold).astype(int)
            current_f1 = float(f1_score(event_df["true_label"], predictions, zero_division=0))
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(round(threshold, 2))

        return best_f1, best_threshold, int(len(event_df))

    def build_probe_model(self, input_shape):
        import tensorflow as tf
        from tensorflow.keras import layers

        model = tf.keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.GRU(DEFAULT_PROBE_GRU_UNITS),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=DEFAULT_PROBE_LEARNING_RATE
            ),
            loss=DEFAULT_PROBE_LOSS,
            metrics=[
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            ],
        )
        return model

    def search_best_windows(self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        import tensorflow as tf
        from tensorflow.keras import callbacks

        results = []
        print("\nWindow size search")
        print("-" * 60)

        for window_hours in self.window_candidates_hours:
            window_steps = self.window_steps_from_hours(window_hours)
            print(f"Testing {window_hours}h window ({window_steps} steps)")

            X_train, y_train, _ = self.build_windows(train_df, feature_cols, window_steps, split_name="train")
            X_val, y_val, val_meta = self.build_windows(val_df, feature_cols, window_steps, split_name="val")

            usable = True
            best_threshold = np.nan
            pr_auc = np.nan
            roc_auc = np.nan
            event_f1 = np.nan
            sampled_train = len(X_train)
            val_events = 0
            note = ""

            if len(X_train) == 0 or len(X_val) == 0:
                usable = False
                note = "not_enough_windows"
            elif len(np.unique(y_train)) < 2:
                usable = False
                note = "train_has_one_class"
            elif len(np.unique(y_val)) < 2:
                usable = False
                note = "val_has_one_class"

            if usable:
                X_probe, y_probe = self.sample_probe_training_windows(X_train, y_train)
                sampled_train = len(X_probe)

                tf.keras.backend.clear_session()
                np.random.seed(self.random_seed)
                tf.random.set_seed(self.random_seed)
                model = self.build_probe_model((window_steps, len(feature_cols)))
                early_stop = callbacks.EarlyStopping(
                    monitor=DEFAULT_PROBE_CALLBACK_MONITOR,
                    mode=DEFAULT_PROBE_CALLBACK_MODE,
                    patience=DEFAULT_PROBE_EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    verbose=0,
                )

                model.fit(
                    X_probe,
                    y_probe,
                    validation_data=(X_val, y_val),
                    epochs=self.probe_epochs,
                    batch_size=self.probe_batch_size,
                    class_weight=self.compute_class_weights(y_probe),
                    callbacks=[early_stop],
                    verbose=0,
                )

                val_scores = model.predict(X_val, batch_size=self.probe_batch_size, verbose=0).reshape(-1)
                pr_auc = self.safe_pr_auc(y_val, val_scores)
                roc_auc = self.safe_roc_auc(y_val, val_scores)
                event_f1, best_threshold, val_events = self.compute_event_f1(val_meta, val_scores)

                del model, X_probe, y_probe, val_scores
                tf.keras.backend.clear_session()
                gc.collect()

            results.append(
                {
                    "window_hours": int(window_hours),
                    "window_steps": int(window_steps),
                    "prediction_horizon_steps": int(self.prediction_horizon_steps),
                    "label_mode": self.label_mode,
                    "label_definition": self.label_definition,
                    "train_windows": int(len(X_train)),
                    "probe_train_windows": int(sampled_train),
                    "val_windows": int(len(X_val)),
                    "val_events": int(val_events),
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "event_f1": event_f1,
                    "best_threshold": best_threshold,
                    "usable": bool(usable),
                    "note": note,
                }
            )

            del X_train, y_train, X_val, y_val, val_meta
            gc.collect()

        results_df = pd.DataFrame(results).sort_values(
            ["event_f1", "pr_auc", "roc_auc"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

        print(results_df[["window_hours", "event_f1", "pr_auc", "roc_auc", "usable", "note"]])
        return results_df

    @staticmethod
    def save_scalers(scalers: dict, scaler_dir: Path) -> None:
        import joblib

        scaler_dir.mkdir(parents=True, exist_ok=True)
        for asset_id, scaler in scalers.items():
            joblib.dump(scaler, scaler_dir / f"asset_{asset_id}.pkl")

    @staticmethod
    def class_counts(y: np.ndarray) -> dict[str, int]:
        labels = np.asarray(y, dtype=np.int8)
        return {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
            "total": int(len(labels)),
        }

    def export_classifier_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        window_hours: int,
        output_dir: Path,
        scalers: dict,
    ) -> dict:
        classifier_dir = output_dir / "classifier"
        classifier_dir.mkdir(parents=True, exist_ok=True)

        window_steps = self.window_steps_from_hours(window_hours)
        X_train, y_train, train_meta = self.build_windows(train_df, feature_cols, window_steps, split_name="train")
        X_val, y_val, val_meta = self.build_windows(val_df, feature_cols, window_steps, split_name="val")
        X_test, y_test, test_meta = self.build_windows(test_df, feature_cols, window_steps, split_name="test")

        np.save(classifier_dir / "X_train.npy", X_train)
        np.save(classifier_dir / "y_train.npy", y_train)
        np.save(classifier_dir / "X_val.npy", X_val)
        np.save(classifier_dir / "y_val.npy", y_val)
        np.save(classifier_dir / "X_test.npy", X_test)
        np.save(classifier_dir / "y_test.npy", y_test)

        train_meta.to_csv(classifier_dir / "train_meta.csv", index=False)
        val_meta.to_csv(classifier_dir / "val_meta.csv", index=False)
        test_meta.to_csv(classifier_dir / "test_meta.csv", index=False)

        self.save_scalers(scalers, classifier_dir / "scalers")
        save_metadata(
            classifier_dir / "metadata.json",
            {
                "source_csv": str(self.csv_path),
                "window_hours": int(window_hours),
                "window_steps": int(window_steps),
                "stride_steps": int(self.stride_steps),
                "prediction_horizon_steps": int(self.prediction_horizon_steps),
                "label_mode": self.label_mode,
                "label_definition": self.label_definition,
                "validation_source": self.validation_source,
                "prediction_val_ratio": self.prediction_val_ratio,
                "scaler_type": self.scaler_type,
                "feature_cols": feature_cols,
                "train_shape": list(X_train.shape),
                "val_shape": list(X_val.shape),
                "test_shape": list(X_test.shape),
                "class_counts": {
                    "train": self.class_counts(y_train),
                    "val": self.class_counts(y_val),
                    "test": self.class_counts(y_test),
                },
            },
        )

        return {
            "train_windows": int(len(X_train)),
            "val_windows": int(len(X_val)),
            "test_windows": int(len(X_test)),
        }

    def _resolve_windows(self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[int], pd.DataFrame]:
        if self.selected_windows_hours:
            rows = [
                {
                    "window_hours": int(window_hours),
                    "window_steps": int(self.window_steps_from_hours(window_hours)),
                    "prediction_horizon_steps": int(self.prediction_horizon_steps),
                    "label_mode": self.label_mode,
                    "label_definition": self.label_definition,
                    "usable": True,
                    "note": "selected_by_user",
                }
                for window_hours in self.selected_windows_hours
            ]
            return [int(x) for x in self.selected_windows_hours], pd.DataFrame(rows)

        if not self.run_window_search:
            first_window = int(self.window_candidates_hours[0])
            rows = [
                {
                    "window_hours": first_window,
                    "window_steps": int(self.window_steps_from_hours(first_window)),
                    "prediction_horizon_steps": int(self.prediction_horizon_steps),
                    "label_mode": self.label_mode,
                    "label_definition": self.label_definition,
                    "usable": True,
                    "note": "window_search_skipped",
                }
            ]
            return [first_window], pd.DataFrame(rows)

        results_df = self.search_best_windows(train_df, val_df, feature_cols)
        usable_results = results_df[results_df["usable"]].copy()
        if usable_results.empty:
            raise RuntimeError("Window search did not find any usable candidate.")

        best_windows = usable_results.head(self.top_k_windows)["window_hours"].astype(int).tolist()
        return best_windows, results_df

    def run(self) -> dict:
        np.random.seed(self.random_seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df, feature_cols = self.load_inputs()
        prepared_df = self.prepare_base_dataframe(df, feature_cols)

        print("=" * 70)
        print("Combined CSV Sequence Preparation")
        print("=" * 70)
        print(f"Source CSV       : {self.csv_path}")
        print(f"Output root      : {self.output_dir}")
        print(f"Loaded rows      : {len(prepared_df):,}")
        print(f"Selected features: {len(feature_cols)}")
        print(f"Assets           : {prepared_df['asset_id'].nunique()}")
        print(f"Sequences        : {prepared_df['sequence_id'].nunique()}")
        print(f"Scaler           : {self.scaler_type}")
        print(f"Validation source: {self.validation_source}")
        if self.validation_source == "prediction":
            print(f"Prediction val % : {self.prediction_val_ratio:.2f}")
        print(f"Label mode       : {self.label_mode}")
        if self.uses_future_horizon:
            print(f"Future horizon   : {self.prediction_horizon_steps} steps")
        else:
            print("Detection label  : max label within input window")

        train_df, val_df, test_df = self.split_train_val_test_by_sequence(prepared_df)
        split_summary = self.summarize_split_rows(train_df, val_df, test_df)
        save_metadata(
            self.output_dir / "split_summary.json",
            {
                "validation_source": self.validation_source,
                "val_ratio": self.val_ratio,
                "prediction_val_ratio": self.prediction_val_ratio,
                "label_mode": self.label_mode,
                "label_definition": self.label_definition,
                "splits": split_summary,
            },
        )
        print("Split rows       :")
        for split_name in ("train", "val", "test"):
            split_info = split_summary[split_name]
            print(
                f"  {split_name:<5} rows={split_info['rows']:,} "
                f"positive_rate={split_info['positive_rate']:.3f}"
            )
        scalers = self.fit_asset_scalers(train_df, feature_cols)

        train_df_sc = self.transform_by_asset(train_df, feature_cols, scalers)
        val_df_sc = self.transform_by_asset(val_df, feature_cols, scalers)
        test_df_sc = self.transform_by_asset(test_df, feature_cols, scalers)

        best_windows, results_df = self._resolve_windows(train_df_sc, val_df_sc, feature_cols)
        results_df.to_csv(self.output_dir / "window_search_results.csv", index=False)
        save_metadata(
            self.output_dir / "best_windows.json",
            {
                "best_windows_hours": best_windows,
                "top_k": int(len(best_windows)),
                "prediction_horizon_steps": int(self.prediction_horizon_steps),
                "label_mode": self.label_mode,
                "label_definition": self.label_definition,
                "validation_source": self.validation_source,
                "prediction_val_ratio": self.prediction_val_ratio,
                "ranked_results": results_df.to_dict(orient="records"),
            },
        )

        export_summaries = []
        for window_hours in best_windows:
            print(f"\nExporting datasets for {window_hours}h window")
            window_dir = self.output_dir / f"window_{window_hours}h"
            if self.skip_classifier:
                print("  [skip] classifier export skipped (--skip-classifier-export)")
                classifier_summary = {}
            else:
                classifier_summary = self.export_classifier_data(
                    train_df_sc,
                    val_df_sc,
                    test_df_sc,
                    feature_cols,
                    window_hours,
                    window_dir,
                    scalers,
                )
            export_summaries.append(
                {
                    "window_hours": int(window_hours),
                    "classifier": classifier_summary,
                }
            )

        payload = {
            "source_csv": str(self.csv_path),
            "feature_file": str(self.feature_file) if self.feature_file else None,
            "output_dir": str(self.output_dir),
            "prediction_horizon_steps": int(self.prediction_horizon_steps),
            "label_mode": self.label_mode,
            "label_definition": self.label_definition,
            "validation_source": self.validation_source,
            "val_ratio": self.val_ratio,
            "prediction_val_ratio": self.prediction_val_ratio,
            "split_summary": split_summary,
            "exports": export_summaries,
        }
        save_metadata(self.output_dir / "export_summary.json", payload)

        print("\n" + "=" * 70)
        print("Combined preparation complete")
        print("=" * 70)
        print(f"Chosen windows: {best_windows}")
        print(f"Saved outputs : {self.output_dir}")
        return payload
