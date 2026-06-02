"""
classifier_realtime_detector.py - inference.classifier_realtime_detector
Real-time style inference for trained sequence classifier models.

This module simulates a deployment loop for the supervised classifiers trained
by `python src/main.py train-sequences --model lstm/gru/cnn_lstm/cnn_gru`.
At each 10-minute SCADA tick, call `ingest(row)`. The detector keeps one
rolling window per asset and returns a probability once the window is full.
"""

from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf

from training.hyperparameters.inference_defaults import DEFAULT_THRESHOLD
from training.sequence_utils import load_json


class RealTimeSequenceClassifier:
    """Rolling-window inference wrapper for a trained sequence classifier.

    The trained classifier expects scaled windows shaped `(1, window_steps,
    feature_count)`. This class accepts one raw 10-minute row at a time,
    applies the per-asset scaler saved by the sequence export step, maintains a
    rolling buffer, and emits an early-warning probability when ready.

    Important: when the model was trained with the default `future_horizon`
    label mode, the output is an early-warning probability for the configured
    future horizon, not a current-timestamp fault label.
    """

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        scaler_dir: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        strict_features: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.scaler_dir = Path(scaler_dir)
        self.threshold = float(threshold)
        self.strict_features = bool(strict_features)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        if not self.scaler_dir.exists():
            raise FileNotFoundError(f"Scaler folder not found: {self.scaler_dir}")

        self.metadata = load_json(self.metadata_path)
        self.feature_cols = list(self.metadata["feature_cols"])
        self.window_steps = int(self.metadata["window_steps"])
        self.prediction_horizon_steps = int(
            self.metadata.get("prediction_horizon_steps", 0)
        )
        self.label_definition = str(self.metadata.get("label_definition", "unknown"))

        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self._buffers: dict[int, deque[list[float]]] = {}
        self._last_raw_vectors: dict[int, list[float]] = {}
        self._scalers: dict[int, Any] = {}
        self._n_ticks: dict[int, int] = {}

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path,
        export_classifier_dir: str | Path,
        threshold: float | None = None,
        strict_features: bool = True,
    ) -> "RealTimeSequenceClassifier":
        """Create detector from standard training/export folders.

        Parameters
        ----------
        model_dir:
            Folder such as
            `results/sequence_training_results/window_24h/classifier/cnn_lstm`.
        export_classifier_dir:
            Folder such as
            `Dataset/processed/sequence_exports/window_24h/classifier`.
        threshold:
            Optional override. If omitted, tries `metrics.json` and falls back
            to 0.5.
        """
        model_dir = Path(model_dir)
        export_classifier_dir = Path(export_classifier_dir)
        metrics_path = model_dir / "metrics.json"
        if threshold is None and metrics_path.exists():
            metrics = load_json(metrics_path)
            threshold = float(
                metrics.get(
                    "selected_threshold",
                    metrics.get("summary", {}).get("threshold", DEFAULT_THRESHOLD),
                )
            )
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        return cls(
            model_path=model_dir / "model.keras",
            metadata_path=export_classifier_dir / "metadata.json",
            scaler_dir=export_classifier_dir / "scalers",
            threshold=threshold,
            strict_features=strict_features,
        )

    def _load_scaler(self, asset_id: int):
        if asset_id not in self._scalers:
            scaler_path = self.scaler_dir / f"asset_{asset_id}.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"No scaler for asset_id={asset_id}: {scaler_path}"
                )
            self._scalers[asset_id] = joblib.load(scaler_path)
        return self._scalers[asset_id]

    @staticmethod
    def _angle_feature_from_raw(feature_name: str, row: dict[str, Any]) -> float | None:
        if feature_name.endswith("_sin"):
            base = feature_name[:-4]
            if base in row:
                return math.sin(math.radians(float(row[base])))
        if feature_name.endswith("_cos"):
            base = feature_name[:-4]
            if base in row:
                return math.cos(math.radians(float(row[base])))
        return None

    def _build_raw_vector(self, row: dict[str, Any], asset_id: int) -> list[float]:
        previous = self._last_raw_vectors.get(asset_id)
        values: list[float] = []
        missing: list[str] = []

        for idx, feature_name in enumerate(self.feature_cols):
            value = row.get(feature_name)
            if value is None:
                value = self._angle_feature_from_raw(feature_name, row)
            if value is None and previous is not None:
                value = previous[idx]
            if value is None:
                missing.append(feature_name)
                value = 0.0
            values.append(float(value))

        if missing and self.strict_features:
            shown = ", ".join(missing[:10])
            extra = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
            raise ValueError(
                "Missing required feature(s) for real-time classifier input: "
                f"{shown}{extra}. Pass engineered feature columns directly, "
                "or pass raw angle columns so *_sin/*_cos can be computed."
            )

        self._last_raw_vectors[asset_id] = values
        return values

    def ingest(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process one new 10-minute SCADA row.

        `row` must contain `asset_id` plus either the exact `feature_cols`
        from export metadata, or raw angle columns for any required `*_sin` /
        `*_cos` engineered features.
        """
        if "asset_id" not in row:
            raise ValueError("row must contain asset_id")

        asset_id = int(row["asset_id"])
        raw_vector = self._build_raw_vector(row, asset_id)
        scaler = self._load_scaler(asset_id)
        scaled_vector = (
            scaler.transform(np.asarray(raw_vector, dtype=np.float32).reshape(1, -1))
            .astype(np.float32)
            .reshape(-1)
            .tolist()
        )

        if asset_id not in self._buffers:
            self._buffers[asset_id] = deque(maxlen=self.window_steps)
            self._n_ticks[asset_id] = 0
        self._buffers[asset_id].append(scaled_vector)
        self._n_ticks[asset_id] += 1

        ready = len(self._buffers[asset_id]) == self.window_steps
        result = {
            "ready": ready,
            "asset_id": asset_id,
            "ticks_seen": self._n_ticks[asset_id],
            "warmup_remaining": max(0, self.window_steps - len(self._buffers[asset_id])),
            "window_steps": self.window_steps,
            "prediction_horizon_steps": self.prediction_horizon_steps,
            "label_definition": self.label_definition,
        }
        if not ready:
            return result

        window = np.asarray(list(self._buffers[asset_id]), dtype=np.float32)
        score = float(self.model.predict(window[np.newaxis, :, :], batch_size=1, verbose=0).reshape(-1)[0])
        result.update(
            {
                "score": score,
                "threshold": self.threshold,
                "alarm": bool(score >= self.threshold),
                "meaning": (
                    "early_warning_probability"
                    if "future_horizon" in self.label_definition
                    else "current_window_detection_probability"
                ),
            }
        )
        return result

    def reset(self, asset_id: int | None = None) -> None:
        """Clear rolling state for one asset or all assets."""
        if asset_id is None:
            self._buffers.clear()
            self._last_raw_vectors.clear()
            self._n_ticks.clear()
            return
        asset_id = int(asset_id)
        self._buffers.pop(asset_id, None)
        self._last_raw_vectors.pop(asset_id, None)
        self._n_ticks.pop(asset_id, None)
