"""
FeatureEngineer — data_pipeline.preprocessing.feature_engineering
Transforms raw SCADA DataFrame columns: angle encoding
feature-column selection, and raw value extraction.
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import (
    ANGLE_FEATURES,
    FEATURE_COLUMNS,
    EXCLUDE_COLUMNS,
)


class FeatureEngineer:
    """
    Applies feature engineering transformations to raw SCADA DataFrames.

    All methods operate on copies of the input; the original DataFrame is
    never modified in-place.

    Args:
        angles: Raw angle columns that should be converted to sine/cosine.
        trust_bad_angles: If True, angle columns outside the expected
            [-180, 180] or [0, 360] ranges are still transformed. If False,
            those raw angle columns are dropped, matching the reference
            AngleTransformer behaviour.
    """

    def __init__(
        self,
        angles: Optional[List[str]] = None,
        trust_bad_angles: bool = False,
    ) -> None:
        self.angles = list(ANGLE_FEATURES if angles is None else angles)
        self.trust_bad_angles = trust_bad_angles

        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.angle_features_: List[str] = []
        self.angles_features_: List[str] = []
        self.invalid_range_features_: List[str] = []
        self.ranges_: dict = {}
        self.n_features_in_ = 0

    @staticmethod
    def wrap_angle_deg(series: pd.Series) -> pd.Series:
        """Wrap raw angles into [-180, 180)."""
        values = pd.to_numeric(series, errors="coerce")
        return ((values + 180.0) % 360.0) - 180.0

    @staticmethod
    def _detect_angle_range(series: pd.Series) -> Optional[tuple]:
        """
        Detect whether a column follows a standard angle range.

        Returns:
            (-180, 180), (0, 360), or None if the observed values do not
            match either range.
        """
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return None

        min_val = values.min()
        max_val = values.max()
        if -180 <= min_val < 0 <= max_val <= 180:
            return (-180, 180)
        if 0 <= min_val and max_val <= 360:
            return (0, 360)
        return None

    def _fit_angle_metadata(self, df: pd.DataFrame) -> None:
        """Detect angle columns, output names, valid ranges, and invalid ranges."""
        self.feature_names_in_ = df.columns.to_list()
        self.n_features_in_ = len(df.columns)

        features_out = [col for col in df.columns if col not in self.angles]
        angle_features = []
        invalid_range_features = []
        ranges = {}

        for col in self.angles:
            if col not in self.feature_names_in_:
                continue

            detected_range = self._detect_angle_range(df[col])
            if detected_range is None:
                angle_features.append(col)
                invalid_range_features.append(col)
                if not self.trust_bad_angles:
                    continue

                ranges[col] = (0, 360)
            else:
                ranges[col] = detected_range
                angle_features.append(col)

            features_out.extend([f"{col}_sin", f"{col}_cos"])

        if (
            "sensor_2_avg" in self.feature_names_in_
            and ("sensor_2_avg" not in invalid_range_features or self.trust_bad_angles)
            and "yaw_misalignment_abs" not in features_out
        ):
            features_out.append("yaw_misalignment_abs")

        self.feature_names_out_ = features_out
        self.angle_features_ = angle_features
        self.angles_features_ = angle_features
        self.invalid_range_features_ = invalid_range_features
        self.ranges_ = ranges

    def fit_angle_features(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Detect angle metadata from a DataFrame.

        This is the project-local equivalent of ``AngleTransformer.fit``.
        """
        self._fit_angle_metadata(df)
        return self

    def transform_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert fitted angle columns to sine/cosine features.

        This is the project-local equivalent of ``AngleTransformer.transform``.
        """
        if not self.feature_names_in_:
            raise RuntimeError("fit_angle_features must be called before transform_angle_features.")
        return self._convert_angle_columns(df)

    def _convert_angle_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted angle conversion rules to a copy of the given DataFrame."""
        df = df.copy()

        if (
            "sensor_2_avg" in df.columns
            and ("sensor_2_avg" not in self.invalid_range_features_ or self.trust_bad_angles)
        ):
            relative_direction = self.wrap_angle_deg(df["sensor_2_avg"])
            df["yaw_misalignment_abs"] = relative_direction.abs()

        for col in self.angle_features_:
            if col not in df.columns:
                continue

            valid_range = col not in self.invalid_range_features_
            valid_for_conversion = valid_range or self.trust_bad_angles
            if valid_for_conversion:
                radians = pd.to_numeric(df[col], errors="coerce") * (np.pi / 180.0)
                # Sine/cosine are periodic, so out-of-range trusted angles do
                # not need modulo adjustment before conversion.
                df[f"{col}_sin"] = np.sin(radians)
                df[f"{col}_cos"] = np.cos(radians)

            df.drop(columns=[col], inplace=True)

        return df

    def engineer_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert angle features to sin/cos components for angular continuity.
        Drops the original degree columns after conversion.

        Args:
            df: DataFrame with raw angle columns.

        Returns:
            DataFrame with sin/cos columns replacing angle columns.
        """
        self.fit_angle_features(df)
        return self.transform_angle_features(df)

    def inverse_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct raw angle values from sine/cosine columns.

        This mirrors the reference inverse_transform logic, using this
        project's ``_sin`` / ``_cos`` suffixes.
        """
        df = df.copy()

        for col in self.angle_features_:
            if col in self.invalid_range_features_ and not self.trust_bad_angles:
                continue

            sine_col = f"{col}_sin"
            cos_col = f"{col}_cos"
            if sine_col not in df.columns or cos_col not in df.columns:
                continue

            sine = df[sine_col].clip(lower=-1, upper=1)
            cosine = df[cos_col].clip(lower=-1, upper=1)

            df[col] = np.arctan2(sine, cosine) * 180 / np.pi
            if self.ranges_.get(col) == (0, 360):
                df.loc[df[col] < 0, col] = df.loc[df[col] < 0, col] + 360

            df.drop(columns=[sine_col, cos_col], inplace=True)

        if self.trust_bad_angles:
            columns = [col for col in self.feature_names_in_ if col in df.columns]
        else:
            columns = [
                col for col in self.feature_names_in_
                if col not in self.invalid_range_features_ and col in df.columns
            ]
        return df[columns]

    def get_feature_names_out(self) -> List[str]:
        """Return output feature names from the most recent angle transformation."""
        return self.feature_names_out_

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Determine the final feature column list after angle engineering.
        Includes base sensor features plus the sin/cos engineered columns.

        Args:
            df: DataFrame after engineer_angle_features.

        Returns:
            Ordered list of feature column names to use for modelling.
        """
        feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]

        for angle_col in self.angles:
            sin_col = f"{angle_col}_sin"
            cos_col = f"{angle_col}_cos"
            if sin_col in df.columns:
                feature_cols.append(sin_col)
            if cos_col in df.columns:
                feature_cols.append(cos_col)

        exclude_all = EXCLUDE_COLUMNS + ["status_type_id"] + self.angles
        feature_cols = [col for col in feature_cols if col not in exclude_all]
        return feature_cols

    def preprocess_features(
        self,
        df: pd.DataFrame,
        feature_cols: list,
    ) -> np.ndarray:
        """
        Extract and clean feature values from a DataFrame.
        Applies forward-fill, back-fill, then zero-fill for remaining NaNs.

        Args:
            df: Source DataFrame.
            feature_cols: Columns to extract.

        Returns:
            2D float array of shape (n_timesteps, n_features).
        """
        features = df[feature_cols].copy()
        features = features.ffill().bfill()
        features = features.fillna(0)
        return features.values

    @staticmethod
    def fill_missing_by_group(
        df: pd.DataFrame,
        feature_cols: List[str],
        group_col: str = "sequence_id",
    ) -> pd.DataFrame:
        """Forward/backward-fill missing feature values within each sequence."""
        df = df.copy()
        df[feature_cols] = (
            df.groupby(group_col, group_keys=False)[feature_cols]
            .apply(lambda part: part.ffill().bfill().fillna(0.0))
        )
        return df
