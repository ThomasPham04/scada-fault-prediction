"""
Feature Engineering — data_pipeline.preprocessing.feature_engineering
Transforms raw SCADA DataFrame columns: angle encoding, counter dropping,
feature-column selection, and raw value extraction.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import (
    NBM_ANGLE_FEATURES,
    NBM_COUNTER_FEATURES,
    NBM_FEATURE_COLUMNS,
    EXCLUDE_COLUMNS,
)


def engineer_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert angle features to sin/cos components for angular continuity.
    Drops the original degree columns after conversion.

    Args:
        df: DataFrame with raw angle columns

    Returns:
        DataFrame with sin/cos columns replacing angle columns
    """
    df_copy = df.copy()
    for col in NBM_ANGLE_FEATURES:
        if col in df_copy.columns:
            radians = np.radians(df_copy[col])
            df_copy[f'{col}_sin'] = np.sin(radians)
            df_copy[f'{col}_cos'] = np.cos(radians)
            df_copy.drop(col, axis=1, inplace=True)
    return df_copy


def drop_counter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop cumulative energy counter columns (Wh, VArh).
    These represent accumulated totals, not instantaneous readings.

    Args:
        df: DataFrame that may contain counter columns

    Returns:
        DataFrame with counter columns removed
    """
    cols_to_drop = [col for col in NBM_COUNTER_FEATURES if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Determine the final feature column list after angle engineering.
    Includes base sensor features plus the sin/cos engineered columns.

    Args:
        df: DataFrame after engineer_angle_features and drop_counter_features

    Returns:
        Ordered list of feature column names to use for modelling
    """
    feature_cols = [col for col in NBM_FEATURE_COLUMNS if col in df.columns]

    for angle_col in NBM_ANGLE_FEATURES:
        sin_col = f'{angle_col}_sin'
        cos_col = f'{angle_col}_cos'
        if sin_col in df.columns:
            feature_cols.append(sin_col)
        if cos_col in df.columns:
            feature_cols.append(cos_col)

    exclude_all = EXCLUDE_COLUMNS + ['status_type_id'] + NBM_ANGLE_FEATURES + NBM_COUNTER_FEATURES
    feature_cols = [col for col in feature_cols if col not in exclude_all]

    return feature_cols


def preprocess_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Extract and clean feature values from a DataFrame.
    Applies forward-fill, back-fill, then zero-fill for remaining NaNs.

    Args:
        df: Source DataFrame
        feature_cols: Columns to extract

    Returns:
        2D float array of shape (n_timesteps, n_features)
    """
    features = df[feature_cols].copy()
    features = features.ffill().bfill()
    features = features.fillna(0)
    return features.values
