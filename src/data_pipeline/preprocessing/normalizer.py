"""
AssetNormalizer — data_pipeline.preprocessing.normalizer
StandardScaler normalization fitted exclusively on each asset's training data.
"""

import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from sklearn.preprocessing import StandardScaler
from config import STRIDE
from data_pipeline.loaders.sequence_maker import SequenceMaker


class AssetNormalizer:
    """
    Normalizes SCADA sequence data using a StandardScaler fitted only on
    training data — ensuring the model never sees anomalous patterns during
    normalization.

    Args:
        output_dir: Root directory where fitted scalers are saved.
        seq_len: Sequence window length used when sequencing test events.
    """

    def __init__(self, output_dir: str, seq_len: int) -> None:
        self.output_dir = output_dir
        self.seq_len    = seq_len

    def normalize_asset(
        self,
        asset_id: int,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        test_data_dict: dict,
    ) -> tuple:
        """
        Normalize data for a single asset using a StandardScaler fit only on that
        asset's training sequences.

        Each asset gets its own scaler (scaler_asset_{id}.pkl) — ensuring the
        normal-behaviour baseline is specific to that turbine.

        Args:
            asset_id: Turbine/asset ID (used for scaler filename).
            X_train: Train sequences (n_seq, window, features).
            X_val: Val sequences (n_seq, window, features).
            y_train: Train targets (n_seq, features).
            y_val: Val targets (n_seq, features).
            test_data_dict: {event_id: {'features': np.ndarray, ...}} from process_asset_test().

        Returns:
            Tuple of (X_train_sc, X_val_sc, y_train_sc, y_val_sc, test_data_scaled_dict).
        """
        n_train, _seq_len, n_features = X_train.shape
        n_val = X_val.shape[0]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, _seq_len, n_features)
        X_val_sc   = scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, _seq_len, n_features)
        y_train_sc = scaler.transform(y_train)
        y_val_sc   = scaler.transform(y_val)

        seq_maker = SequenceMaker(window_size=self.seq_len, stride=STRIDE)
        test_data_scaled = {}
        for event_id, data in test_data_dict.items():
            X_test, y_test = seq_maker.create_sequences(data["features"])
            if len(X_test) == 0:
                continue
            X_test_sc = scaler.transform(X_test.reshape(-1, n_features)).reshape(-1, self.seq_len, n_features)
            y_test_sc = scaler.transform(y_test)
            
            seq_ts = []
            if "time_stamps" in data:
                ts = data["time_stamps"]
                stride = seq_maker.stride
                seq_ts = [ts[i * stride + self.seq_len] for i in range(len(X_test))]
                
            test_data_scaled[event_id] = {
                "X": X_test_sc, "y": y_test_sc,
                "time_stamps": seq_ts,
                "label":      data["label"],
                "event_start": data["event_start"],
                "event_end":   data["event_end"],
                "asset_id":    data.get("asset_id", asset_id),
            }

        os.makedirs(self.output_dir, exist_ok=True)
        scaler_path = os.path.join(self.output_dir, f"scaler_asset_{asset_id}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"  Asset {asset_id}: scaler saved → {scaler_path}")

        return X_train_sc, X_val_sc, y_train_sc, y_val_sc, test_data_scaled


# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases
# ---------------------------------------------------------------------------

def normalize_data(
    X_train, X_val, y_train, y_val,
    test_data_dict: dict,
    output_dir: str,
) -> tuple:
    """Legacy alias — wraps AssetNormalizer.normalize_data()."""
    norm = AssetNormalizer(output_dir=output_dir, seq_len=X_train.shape[1])
    return norm.normalize_data(X_train, X_val, y_train, y_val, test_data_dict)


def normalize_asset(
    asset_id: int,
    X_train, X_val, y_train, y_val,
    test_data_dict: dict,
    output_dir: str,
    seq_len: int,
) -> tuple:
    """Legacy alias — wraps AssetNormalizer.normalize_asset()."""
    norm = AssetNormalizer(output_dir=output_dir, seq_len=seq_len)
    return norm.normalize_asset(asset_id, X_train, X_val, y_train, y_val, test_data_dict)
