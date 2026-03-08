"""
SequenceMaker — data_pipeline.loaders.sequence_maker
Creates sliding-window sequences for LSTM training and probe models.
"""

import numpy as np


class SequenceMaker:
    """
    Creates fixed-length sliding-window sequences from a time-series array.

    Args:
        window_size: Number of timesteps per sequence (X window).
        stride: Step size between consecutive sequence starts.
    """

    def __init__(self, window_size: int, stride: int = 1) -> None:
        self.window_size = window_size
        self.stride      = stride

    def create_sequences(self, data: np.ndarray) -> tuple:
        """
        Create sliding-window sequences for LSTM prediction.

        X: sequences of length window_size.
        y: next-timestep values (target to predict).

        Args:
            data: 2D array (n_timesteps, n_features).

        Returns:
            Tuple of (X, y) arrays.
            X shape: (n_sequences, window_size, n_features)
            y shape: (n_sequences, n_features)
        """
        n_timesteps, n_features = data.shape
        n_sequences = (n_timesteps - self.window_size) // self.stride

        if n_sequences <= 0:
            return np.array([]), np.array([])

        X = np.zeros((n_sequences, self.window_size, n_features), dtype=np.float32)
        y = np.zeros((n_sequences, n_features), dtype=np.float32)

        for i in range(n_sequences):
            start = i * self.stride
            end   = start + self.window_size
            X[i]  = data[start:end]
            y[i]  = data[end]

        return X, y

    def create_probe_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create non-overlapping (or strided) sequences for probe / feature-
        selection LSTM; returns only X (no target needed).

        Args:
            data: 2D array (n_timesteps, n_features).

        Returns:
            3D array (n_sequences, window_size, n_features).
        """
        sequences = []
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            seq = data[i: i + self.window_size].astype(np.float32)
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backward-compatible module-level aliases
# ---------------------------------------------------------------------------

def create_sequences(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple:
    """Legacy alias — wraps SequenceMaker.create_sequences()."""
    return SequenceMaker(window_size=window_size, stride=stride).create_sequences(data)


def create_probe_sequences(
    data: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Legacy alias — wraps SequenceMaker.create_probe_sequences()."""
    return SequenceMaker(window_size=window_size, stride=stride).create_probe_sequences(data)
