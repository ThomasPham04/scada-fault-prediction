"""Sequence Maker — data_pipeline.loaders.sequence_maker
Creates sliding-window sequences for LSTM training and probe models.
"""

import numpy as np


def create_sequences(data: np.ndarray, window_size: int, stride: int = 1) -> tuple:
    """
    Create sliding window sequences for LSTM prediction.

    X: sequences of length window_size
    y: next timestep values (target to predict)

    Args:
        data: 2D array (n_timesteps, n_features)
        window_size: number of timesteps per sequence
        stride: step between sequence starts

    Returns:
        Tuple of (X, y) arrays
    """
    n_timesteps, n_features = data.shape
    n_sequences = (n_timesteps - window_size) // stride

    if n_sequences <= 0:
        return np.array([]), np.array([])

    X = np.zeros((n_sequences, window_size, n_features), dtype=np.float32)
    y = np.zeros((n_sequences, n_features), dtype=np.float32)

    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + window_size
        X[i] = data[start_idx:end_idx]
        y[i] = data[end_idx]

    return X, y


def create_probe_sequences(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Create non-overlapping (or strided) sequences for the probe LSTM
    used during model-based feature selection.

    Args:
        data: 2D array (T, F)
        window_size: sequence length
        stride: step between windows

    Returns:
        3D array (N, window_size, F)
    """
    sequences = []
    for i in range(0, len(data) - window_size + 1, stride):
        seq = data[i: i + window_size].astype(np.float32)
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)
