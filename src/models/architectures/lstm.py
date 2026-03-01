"""
LSTM Architecture — models.architectures.lstm
Stacked LSTM prediction model for fault detection via reconstruction error.
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    NBM_LSTM_UNITS_1,
    NBM_LSTM_UNITS_2,
    NBM_DROPOUT_RATE,
    NBM_DENSE_UNITS,
    NBM_LEARNING_RATE,
)


def build_lstm_model(input_shape: tuple, output_dim: int) -> Model:
    """
    Build a stacked LSTM prediction model.

    Architecture:
        Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(64) → Dense(output_dim)

    Used for next-timestep prediction; reconstruction error on test data
    is the anomaly signal.

    Args:
        input_shape: (window_size, n_features)
        output_dim: Number of output features (= n_features)

    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input_sequence')

    x = layers.LSTM(NBM_LSTM_UNITS_1, return_sequences=True, name='lstm_1')(inputs)
    x = layers.Dropout(NBM_DROPOUT_RATE, name='dropout_1')(x)

    x = layers.LSTM(NBM_LSTM_UNITS_2, return_sequences=False, name='lstm_2')(x)
    x = layers.Dropout(NBM_DROPOUT_RATE, name='dropout_2')(x)

    x = layers.Dense(NBM_DENSE_UNITS, activation='relu', name='dense_intermediate')(x)
    outputs = layers.Dense(output_dim, activation='linear', name='output_prediction')(x)

    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Prediction_V2')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NBM_LEARNING_RATE),
        loss='mse',
        metrics=['mae'],
    )
    model.summary()
    return model
