"""
LSTMModel — models.architectures.lstm
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
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DROPOUT_RATE,
    DENSE_UNITS,
    LEARNING_RATE,
)


class LSTMModel:
    """
    Stacked LSTM prediction model for next-timestep forecasting.

    The anomaly signal is the reconstruction/prediction error (MAE) on the
    test set: high error → the current sensor behaviour deviates from the
    normal pattern learned during training.

    Architecture:
        Input → LSTM(units_1) → Dropout → LSTM(units_2) → Dropout
              → Dense(dense_units, relu) → Dense(output_dim, linear)

    Args:
        input_shape: Tuple (window_size, n_features).
        output_dim:  Number of output features (= n_features).
        units_1:     Hidden units for the first LSTM layer.
        units_2:     Hidden units for the second LSTM layer.
        dropout:     Dropout rate applied after each LSTM layer.
        dense_units: Units in the intermediate Dense layer.
        learning_rate: Adam learning rate.
    """

    def __init__(
        self,
        input_shape: tuple,
        output_dim: int,
        units_1: int = LSTM_UNITS_1,
        units_2: int = LSTM_UNITS_2,
        dropout: float = DROPOUT_RATE,
        dense_units: int = DENSE_UNITS,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self.input_shape   = input_shape
        self.output_dim    = output_dim
        self.units_1       = units_1
        self.units_2       = units_2
        self.dropout       = dropout
        self.dense_units   = dense_units
        self.learning_rate = learning_rate

    def build(self) -> Model:
        """
        Construct and compile the Keras model.

        Returns:
            Compiled Keras Model ready for training.
        """
        inputs = layers.Input(shape=self.input_shape, name='input_sequence')

        x = layers.LSTM(self.units_1, return_sequences=True, name='lstm_1')(inputs)
        x = layers.Dropout(self.dropout, name='dropout_1')(x)

        x = layers.LSTM(self.units_2, return_sequences=False, name='lstm_2')(x)
        x = layers.Dropout(self.dropout, name='dropout_2')(x)

        x = layers.Dense(self.dense_units, activation='relu', name='dense_intermediate')(x)
        outputs = layers.Dense(self.output_dim, activation='linear', name='output_prediction')(x)

        model = Model(inputs=inputs, outputs=outputs, name='LSTM')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        model.summary()
        return model


# ---------------------------------------------------------------------------
# Backward-compatible module-level alias
# ---------------------------------------------------------------------------

def build_lstm_model(input_shape: tuple, output_dim: int) -> Model:
    """Legacy alias — wraps LSTMModel(...).build()."""
    return LSTMModel(input_shape=input_shape, output_dim=output_dim).build()
