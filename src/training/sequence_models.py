"""
sequence_models.py - training.sequence_models
Keras model builders and training callbacks for sequence classifiers.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers

from training.hyperparameters.classifier_defaults import (
    CLASSIFIER_CALLBACK_MODE,
    CLASSIFIER_CALLBACK_MONITOR,
    CLASSIFIER_CNN_FILTERS_1,
    CLASSIFIER_CNN_FILTERS_2,
    CLASSIFIER_CNN_KERNEL_SIZE_1,
    CLASSIFIER_CNN_KERNEL_SIZE_2,
    CLASSIFIER_CNN_POOL_SIZE,
    CLASSIFIER_CNN_RECURRENT_UNITS,
    CLASSIFIER_DENSE_UNITS,
    CLASSIFIER_EARLY_STOPPING_PATIENCE,
    CLASSIFIER_GRU_UNITS_1,
    CLASSIFIER_GRU_UNITS_2,
    CLASSIFIER_LSTM_UNITS_1,
    CLASSIFIER_LSTM_UNITS_2,
    CLASSIFIER_MIN_LEARNING_RATE,
    CLASSIFIER_REDUCE_LR_FACTOR,
    CLASSIFIER_REDUCE_LR_PATIENCE,
    DEFAULT_CLASSIFIER_DROPOUT,
    DEFAULT_CLASSIFIER_FOCAL_ALPHA,
    DEFAULT_CLASSIFIER_FOCAL_GAMMA,
    DEFAULT_CLASSIFIER_L2,
    DEFAULT_CLASSIFIER_LEARNING_RATE,
    DEFAULT_CLASSIFIER_LOSS,
    DEFAULT_CONV_DROPOUT,
    DEFAULT_RECURRENT_DROPOUT,
)


def build_classifier_model(
    model_name: str,
    input_shape: tuple,
    learning_rate: float = DEFAULT_CLASSIFIER_LEARNING_RATE,
    dropout_rate: float | None = DEFAULT_CLASSIFIER_DROPOUT,
    l2_strength: float = DEFAULT_CLASSIFIER_L2,
    loss_name: str = DEFAULT_CLASSIFIER_LOSS,
    focal_gamma: float = DEFAULT_CLASSIFIER_FOCAL_GAMMA,
    focal_alpha: float = DEFAULT_CLASSIFIER_FOCAL_ALPHA,
):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if dropout_rate is not None and not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1).")
    if l2_strength < 0:
        raise ValueError("l2_strength must be non-negative.")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative.")
    if not 0.0 <= focal_alpha <= 1.0:
        raise ValueError("focal_alpha must be in [0, 1].")

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    regularizer = regularizers.l2(l2_strength) if l2_strength > 0 else None
    recurrent_dropout = DEFAULT_RECURRENT_DROPOUT if dropout_rate is None else dropout_rate
    conv_dropout = DEFAULT_CONV_DROPOUT if dropout_rate is None else dropout_rate

    if model_name == "lstm":
        x = layers.LSTM(
            CLASSIFIER_LSTM_UNITS_1,
            return_sequences=True,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(inputs)
        x = layers.Dropout(recurrent_dropout)(x)
        x = layers.LSTM(
            CLASSIFIER_LSTM_UNITS_2,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    elif model_name == "gru":
        x = layers.GRU(
            CLASSIFIER_GRU_UNITS_1,
            return_sequences=True,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(inputs)
        x = layers.Dropout(recurrent_dropout)(x)
        x = layers.GRU(
            CLASSIFIER_GRU_UNITS_2,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    elif model_name == "cnn_lstm":
        x = layers.Conv1D(
            CLASSIFIER_CNN_FILTERS_1,
            CLASSIFIER_CNN_KERNEL_SIZE_1,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(inputs)
        x = layers.MaxPooling1D(pool_size=CLASSIFIER_CNN_POOL_SIZE)(x)
        x = layers.Dropout(conv_dropout)(x)
        x = layers.Conv1D(
            CLASSIFIER_CNN_FILTERS_2,
            CLASSIFIER_CNN_KERNEL_SIZE_2,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.MaxPooling1D(pool_size=CLASSIFIER_CNN_POOL_SIZE)(x)
        x = layers.LSTM(
            CLASSIFIER_CNN_RECURRENT_UNITS,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    elif model_name == "cnn_gru":
        x = layers.Conv1D(
            CLASSIFIER_CNN_FILTERS_1,
            CLASSIFIER_CNN_KERNEL_SIZE_1,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(inputs)
        x = layers.MaxPooling1D(pool_size=CLASSIFIER_CNN_POOL_SIZE)(x)
        x = layers.Dropout(conv_dropout)(x)
        x = layers.Conv1D(
            CLASSIFIER_CNN_FILTERS_2,
            CLASSIFIER_CNN_KERNEL_SIZE_2,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizer,
        )(x)
        x = layers.MaxPooling1D(pool_size=CLASSIFIER_CNN_POOL_SIZE)(x)
        x = layers.GRU(
            CLASSIFIER_CNN_RECURRENT_UNITS,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)
        x = layers.Dropout(recurrent_dropout)(x)
    else:
        raise ValueError(f"Unsupported classifier model: {model_name}")

    x = layers.Dense(
        CLASSIFIER_DENSE_UNITS,
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    if loss_name == "binary_crossentropy":
        loss = "binary_crossentropy"
    elif loss_name == "focal":
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    else:
        raise ValueError("loss_name must be 'binary_crossentropy' or 'focal'")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def classifier_callbacks(model_path: Path):
    return [
        callbacks.EarlyStopping(
            monitor=CLASSIFIER_CALLBACK_MONITOR,
            mode=CLASSIFIER_CALLBACK_MODE,
            patience=CLASSIFIER_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=CLASSIFIER_CALLBACK_MONITOR,
            mode=CLASSIFIER_CALLBACK_MODE,
            factor=CLASSIFIER_REDUCE_LR_FACTOR,
            patience=CLASSIFIER_REDUCE_LR_PATIENCE,
            min_lr=CLASSIFIER_MIN_LEARNING_RATE,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=CLASSIFIER_CALLBACK_MONITOR,
            mode=CLASSIFIER_CALLBACK_MODE,
            save_best_only=True,
            verbose=0,
        ),
    ]
