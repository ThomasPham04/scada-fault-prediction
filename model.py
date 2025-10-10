import numpy as np
from tensorflow import keras 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping  # Tùy chọn: Dừng sớm nếu val loss không cải thiện
import os

# Load split info (from process_data.py)
train_batch_ids = np.load('preprocessing/train_batch_indices.npy')
val_batch_ids = np.load('preprocessing/val_batch_indices.npy')

print(f"Train batches: {len(train_batch_ids)}, Val batches: {len(val_batch_ids)}")

# Generator for train/val (load from saved batches)
def data_generator(batch_ids):
    while True:  # Loop for repeat data
        for b_id in batch_ids:
            x_file = f'preprocessing/batches/X_batch_{b_id}.npy'
            y_file = f'preprocessing/batches/y_batch_{b_id}.npy'
            if os.path.exists(x_file) and os.path.exists(y_file):
                Xb = np.load(x_file)
                yb = np.load(y_file)
                yield Xb, yb  # Yield batch
            else:
                print(f"Warning: Batch {b_id} files not found! Skipping.")
                continue  # Skip if file is error

# Build model (basic LSTM for binary classification: anomaly/normal)
# Hard-code num_features = 82 based on shape X_batch from process_data (window=6)
input_shape = (6, 82)
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=input_shape),
#     LSTM(50),
#     Dense(1, activation='sigmoid'),  # Output: probability anomaly (0-1)
#     Dropout(0.5),
#     Dense(1)
# ])
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(input_shape)))

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3rd Layer (Dense)
model.add(keras.layers.Dense(128, activation="relu"))

# 4th Layer (Dropout)
model.add(keras.layers.Dropout(0.5))

# Final Output Layer
model.add(keras.layers.Dense(1))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # binary (status_type_id: 0=normal, 1=anomaly?)
    metrics=['accuracy']
)

print(f"Model input shape: {input_shape}")
model.summary()  # Print the model architecture

# Generators
train_gen = data_generator(train_batch_ids)
val_gen = data_generator(val_batch_ids)

# Steps per epoch: Number of batches
steps_train = len(train_batch_ids)
steps_val = len(val_batch_ids)

# Callbacks (Optional)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
print("====================Training model=========================")
history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    epochs=20,
    epochs=20,
    validation_data=val_gen,
    validation_steps=steps_val,
    callbacks=[early_stop],
    verbose=1  # In progress
)

# Save model
model.save('lstm_scada_model.h5')
print("Model saved!")