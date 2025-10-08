import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping  # Tùy chọn: Dừng sớm nếu val loss không cải thiện
import os

# Load split info (từ process_data.py đã lưu)
train_batch_ids = np.load('train_batch_indices.npy')
val_batch_ids = np.load('val_batch_indices.npy')

print(f"Train batches: {len(train_batch_ids)}, Val batches: {len(val_batch_ids)}")

# Generator cho train/val (load từ batches đã lưu, không cần df_all hay generator gốc)
def data_generator(batch_ids):
    while True:  # Vô hạn loop để repeat data
        for b_id in batch_ids:
            x_file = f'batches/X_batch_{b_id}.npy'
            y_file = f'batches/y_batch_{b_id}.npy'
            if os.path.exists(x_file) and os.path.exists(y_file):
                Xb = np.load(x_file)
                yb = np.load(y_file)
                yield Xb, yb  # Yield batch
            else:
                print(f"Warning: Batch {b_id} files not found! Skipping.")
                continue  # Skip nếu file lỗi

# Build model (LSTM cơ bản cho binary classification: anomaly/normal)
# Hard-code num_features = 82 dựa trên shape X_batch từ process_data (window=6)
input_shape = (6, 82)
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=input_shape),
    LSTM(50),
    Dense(1, activation='sigmoid')  # Output: probability anomaly (0-1)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Phù hợp binary (status_type_id: 0=normal, 1=anomaly?)
    metrics=['accuracy']
)

print(f"Model input shape: {input_shape}")
model.summary()  # In kiến trúc model

# Tạo generators
train_gen = data_generator(train_batch_ids)
val_gen = data_generator(val_batch_ids)

# Steps per epoch: Số batches (vì mỗi yield là 1 batch)
steps_train = len(train_batch_ids)
steps_val = len(val_batch_ids)

# Callbacks (tùy chọn: Dừng nếu val loss không cải thiện)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
print("====================Training model=========================")
history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    epochs=1,
    validation_data=val_gen,
    validation_steps=steps_val,
    callbacks=[early_stop],
    verbose=1  # In progress
)

# Lưu model
model.save('lstm_scada_model.h5')
print("Model saved! Plot history nếu cần: import matplotlib.pyplot as plt; plt.plot(history.history['loss']); plt.show()")