import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# ============================================= CONFIG =============================================
MODEL_PATH = 'lstm_scada_model.h5'
BATCHES_DIR = 'preprocessing/batches'
VAL_SPLIT_PATH = 'preprocessing/val_batch_indices.npy'

# ============================================= LOAD MODEL =============================================
print(f"[Predict] Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ============================================= LOAD VALIDATION DATA =============================================
val_batch_ids = np.load(VAL_SPLIT_PATH)
print(f"[Predict] Found {len(val_batch_ids)} validation batches.")

X_list, y_list = [], []
for b_id in val_batch_ids:
    x_file = os.path.join(BATCHES_DIR, f'X_batch_{b_id}.npy')
    y_file = os.path.join(BATCHES_DIR, f'y_batch_{b_id}.npy')
    if not (os.path.exists(x_file) and os.path.exists(y_file)):
        print(f"⚠️  Warning: Batch {b_id} missing, skipping.")
        continue

    Xb = np.load(x_file)
    yb = np.load(y_file)
    if yb.dtype == object:  # convert string labels if needed
        yb = np.array([1 if str(v).lower() == 'anomaly' else 0 for v in yb])
    X_list.append(Xb)
    y_list.append(yb)

# Combine all batches into single arrays
X_test = np.concatenate(X_list, axis=0)
y_test = np.concatenate(y_list, axis=0)
print(f"[Predict] Loaded test set: X={X_test.shape}, y={y_test.shape}")

# ============================================= MAKE PREDICTIONS =============================================
print("[Predict] Running inference...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# ============================================= METRICS =============================================
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================= VISUALIZATION =============================================
# Compare predicted vs actual labels
plt.figure(figsize=(12, 5))
plt.plot(y_test[:200], label="Actual", marker='o', alpha=0.7)
plt.plot(y_pred[:200], label="Predicted", marker='x', alpha=0.7)
plt.title("Predicted vs Actual (first 200 samples)")
plt.legend()
plt.show()

print("✅ Prediction and evaluation complete.")
