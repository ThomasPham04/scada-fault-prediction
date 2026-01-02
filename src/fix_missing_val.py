
import os
import numpy as np
import sys

# Define validation data path manually or relative to script
# Assuming script is run from src/model or similar, but we'll use absolute paths based on knowledge
BASE_DIR = r"d:\Final Project\scada-fault-prediction"
NBM_DIR = os.path.join(BASE_DIR, "Dataset", "processed", "Wind Farm A", "NBM_7day")
VAL_DIR = os.path.join(NBM_DIR, "val_by_event")

def fix_val_event():
    print(f"Checking directory: {NBM_DIR}")
    if not os.path.exists(NBM_DIR):
        print("Error: NBM_7day directory not found!")
        return

    x_val_path = os.path.join(NBM_DIR, "X_val.npy")
    y_val_path = os.path.join(NBM_DIR, "y_val.npy")

    if not os.path.exists(x_val_path) or not os.path.exists(y_val_path):
        print("Error: X_val.npy or y_val.npy not found!")
        return

    print("Loading validation data...")
    X_val = np.load(x_val_path)
    y_val = np.load(y_val_path)
    print(f"Loaded X_val: {X_val.shape}, y_val: {y_val.shape}")

    print(f"Creating directory: {VAL_DIR}")
    os.makedirs(VAL_DIR, exist_ok=True)

    val_event_id = 999
    val_file = os.path.join(VAL_DIR, f"event_{val_event_id}.npz")
    
    print(f"Saving {val_file}...")
    np.savez(
        val_file,
        X=X_val,
        y=y_val,
        label='normal'
    )
    print("Success! val_by_event restored.")

if __name__ == "__main__":
    fix_val_event()
