import pandas as pd
import numpy as np
import os
from numpy.lib.stride_tricks import sliding_window_view

# Config
csv_file = 'df_all_clean.csv'
batch_dir = 'batches'
total_batches = 585
window_size = 6

print("=== Starting to Verify Data using df_all_clean.csv file ... ===")

# Load full CSV và pre-compute arrays
df_all = pd.read_csv(csv_file)
print(f"Full DF shape: {df_all.shape}")

# Extract features và label
feature_cols = [col for col in df_all.columns if "sensor" in col or "power" in col or "wind_speed" in col]
all_features = feature_cols + ['asset_id']
label_col = 'status_type_id'
print(f"Features: {len(all_features)} (Expected: 82)")

arr_feat = df_all[all_features].to_numpy()  # (rows, features)
arr_label = df_all[label_col].to_numpy()     # (rows,)
print(f"Full arr_feat shape: {arr_feat.shape}, arr_label: {arr_label.shape}")

# Hàm vectorized sliding window (fixed transpose)
def create_sliding_windows_vectorized(arr_feat, arr_label, start_idx, num_samples):
    end_idx = start_idx + num_samples + window_size
    if end_idx > len(arr_feat):
        num_samples = len(arr_feat) - start_idx - window_size
        print(f"⚠️ Decreasing num_samples to {num_samples} (end of data)")
    
    feat_slice = arr_feat[start_idx:end_idx]  # (nrows_chunk, features)
    label_slice = arr_label[start_idx + window_size : start_idx + window_size + num_samples]
    
    # Sliding window on axis=0 (rows): shape (num_samples, features, window) → Transpose to (num_samples, window, features)
    X_view = sliding_window_view(feat_slice, window_size, axis=0)[:num_samples]
    X = np.transpose(X_view, (0, 2, 1))  # Swap: (samples, features, window) → (samples, window, features)
    
    return X, label_slice

# Loop verify
offset = 0
verification_results = []
all_match = True
mismatch_details = []

for b_id in range(total_batches):
    x_file = os.path.join(batch_dir, f'X_batch_{b_id}.npy')
    y_file = os.path.join(batch_dir, f'y_batch_{b_id}.npy')
    
    if os.path.exists(x_file) and os.path.exists(y_file):
        X_saved = np.load(x_file)
        y_saved = np.load(y_file)
        saved_samples = len(y_saved)
        
        # Recreate từ array slice
        X_recreate, y_recreate = create_sliding_windows_vectorized(arr_feat, arr_label, offset, saved_samples)
        
        # So sánh
        shape_match = X_recreate.shape == X_saved.shape and y_recreate.shape == y_saved.shape
        content_match_X = np.allclose(X_recreate, X_saved, atol=1e-6)
        content_match_y = np.allclose(y_recreate, y_saved, atol=1e-6)
        
        match = shape_match and content_match_X and content_match_y
        all_match = all_match and match
        
        verification_results.append({
            'batch_id': b_id,
            'samples': saved_samples,
            'shape_match': shape_match,
            'content_X_match': content_match_X,
            'content_y_match': content_match_y,
            'overall_match': match
        })
        if not match:
            mismatch_details.append(b_id)
            print(f"Batch {b_id}: Samples={saved_samples}, X match={content_match_X}, y match={content_match_y}")
        elif b_id % 50 == 0:
            if b_id != 0:
                print(f"Batch {b_id - 50} - {b_id}: Samples={saved_samples}, All match OK")
        elif match and b_id == 584:
            print(f"Batch 550 - {b_id}: Samples={saved_samples}, All match OK")
        offset += saved_samples
    else:
        print(f"⚠️ Batch {b_id}: File not found")

# Tóm tắt
total_verified = len(verification_results)
total_samples = sum(r['samples'] for r in verification_results)
num_mismatch = len(mismatch_details)

print(f"\n=== Full Verification Summary ===")
print(f"Number of batches verified: {total_verified}/{total_batches}")
print(f"Number of samples verified: {total_samples}")
print(f"Mismatch batches: {num_mismatch} (IDs: {mismatch_details[:10]}... if > 10)")
print(f"All data aligned: {all_match}")

if all_match:
    print("✅ All features match labels – Data is ready to train for Model!")
else:
    print("❌ Có mismatch – Kiểm tra process_data.py (e.g., scaler fit toàn bộ vs chunk).")
