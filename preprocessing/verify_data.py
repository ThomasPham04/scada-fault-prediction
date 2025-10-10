import pandas as pd
import numpy as np
import os
from numpy.lib.stride_tricks import sliding_window_view

# Configuration
CSV_FILE = 'df_all_clean.csv'
BATCH_DIR = 'batches'
TOTAL_BATCHES = 585
WINDOW_SIZE = 6

# Load full CSV and pre-compute arrays
def load_df_all(): 
    df_all = pd.read_csv(CSV_FILE)
    print(f"[verify_data] Full DF shape: {df_all.shape}")
    return df_all

# Extract features and label
def extract_features_labels(df_all):
    feature_cols = [col for col in df_all.columns if "sensor" in col or "power" in col or "wind_speed" in col]
    all_features = feature_cols + ['asset_id']
    label_col = 'status_type_id'
    print(f"[verify_data] Features: {len(all_features)} (Expected: 82)")

    arr_feat = df_all[all_features].to_numpy()  # (rows, features)
    arr_label = df_all[label_col].to_numpy()     # (rows,)
    print(f"[verify_data] Full arr_feat shape: {arr_feat.shape}, arr_label: {arr_label.shape}")
    
    return arr_feat, arr_label

# vectorized sliding window (fixed transpose)
def create_sliding_windows_vectorized(arr_feat, arr_label, start_idx, num_samples):
    end_idx = start_idx + num_samples + WINDOW_SIZE
    if end_idx > len(arr_feat):
        num_samples = len(arr_feat) - start_idx - WINDOW_SIZE
        print(f"[verify_data] Decreasing num_samples to {num_samples} (end of data)")
    
    feat_slice = arr_feat[start_idx:end_idx]  # (nrows_chunk, features)
    label_slice = arr_label[start_idx + WINDOW_SIZE : start_idx + WINDOW_SIZE + num_samples]
    
    # Sliding window on axis=0 (rows): shape (num_samples, features, window) → Transpose to (num_samples, window, features)
    X_view = sliding_window_view(feat_slice, WINDOW_SIZE, axis=0)[:num_samples]
    X = np.transpose(X_view, (0, 2, 1))  # Swap: (samples, features, window) → (samples, window, features)
    
    return X, label_slice

# Loop verify
def verify_matching(arr_feat, arr_label):
    offset = 0
    verification_results = []
    all_match = True
    mismatch_details = []

    for b_id in range(TOTAL_BATCHES):
        x_file = os.path.join(BATCH_DIR, f'X_batch_{b_id}.npy')
        y_file = os.path.join(BATCH_DIR, f'y_batch_{b_id}.npy')
        
        if os.path.exists(x_file) and os.path.exists(y_file):
            X_saved = np.load(x_file)
            y_saved = np.load(y_file)
            saved_samples = len(y_saved)
            
            # Recreate from array slice
            X_recreate, y_recreate = create_sliding_windows_vectorized(arr_feat, arr_label, offset, saved_samples)
            
            # Compare
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
                print(f"[verify_data] Batch {b_id}: Samples={saved_samples}, X match={content_match_X}, y match={content_match_y}")
            elif b_id % 50 == 0:
                if b_id != 0:
                    print(f"[verify_data] Batch {b_id - 50} - {b_id}: Samples={saved_samples}, All match OK")
            elif match and b_id == TOTAL_BATCHES - 1:
                print(f"[verify_data] Batch 550 - {b_id}: Samples={saved_samples}, All match OK")
            offset += saved_samples
        else:
            print(f"[verify_data] Batch {b_id}: File not found")

    # Summarize
    total_verified = len(verification_results)
    total_samples = sum(r['samples'] for r in verification_results)
    num_mismatch = len(mismatch_details)

    print(f"\n======== Full Verification Summary ========")
    print(f"[verify_data] Number of batches verified: {total_verified}/{TOTAL_BATCHES}")
    print(f"[verify_data] Number of samples verified: {total_samples}")
    print(f"[verify_data] Mismatch batches: {num_mismatch} (IDs: {mismatch_details[:10]}... if > 10)")
    print(f"[verify_data] All data aligned: {all_match}")

    if all_match:
        print("[verify_data] All features match labels – Data is ready to train for Model!")
    else:
        print("[verify_data] Data is mismatched -> Please double-check process_data.py before training Model!")
    
# Main execution
if __name__ == "__main__":
    print("================ Starting to Verify Data ================\n")
    
    # Load df_all
    df_all = load_df_all()
    
    # Extract Features and Labels
    arr_feat, arr_label = extract_features_labels(df_all)
    
    # Verify them
    verify_matching(arr_feat, arr_label)
    
    # Finish Data Verification 
    print("\n=> Data Verification is finished")
