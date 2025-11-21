import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Constants
TEST_SIZE = 0.2
WINDOW_SIZE = 6
BATCH_SIZE_GEN = 2048
BATCH_SIZE_TEST = 512 
DATA_PATH = "Dataset/Wind_Farm_A/datasets/"
EVENT_INFO_PATH = "Dataset/Wind_Farm_A/event_info.csv"
OUTPUT_CSV = "df_all_clean.csv"
BATCHES_DIR = "batches"
TRAIN_INDICES = "train_batch_indices.npy"
VAL_INDICES = "val_batch_indices.npy"

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load all CSV files and concat into df_all."""
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    if len(all_files) == 0:
        raise ValueError(f"[process_data] No CSV files found in {data_path}")
    
    df_list = []
    for filename in all_files:
        temp_df = pd.read_csv(filename, sep=";")
        event_id = int(os.path.splitext(os.path.basename(filename))[0])
        temp_df["event_id"] = event_id
        df_list.append(temp_df)
    
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"[process_data] Loaded data shape: {df_all.shape}")
    return df_all

def merge_event_info(df: pd.DataFrame, event_info_path: str) -> pd.DataFrame:
    """Merge event_info with df_all."""
    event_info = pd.read_csv(event_info_path, sep=";")
    df_merged = df.merge(event_info[['event_id', 'event_label']], on='event_id', how='left')
    print(f"[process_data] Unique labels: {df_merged['event_label'].unique()}")
    return df_merged

def select_and_clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Select features, fill NaN, scale numeric cols."""
    feature_cols = [col for col in df.columns if "sensor" in col or "power" in col or "wind_speed" in col]
    df_selected = df[['time_stamp', 'asset_id', 'status_type_id', 'event_id', 'event_label'] + feature_cols].copy()  # Force copy
    
    # Fill NaN numeric
    num_cols = df_selected[feature_cols].select_dtypes(include=[np.number]).columns
    df_selected.loc[:, num_cols] = df_selected[num_cols].interpolate().ffill().bfill() 
    
    # Scale
    scaler = StandardScaler()
    df_selected.loc[:, num_cols] = scaler.fit_transform(df_selected[num_cols])
    
    # Encode event_label
    df_selected['event_label_encoded'] = df_selected['event_label'].map({'anomaly': 0, 'normal': 1})  
    
    print(f"[process_data] Data after cleaning & scaling: {df_selected.shape}")
    # print(df_selected)
    return df_selected, feature_cols

def sliding_window_generator(data: pd.DataFrame, feature_columns: list, label_column: str, 
                             window_size: int = WINDOW_SIZE, batch_size: int = BATCH_SIZE_GEN):
    """Generate sliding windows in batches (yield X, y)."""
    arr_feat = data[feature_columns].to_numpy()
    arr_label = data[label_column].to_numpy()
    n = len(data)

    for i in range(0, n - window_size, batch_size):
        X_batch, y_batch = [], []
        end = min(i + batch_size, n - window_size)
        for j in range(i, end):
            X_batch.append(arr_feat[j:j+window_size])
            y_batch.append(arr_label[j+window_size])
        yield np.array(X_batch), np.array(y_batch)

def test_generator(df: pd.DataFrame, feature_cols: list, label_col: str) -> None:
    """Test sliding window generator."""
    all_features = feature_cols + ['asset_id']
    gen = sliding_window_generator(df, all_features, label_col, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE_TEST)
    X_batch, y_batch = next(gen)
    print(f"[process_data] One batch X shape: {X_batch.shape}")
    print(f"[process_data] One batch y shape: {y_batch.shape}")

def save_all_batches(df: pd.DataFrame, feature_cols: list, label_col: str, batches_dir: str) -> tuple[int, list, int]:
    """Save all sliding windows as .npy batches."""
    os.makedirs(batches_dir, exist_ok=True)
    all_features = feature_cols + ['asset_id']
    
    batch_idx = 0
    total_samples = 0
    batch_shapes = []
    gen = sliding_window_generator(df, all_features, label_col, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE_GEN)
    
    print("[process_data] Saving batches...")
    for Xb, yb in gen:
        np.save(os.path.join(batches_dir, f'X_batch_{batch_idx}.npy'), Xb)
        np.save(os.path.join(batches_dir, f'y_batch_{batch_idx}.npy'), yb)
        total_samples += len(yb)
        batch_shapes.append((Xb.shape, yb.shape))
        batch_idx += 1
    
    print(f"[process_data] Total batches: {batch_idx}, Total samples: {total_samples}")
    return batch_idx, batch_shapes, total_samples

def train_test_split(batch_shapes: list, total_samples: int, test_size: float = TEST_SIZE) -> tuple[list, list, int, int]:
    """Split batch indices for train/val based on test_size."""
    n_val = int(test_size * total_samples)
    n_train = total_samples - n_val
    
    train_batches = []
    val_batches = []
    current_sample = 0
    
    for b_idx, (_, y_shape) in enumerate(batch_shapes):
        batch_size = y_shape[0]
        if current_sample + batch_size <= n_train:
            train_batches.append(b_idx)
            current_sample += batch_size
        else:
            val_batches.append(b_idx)
    
    return train_batches, val_batches, n_train, n_val

def save_split_info(train_batches: list, val_batches: list, train_samples: int, val_samples: int) -> None:
    """Save train/val batch indices as .npy."""
    np.save(TRAIN_INDICES, np.array(train_batches))
    np.save(VAL_INDICES, np.array(val_batches))
    print(f"[process_data] Train batches: {len(train_batches)} (samples: {train_samples})")
    print(f"[process_data] Val batches: {len(val_batches)} (samples: {val_samples})")

def save_clean_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned df to CSV."""
    df.to_csv(output_path, index=False)
    print(f"[process_data] CSV saved: {output_path}")

# Main execution
if __name__ == "__main__":
    print("================ Starting to process Data ================\n")
    
    # Load and merge
    df_all = load_raw_data(DATA_PATH)
    df_all = merge_event_info(df_all, EVENT_INFO_PATH)
    
    # Select, clean, scale
    df_all, feature_cols = select_and_clean_features(df_all)
    
    anomaly_total = 0
    normal_total = 0
    for b in range(584):  # Sample 5 batches
        yb = np.load(f'batches/y_batch_{b}.npy')
        anomaly_total += np.sum(yb == 1)
        normal_total += np.sum(yb == 0)
    print(f"[process_data] Sample anomaly/normal: {anomaly_total} / {normal_total}")
    
    # Balance data (thêm ở đây)
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
    num_features = df_all[feature_cols].to_numpy()
    label = df_all['event_label_encoded'].to_numpy().reshape(-1, 1)
    balanced_num_feat, balanced_label = ros.fit_resample(num_features, label)
    
    df_balanced = pd.DataFrame(balanced_num_feat, columns=feature_cols)
    df_balanced['event_label_encoded'] = balanced_label.flatten()
    df_balanced['time_stamp'] = df_all['time_stamp'].iloc[0]  # Dummy time (hoặc interpolate nếu cần)
    df_balanced['asset_id'] = df_all['asset_id'].iloc[0]  # Dummy
    df_balanced['status_type_id'] = df_all['status_type_id'].iloc[0]  # Dummy
    df_balanced['event_id'] = df_all['event_id'].iloc[0]  # Dummy
    df_balanced['event_label'] = np.where(balanced_label.flatten() == 1, 'anomaly', 'normal')
    
    print(f"[balance_data] Original shape: {df_all.shape}, Balanced shape: {df_balanced.shape}")
    print(f"Original ratio anomaly/normal: {(df_all['event_label_encoded'] == 1).mean():.2%}")
    print(f"Balanced ratio anomaly/normal: {(df_balanced['event_label_encoded'] == 1).mean():.2%}")
    
    # Dùng df_balanced cho gen
    df_all = df_balanced
    # Test generator
    # test_generator(df_all, feature_cols, 'status_type_id')
    
    # Save clean CSV
    # print(df_all)
    # save_clean_csv(df_all, OUTPUT_CSV)
    
    # Save batches
    num_batches, batch_shapes, total_samples = save_all_batches(df_all, feature_cols, 'event_label_encoded', BATCHES_DIR)
    
    # Split indices
    train_batches, val_batches, n_train, n_val = train_test_split(batch_shapes, total_samples, TEST_SIZE)
    
    # Save split
    save_split_info(train_batches, val_batches, n_train, n_val)
    
    print("\n=> Process Data is Finished")