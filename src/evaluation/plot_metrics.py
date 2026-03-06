import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import joblib

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import PER_ASSET_PROCESSED_DIR, MODELS_DIR
def smooth_mae(mae: np.ndarray, window: int = 3) -> np.ndarray:
    if len(mae) < window: return mae
    return np.convolve(mae, np.ones(window)/window, mode='same')

plt.style.use('default')

def plot_asset_metrics(model_type, asset_id, test_event_id=None):
    """
    Generate the two subplots for a given model and asset.
    1. Residual of Prediction Data (Test Event)
    2. Residual During Training (Train Data)
    """
    print(f"Generating plots for {model_type.upper()} - Asset {asset_id}")
    
    # Paths
    asset_dir = os.path.join(PER_ASSET_PROCESSED_DIR, f"asset_{asset_id}")
    plot_dir = os.path.join(ROOT, "results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load Model
    if model_type == "lstm":
        from tensorflow.keras.models import load_model
        model_path = os.path.join(MODELS_DIR, f"lstm_asset_{asset_id}.keras")
        model = load_model(model_path)
        
        # Determine threshold
        val_mae_path = os.path.join(asset_dir, "X_val.npy")
        y_val_path = os.path.join(asset_dir, "y_val.npy")
        if os.path.exists(val_mae_path) and os.path.exists(y_val_path):
            X_val = np.load(val_mae_path)
            y_val = np.load(y_val_path)
            val_pred = model.predict(X_val, verbose=0, batch_size=128)
            val_mae = np.mean(np.abs(y_val - val_pred), axis=1)
            val_mae = smooth_mae(val_mae, window=3)
            threshold = np.percentile(val_mae, 85)
        else:
            threshold = 0.5
            
    else: # Tree models
        model_path = os.path.join(MODELS_DIR, f"{model_type}_asset_{asset_id}.pkl")
        data = joblib.load(model_path)
        model = data["model"]
        threshold = data.get("threshold", 0.5)
        feature_mode = data.get("feature_mode", "statistical")
        
        from src.data_pipeline.loaders.tabular_loader import TabularLoader
        loader = TabularLoader(use_stats=(feature_mode == "statistical"))
        fn = loader.compute_statistical_features if loader.use_stats else loader.flatten_sequences

    # ==========================================
    # 1. Load and process Train Data
    # ==========================================
    X_train_raw = np.load(os.path.join(asset_dir, "X_train.npy"))
    y_train_raw = np.load(os.path.join(asset_dir, "y_train.npy"))
    
    # We need to load event_info to distinguish normal/anomaly in training
    event_info_path = os.path.join(ROOT, "Dataset", "raw", "Wind Farm A", "event_info.csv")
    ei = pd.read_csv(event_info_path, sep=';')
    asset_events = ei[ei['asset'] == int(asset_id) if asset_id.isdigit() else asset_id]
    
    # Calculate scores on training data
    if model_type == "lstm":
        train_pred = model.predict(X_train_raw, verbose=0, batch_size=256)
        train_scores = np.mean(np.abs(y_train_raw - train_pred), axis=1)
        train_scores = smooth_mae(train_scores, window=3)
    else:
        X_train_flat = fn(X_train_raw)
        train_scores = model.predict_proba(X_train_flat)[:, 1]

    # For training data, we just distribute the points across an arbitrary X axis
    # Get true train labels derived from the raw datasets and `status_type_id`:
    # Normal = 0, 2. Anomaly = 1, 3, 4, 5
    metadata_path = os.path.join(asset_dir, "metadata.pkl")
    meta = joblib.load(metadata_path)
    window_size = meta.get("window_size", 10)
    
    status_list = []
    for ev_id in asset_events['event_id']:
        csv_path = os.path.join(ROOT, "Dataset", "raw", "Wind Farm A", "datasets", f"{ev_id}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=';')
            df_train = df[df['train_test'] == 'train']
            if len(df_train) >= window_size:
                status_seq = df_train['status_type_id'].values[window_size - 1:]
                # 1, 3, 4, 5 are anomalies -> map to 1
                anomaly_mask = np.isin(status_seq, [1, 3, 4, 5]).astype(int)
                status_list.append(anomaly_mask)
                
    if status_list:
        y_train_bin = np.concatenate(status_list)
        # Match lengths in case of any dropped sequences during preprocessing
        if len(y_train_bin) > len(train_scores):
            y_train_bin = y_train_bin[:len(train_scores)]
        elif len(y_train_bin) < len(train_scores):
            y_train_bin = np.pad(y_train_bin, (0, len(train_scores) - len(y_train_bin)))
    else:
        y_train_bin = np.zeros(len(train_scores))
        
    train_x = np.arange(len(train_scores))
    normal_idx = (y_train_bin == 0)
    anomaly_idx = (y_train_bin == 1)

    # ==========================================
    # 2. Load and process Test Event Data
    # ==========================================
    test_dir = os.path.join(asset_dir, "test_by_event")
    event_files = sorted(glob.glob(os.path.join(test_dir, "event_*.npz")))
    
    if test_event_id:
        # filter to specific
        event_files = [f for f in event_files if f"event_{test_event_id}.npz" in f]
    
    if not event_files:
        print(f"No test events found for asset {asset_id}")
        return
        
    for ef in event_files:
        evt_name = os.path.basename(ef).replace("event_", "").replace(".npz", "")
        print(f"Plotting Test Event: {evt_name}")
        npz = np.load(ef, allow_pickle=True)
        X_test = npz["X"]
        y_test = npz["y"]
        ts_arr = npz.get("time_stamps", [])
        label = str(npz["label"])
        
        if len(ts_arr) == len(X_test):
            # Convert strings to datetime
            dates = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in ts_arr]
        else:
            dates = np.arange(len(X_test))
            
        if model_type == "lstm":
            test_pred = model.predict(X_test, verbose=0, batch_size=256)
            raw_mae = np.mean(np.abs(y_test - test_pred), axis=1)
            test_scores = smooth_mae(raw_mae, window=3)
        else:
            X_test_flat = fn(X_test)
            test_scores = model.predict_proba(X_test_flat)[:, 1]

        # ==========================================
        # 3. Create the Plot
        # ==========================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.patch.set_facecolor('white')
        
        # ------------------
        # Subplot 1: Test Data
        # ------------------
        ax1.scatter(dates, test_scores, s=2, c='tab:blue')
        ax1.axhline(threshold, color='black', linestyle='--', label='threshold')
        ax1.set_yscale('log')
        ax1.set_title(f"Residual of Prediction Data (Event {evt_name})", fontsize=14)
        ax1.set_ylabel("residual", fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        
        # Highlight regions where the model detects an anomaly (residual > threshold)
        pred_anomaly = (test_scores > threshold)
        if np.any(pred_anomaly):
            # Using fill_between to highlight the background in red for predicted anomalies
            # We define the fill to cover the full height (0 to 1 in axes coordinates)
            ax1.fill_between(dates, 0, 1, where=pred_anomaly, color='red', alpha=0.1, 
                             transform=ax1.get_xaxis_transform(), label='predicted anomaly')
        
        if isinstance(dates[0], datetime):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.get_xticklabels(), rotation=0)

        # ------------------
        # Subplot 2: Train Data
        # ------------------
        ax2.scatter(train_x[normal_idx], train_scores[normal_idx], s=2, c='tab:blue', label='normal status')
        if np.sum(anomaly_idx) > 0:
            ax2.scatter(train_x[anomaly_idx], train_scores[anomaly_idx], s=2, c='tab:orange', label='anomalous status')
        
        ax2.axhline(threshold, color='black', linestyle='--', label='threshold')
        ax2.set_yscale('log')
        ax2.set_title("Residual During Training", fontsize=14)
        ax2.set_ylabel("residual", fontsize=12)
        
        # Hide x-axis labels for training since it's just sequential index here
        ax2.set_xticks([]) 
        ax2.legend(loc='upper left', fontsize=12)

        plt.tight_layout()
        
        out_path = os.path.join(plot_dir, f"{model_type}_asset_{asset_id}_evt_{evt_name}_analysis.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {out_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "random_forest", "xgboost"])
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--event", type=str, default=None)
    
    args = parser.parse_args()
    plot_asset_metrics(args.model, args.asset, args.event)
