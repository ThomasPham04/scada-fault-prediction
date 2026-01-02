"""
Replot LSTM Training History from TensorBoard Logs
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

def extract_scalars(log_dir):
    """Extract scalars manually handling V2 TensorBoard logs."""
    files = [f for f in os.listdir(log_dir) if 'events' in f]
    if not files:
        return {}
    
    event_file = os.path.join(log_dir, max(files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x))))
    print(f"Parsing {event_file}...")
    
    data = {}
    try:
        for e in summary_iterator(event_file):
            for v in e.summary.value:
                # Value could be simple_value (V1) or tensor (V2)
                val = None
                if v.HasField('simple_value'):
                    val = v.simple_value
                elif v.HasField('tensor'):
                    try:
                        val = tf.make_ndarray(v.tensor)
                        # Handle 0-d tensor
                        if val.ndim == 0:
                            val = float(val)
                    except:
                        pass
                
                if val is not None:
                    if v.tag not in data:
                        data[v.tag] = []
                    data[v.tag].append({'step': e.step, 'value': val})
    except Exception as e:
        print(f"Warning parsing {event_file}: {e}")
        
    return data

def main():
    log_dir = os.path.join(RESULTS_DIR, 'nbm_7day_logs')
    train_dir = os.path.join(log_dir, 'train')
    val_dir = os.path.join(log_dir, 'validation')
    
    print(f"Reading logs from: {log_dir}")
    
    train_data = extract_scalars(train_dir)
    val_data = extract_scalars(val_dir)
    
    print("Train tags:", train_data.keys())
    print("Val tags:", val_data.keys())
    
    history = {}
    
    # Process Train
    for tag, events in train_data.items():
        # Handle epoch_loss vs loss
        key = tag.replace('epoch_', '')
        # Only care about loss and mae
        if key in ['loss', 'mae']:
            history[key] = [x['value'] for x in events]
            
    # Process Val
    for tag, events in val_data.items():
        key = tag.replace('epoch_', '')
        if key in ['loss', 'mae']:
            history[f"val_{key}"] = [x['value'] for x in events]
            
    # Create DF
    # Note: Val logs might lag or have different frequency if not per epoch
    # Assuming standard Keras callback usage (per epoch)
    
    # Ensure lengths match
    max_len = max([len(v) for v in history.values()] or [0])
    for k in history:
        if len(history[k]) < max_len:
            print(f"Warning: padding {k} (len={len(history[k])}) to {max_len}")
            history[k] = history[k] + [history[k][-1]] * (max_len - len(history[k]))
            
    df = pd.DataFrame(history)
    print("\nExtracted Data:")
    print(df.head())
    print("...")
    print(df.tail())
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    if 'loss' in df.columns:
        ax1.plot(df.index + 1, df['loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    if 'val_loss' in df.columns:
        ax1.plot(df.index + 1, df['val_loss'], label='Validation Loss', linewidth=2, marker='o', markersize=4)
        
    ax1.set_title('Model Loss (MSE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle= '--', alpha=0.7)
    
    # MAE
    if 'mae' in df.columns:
        ax2.plot(df.index + 1, df['mae'], label='Train MAE', linewidth=2, marker='o', markersize=4, color='green')
    if 'val_mae' in df.columns:
        ax2.plot(df.index + 1, df['val_mae'], label='Validation MAE', linewidth=2, marker='o', markersize=4, color='orange')
        
    ax2.set_title('Model Metric (MAE)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('NBM LSTM 7-Day Model Training History', fontsize=16, y=0.98)
    
    # Save
    out_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'training_history_refined.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")

if __name__ == "__main__":
    main()
