"""Quick debug script to check normal data availability."""
import pandas as pd
import os

# Load one normal event to check
event_id = 25  # First normal event
file_path = "Dataset/raw/Wind Farm A/datasets/25.csv"

df = pd.read_csv(file_path, sep=';')
print(f"Event {event_id} - Total timesteps: {len(df)}")
print(f"Train/Test split:")
print(df['train_test'].value_counts())

# Filter for normal operation
mask_train = df['train_test'] == 'train'
mask_status = df['status_type'] == 0
mask_wind = df['wind_speed_3_avg'] > 4.0
mask_power = df['power_29_avg'] > 0

df_train = df[mask_train]
df_status0 = df[mask_train & mask_status]
df_with_wind = df[mask_train & mask_status & mask_wind]
df_full_filter = df[mask_train & mask_status & mask_wind & mask_power]

print(f"\nFiltering progression:")
print(f"  Original: {len(df)}")
print(f"  train_test=='train': {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
print(f"  + status_type==0: {len(df_status0)} ({len(df_status0)/len(df)*100:.1f}%)")
print(f"  + wind>4.0: {len(df_with_wind)} ({len(df_with_wind)/len(df)*100:.1f}%)")
print(f"  + power>0: {len(df_full_filter)} ({len(df_full_filter)/len(df)*100:.1f}%)")

print(f"\n14-day window needs {2016} consecutive timesteps")
print(f"Available after filtering: {len(df_full_filter)}")
print(f"Enough data? {len(df_full_filter) >= 2016}")

# Check largest consecutive runs
