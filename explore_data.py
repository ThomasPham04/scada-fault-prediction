import pandas as pd
import os

# Load event info
event_info = pd.read_csv('Dataset/raw/Wind Farm A/event_info.csv', sep=';')
print("=" * 80)
print("EVENT INFO")
print("=" * 80)
print(f"Total events: {len(event_info)}")
print(f"\nEvent labels:")
print(event_info['event_label'].value_counts())
print(f"\nFault types (anomalies only):")
print(event_info[event_info['event_label'] == 'anomaly']['event_description'].value_counts())

# Load a sample dataset
print("\n" + "=" * 80)
print("SAMPLE DATASET (event 0)")
print("=" * 80)
df = pd.read_csv('Dataset/raw/Wind Farm A/datasets/0.csv', sep=';')
print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)} total):")
for i, col in enumerate(df.columns[:20]):
    print(f"  {i}: {col}")
print(f"  ... and {len(df.columns) - 20} more columns")

print(f"\nFirst 3 rows (first 10 columns):")
print(df.iloc[:3, :10])

print(f"\nData types:")
print(df.dtypes.value_counts())

print(f"\nMissing values:")
missing = df.isnull().sum()
print(f"Columns with missing values: {(missing > 0).sum()}")
if (missing > 0).sum() > 0:
    print(missing[missing > 0].head(10))

print(f"\nTime range:")
df['time_stamp'] = pd.to_datetime(df['time_stamp'])
print(f"Start: {df['time_stamp'].min()}")
print(f"End: {df['time_stamp'].max()}")
print(f"Duration: {df['time_stamp'].max() - df['time_stamp'].min()}")

# Check train/test split if exists
if 'train_test' in df.columns:
    print(f"\nTrain/Test split:")
    print(df['train_test'].value_counts())
