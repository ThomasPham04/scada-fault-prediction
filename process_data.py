import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load data from csv file 
path = "Dataset/Wind_Farm_A/datasets/"   
all_files = glob.glob(os.path.join(path, "*.csv"))

df_list = []
for filename in all_files:
    temp_df = pd.read_csv(filename, sep=";")
    event_id = int(os.path.splitext(os.path.basename(filename))[0])
    temp_df["event_id"] = event_id
    df_list.append(temp_df)

df_all = pd.concat(df_list, ignore_index=True)
print("Loaded data shape:", df_all.shape)

# merge event_info
event_info = pd.read_csv("Dataset/Wind_Farm_A/event_info.csv", sep=";")
df_all = df_all.merge(event_info[['event_id', 'event_label']],
                      on='event_id', how='left')

print("Unique labels:", df_all['event_label'].unique())

# choose feature
feature_cols = [col for col in df_all.columns
                if "sensor" in col or "power" in col or "wind_speed" in col]

df_all = df_all[['time_stamp', 'asset_id', 'status_type_id', 
                 'event_id', 'event_label'] + feature_cols]

# Fill missing values (numeric only)
num_cols = df_all[feature_cols].select_dtypes(include=[np.number]).columns
df_all[num_cols] = df_all[num_cols].interpolate().ffill().bfill()


scaler = StandardScaler()
df_all[num_cols] = scaler.fit_transform(df_all[num_cols])

print("Data after cleaning & scaling:", df_all.shape)

# Return batch (X, y) to train without loading everything into RAM.
def sliding_window_generator(data, feature_columns, label_column, window_size=6, batch_size=1024):
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

# test generator
all_features = feature_cols + ['asset_id']
gen = sliding_window_generator(df_all, all_features, 'status_type_id', window_size=6, batch_size=512)

X_batch, y_batch = next(gen)
print("One batch X shape:", X_batch.shape)
print("One batch y shape:", y_batch.shape)

# output to csv file
df_all.to_csv("df_all_clean.csv", index=False)
print("CSV converted")


X, y = [], []
for Xb, yb in sliding_window_generator(df_all, all_features, 'status_type_id', window_size=6, batch_size=2048):
    X.append(Xb)
    y.append(yb)

X = np.vstack(X)
y = np.hstack(y)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

#train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)

