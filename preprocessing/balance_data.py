import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

df = pd.read_csv("df_feature_selected.csv")
print(Counter(df['event_label']))

# Nếu imbalance: tính trọng số để dùng khi train model
classes = df['event_label'].unique()
weights = compute_class_weight('balanced', classes=classes, y=df['event_label'])
class_weights = dict(zip(classes, weights))
print("Trọng số class:", class_weights)
