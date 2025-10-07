import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv("../df_all_clean.csv")
feature_cols = [col for col in df.columns if "sensor" in col or "power" in col or "wind_speed" in col]

# 1. Loại bỏ đặc trưng có phương sai thấp
selector = VarianceThreshold(threshold=0.01)
selector.fit(df[feature_cols])
selected_features = df[feature_cols].columns[selector.get_support()]
print("Còn lại:", len(selected_features), "đặc trưng")

# 2. Loại bỏ tương quan cao
corr_matrix = df[selected_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df = df.drop(columns=to_drop)
print("Sau khi bỏ tương quan cao:", len(df.columns))

df.to_csv("df_feature_selected.csv", index=False)
