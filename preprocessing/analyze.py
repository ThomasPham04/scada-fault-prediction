import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ============================
# 1. Load dữ liệu
# ============================
base_path = os.path.dirname(os.path.dirname(__file__))  # thư mục cha
csv_path = os.path.join(base_path, "df_all_clean.csv")

df = pd.read_csv(csv_path)
print("Dataset shape:", df.shape)

# # ============================
# # 2. Phân bố nhãn sự kiện
# # ============================
# plt.figure(figsize=(6, 4))
# sns.countplot(x='event_label', data=df, palette='Set2')
# plt.title("Phân bố nhãn sự kiện (Normal vs Anomaly)")
# plt.xlabel("Loại sự kiện")
# plt.ylabel("Số lượng mẫu")
# plt.tight_layout()
# plt.show()

# # ============================
# # 3. Trực quan hóa phân phối cảm biến
# # ============================
# sensor_cols = [col for col in df.columns if "sensor" in col or "power" in col or "wind_speed" in col]
# print(f"Tổng số cột cảm biến: {len(sensor_cols)}")

# # Lấy ngẫu nhiên 6 cảm biến đại diện để vẽ
# sample_sensors = sensor_cols[:6]

# for col in sample_sensors:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[col], kde=True, bins=40, color='steelblue')
#     plt.title(f'Phân bố giá trị của {col}')
#     plt.xlabel("Giá trị sau chuẩn hóa (StandardScaler)")
#     plt.ylabel("Tần suất xuất hiện")
#     plt.tight_layout()
#     plt.show()

# ============================
# 4. Thống kê cơ bản & lưu lại
# ============================
desc = df.describe()
# desc.to_csv("df_describe.csv", index=True)
# print("Đã lưu thống kê mô tả vào df_describe.csv")

# ============================
# 5. Hiển thị thống kê trung bình và độ lệch chuẩn
# ============================
desc_t = desc.T
desc_t.reset_index(inplace=True)
desc_t.rename(columns={'index': 'feature'}, inplace=True)

sensor_stats = desc_t[desc_t['feature'].isin(sensor_cols)]

plt.figure(figsize=(14, 5))
sns.barplot(x='feature', y='mean', data=sensor_stats)
plt.xticks(rotation=90)
plt.title("Giá trị trung bình (mean) của các đặc trưng cảm biến sau chuẩn hoá")
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
sns.barplot(x='feature', y='std', data=sensor_stats)
plt.xticks(rotation=90)
plt.title("Độ lệch chuẩn (std) của các đặc trưng cảm biến sau chuẩn hoá")
plt.axhline(1, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()
