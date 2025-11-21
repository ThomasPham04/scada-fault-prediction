
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ====================================================================
# SCADA Early-Warning Preprocessing (Low-RAM + Faster Vectorization) - FIXED
# - Fixes IndexError when anomaly_starts has size 1 (or any) by avoiding
#   out-of-bounds advanced indexing inside np.where.
# ====================================================================

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Các prefix để nhận diện các cột feature (sensor data)
NUMERIC_PREFIXES = ["sensor_", "wind_speed_", "power_", "reactive_power_"]

def read_feature_description(fd_path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Đọc file feature_description.csv để xác định các cột đặc biệt:
    - Cột góc (angle): cần chuyển sang sin/cos
    - Cột counter: cần chuyển sang rate (tốc độ thay đổi)
    
    Flow:
    1. Đọc CSV với separator ";"
    2. Tìm các cột có flag "is_angle" = true -> danh sách angle columns
    3. Tìm các cột có flag "is_counter" = true -> danh sách counter columns
    4. Trả về 2 sets: angle_basenames và counter_basenames
    
    Returns:
        Tuple[Set[str], Set[str]]: (angle_basenames, counter_basenames)
    """
    if not fd_path.exists():
        warnings.warn(f"[WARN] feature_description.csv not found at {fd_path}; assuming no special types.")
        return set(), set()
    df = pd.read_csv(fd_path, sep=";")
    # df.columns = [c.replace("statistics_type", "statistic_type") for c in df.columns]

    def pick_set(flag_col: str) -> Set[str]:
        if flag_col not in df.columns:
            return set()
        mask = (
            df[flag_col]
            .astype(str)
            .isin(["true", "1", "yes"])
        )
        if "sensor_name" in df.columns:
            return set(df.loc[mask, "sensor_name"].astype(str))
        if {"sensor", "id"}.issubset(set(df.columns)):
            return set((df.loc[mask, "sensor"].astype(str) + "_" +
                        df.loc[mask, "id"].astype(str)))
        return set()

    return pick_set("is_angle"), pick_set("is_counter")

def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Tìm tất cả các cột feature trong DataFrame.
    
    Flow:
    - Duyệt qua tất cả columns của df
    - Giữ lại các cột có tên bắt đầu bằng một trong NUMERIC_PREFIXES
    - Ví dụ: "sensor_001", "wind_speed_avg", "power_total" sẽ được giữ lại
    
    Returns:
        List[str]: Danh sách tên các cột feature
    """
    return [c for c in df.columns if any(str(c).startswith(p) for p in NUMERIC_PREFIXES)]

def map_angle_and_counter_columns(feature_cols: List[str],
                                  angle_basenames: Set[str],
                                  counter_basenames: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Phân loại các feature columns thành angle columns và counter columns.
    
    Flow:
    1. Với mỗi feature column (ví dụ: "sensor_001_avg"):
       - Tách tên bằng "_" -> ["sensor", "001", "avg"]
       - Lấy 2 phần đầu làm base -> "sensor_001"
    2. Kiểm tra base có trong angle_basenames không -> thêm vào angle_cols
    3. Kiểm tra base có trong counter_basenames không -> thêm vào counter_cols
    
    Returns:
        Tuple[List[str], List[str]]: (angle_cols, counter_cols)
    """
    angle_cols, counter_cols = [], []
    for col in feature_cols:
        parts = str(col).split("_")
        base = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if base in angle_basenames:
            angle_cols.append(col)
        if base in counter_basenames:
            counter_cols.append(col)
    return angle_cols, counter_cols

def angle_to_sin_cos(df: pd.DataFrame, angle_cols: List[str]) -> None:
    """
    Chuyển đổi các cột góc (degrees) sang sin và cos để xử lý tính tuần hoàn của góc.
    
    Flow:
    1. Với mỗi angle column:
       - Chuyển giá trị từ degrees sang radians
       - Tạo 2 cột mới: col_sin và col_cos
       - Ví dụ: "wind_direction" -> "wind_direction_sin" và "wind_direction_cos"
    2. Xóa các cột góc gốc (vì đã có sin/cos thay thế)
    
    Lý do: Góc có tính tuần hoàn (0° = 360°), nên chuyển sang sin/cos 
    giúp model học tốt hơn.
    """
    for col in angle_cols:
        rad = np.deg2rad(pd.to_numeric(df[col], errors="coerce").astype("float32", copy=False))
        df[col + "_sin"] = np.sin(rad, dtype="float32")
        df[col + "_cos"] = np.cos(rad, dtype="float32")
    df.drop(columns=angle_cols, inplace=True)

def counters_to_rates(df: pd.DataFrame, counter_cols: List[str]) -> None:
    """
    Chuyển đổi các cột counter (bộ đếm tích lũy) sang rate (tốc độ thay đổi).
    
    Flow:
    1. Với mỗi counter column:
       - Tính diff() = giá trị hiện tại - giá trị trước đó
       - clip(lower=0): đảm bảo rate >= 0 (counter chỉ tăng, không giảm)
       - fillna(0.0): giá trị đầu tiên không có diff -> gán 0
       - Tạo cột mới: col_rate
    2. Xóa các cột counter gốc
    
    Lý do: Counter là giá trị tích lũy, rate (tốc độ) quan trọng hơn cho prediction.
    """
    for col in counter_cols:
        s = pd.to_numeric(df[col], errors="coerce").astype("float32", copy=False)
        diff = s.diff().clip(lower=0).fillna(0.0).astype("float32", copy=False)
        df[col + "_rate"] = diff
    df.drop(columns=counter_cols, inplace=True)

def standardize_inplace(df: pd.DataFrame, feature_cols: List[str],
                        train_mask: Optional[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Chuẩn hóa (standardize) các feature columns: (x - mean) / std
    
    Flow:
    1. Xác định reference data để tính mean/std:
       - Nếu có train_mask: chỉ dùng train data
       - Nếu không: dùng toàn bộ data
    2. Tính mean và std cho mỗi feature
       - std = 0 -> thay bằng 1 (tránh chia 0)
    3. Áp dụng công thức: (x - mean) / std cho tất cả rows
    4. Lưu lại stats (mean, std) để dùng sau này
    
    Returns:
        Dict: {feature_name: {"mean": float, "std": float}}
    """
    stats: Dict[str, Dict[str, float]] = {}
    ref = df[feature_cols] if train_mask is None or len(train_mask) != len(df) else df.loc[train_mask, feature_cols]
    means = ref.mean().astype("float64")
    stds = ref.std(ddof=0).replace(0, np.nan).astype("float64").fillna(1.0)
    fvals = df[feature_cols].astype("float32", copy=False)
    for c in feature_cols:
        f32 = (fvals[c].astype("float64") - means[c]) / stds[c]
        df[c] = f32.astype("float32", copy=False)
        stats[c] = {"mean": float(means[c]), "std": float(stds[c])}
    return stats

def guess_usecols(csv_path: Path) -> Optional[List[str]]:
    """
    Đoán các cột cần đọc từ CSV (chưa được sử dụng trong code hiện tại).
    
    Flow:
    1. Đọc header của CSV (nrows=0)
    2. Giữ lại các cột:
       - Cột thời gian: time_stamp, time, timestamp, date, datetime
       - Cột metadata: status_type, train_test
       - Cột feature: bắt đầu bằng NUMERIC_PREFIXES
    
    Returns:
        Optional[List[str]]: Danh sách cột cần đọc, hoặc None nếu lỗi
    """
    try:
        head = pd.read_csv(csv_path, sep=";", nrows=0)
    except Exception:
        return None
    cols = [c for c in head.columns]
    usecols = []
    for c in cols:
        cs = c
        if cs in {"time_stamp", "time", "timestamp", "date", "datetime", "status_type", "train_test"}:
            usecols.append(c)
        elif any(cs.startswith(p) for p in NUMERIC_PREFIXES):
            usecols.append(c)
    return usecols or None

def normalize_time_column(df: pd.DataFrame) -> None:
    """
    Chuẩn hóa cột thời gian: tìm và chuyển đổi sang datetime.
    
    Flow:
    1. Kiểm tra xem có cột "time_stamp" không -> chuyển sang datetime
    2. Nếu không, tìm các tên khác: "time", "timestamp", "date", "datetime"
    3. Đổi tên cột đó thành "time_stamp" và chuyển sang datetime
    4. Nếu không tìm thấy -> raise error
    
    Mục đích: Đảm bảo có cột "time_stamp" với kiểu datetime để xử lý sau.
    """
    # df.columns = [c for c in df.columns]
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        return
    else:
        print ("No time_stamp columns in dataset")
    # for cand in ["time", "timestamp", "date", "datetime"]:
    #     if cand in df.columns:
    #         df.rename(columns={cand: "time_stamp"}, inplace=True)
    #         df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    #         return
    # raise ValueError("[ERR] No time column found (time_stamp/time/timestamp/date/datetime).")

def discover_train_mask(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Tìm mask để xác định dữ liệu train (dùng cho standardization).
    
    Flow:
    - Nếu có cột "train_test":
      - Chuyển sang string và so sánh với "train"
      - Trả về boolean array: True = train, False = test
    - Nếu không có -> trả về None (dùng toàn bộ data)
    
    Returns:
        Optional[np.ndarray]: Boolean mask cho train data, hoặc None
    """
    if "train_test" in df.columns:
        return df["train_test"].astype(str).eq("train").to_numpy()
    return None

def build_windows_vectorized(df: pd.DataFrame,
                             feature_cols: List[str],
                             window: int,
                             stride: int,
                             anomaly_starts: np.ndarray,
                             horizon_minutes: int):
    """
    Tạo sliding windows từ time series data và gán nhãn (label) cho mỗi window.
    
    Flow chi tiết:
    
    1. KIỂM TRA ĐIỀU KIỆN:
       - Nếu số rows < window -> trả về empty arrays
    
    2. CHUẨN BỊ DỮ LIỆU:
       - times: mảng datetime từ cột time_stamp
       - A: mảng numpy của features (T rows x F features)
    
    3. TẠO SLIDING WINDOWS:
       - valid_rows: các rows không có NaN
       - valid_win: các windows mà TẤT CẢ rows đều valid (không có NaN)
       - X_view: sliding windows view (T-W+1, W, F) - dùng stride tricks để hiệu quả
    
    4. STRIDE SAMPLING (lấy mẫu theo bước nhảy):
       - Xv = X_view[::stride]: lấy mỗi stride-th window
       - end_times: thời gian kết thúc của mỗi window (times[window-1::stride])
    
    5. LỌC WINDOWS HỢP LỆ:
       - Chỉ giữ lại windows không có NaN
    
    6. GÁN NHÃN (LABELING):
       - Nếu không có anomaly_starts: tất cả labels = 0
       - Nếu có:
         a. Sắp xếp anomaly_starts
         b. Với mỗi window end_time, tìm anomaly gần nhất tiếp theo
         c. Nếu khoảng cách <= horizon_minutes -> label = 1 (positive)
         d. Ngược lại -> label = 0 (negative)
    
    7. TẠO METADATA:
       - meta_df chứa end_time và label cho mỗi window
    
    Returns:
        X: numpy array (N, W, F) - Feature windows
            - N: số lượng windows
            - W: số timesteps trong mỗi window (window size, mặc định 144)
            - F: số features (số cột feature sau khi transform)
            
            Mỗi window chứa:
            - W hàng liên tiếp từ time series gốc
            - Mỗi hàng có F giá trị feature (đã được chuẩn hóa)
            - Ví dụ với window=144, F=50:
              * Window 0: rows 0-143 của DataFrame, mỗi row có 50 features
              * Window 1: rows 3-146 (nếu stride=3), mỗi row có 50 features
              * ...
              
            Các features trong mỗi window bao gồm:
            - Sensor data: sensor_001, sensor_002, ...
            - Wind speed: wind_speed_avg, wind_speed_max, ...
            - Power: power_total, power_active, ...
            - Reactive power: reactive_power_...
            - Angle features (nếu có): đã chuyển thành _sin và _cos
            - Counter features (nếu có): đã chuyển thành _rate
            - Status dummies (nếu --use_status): status_xxx
            - Tất cả đã được chuẩn hóa (mean=0, std=1)
            
        y: numpy array (N,) - labels (0 hoặc 1)
            - 0: không có anomaly trong horizon
            - 1: có anomaly sẽ xảy ra trong horizon_minutes từ end_time của window
            
        meta_df: DataFrame với end_time và label
    """
    T = len(df)
    if T < window:
        return np.zeros((0, window, len(feature_cols)), np.float32), np.zeros((0,), np.int8), pd.DataFrame([])

    # Chuyển đổi time và features sang numpy arrays
    # A có shape (T, F) - T rows x F features
    # Ví dụ: nếu có 1000 rows và 50 features -> A.shape = (1000, 50)
    times = pd.to_datetime(df["time_stamp"]).to_numpy('datetime64[ns]')
    A = df[feature_cols].to_numpy(dtype="float32", copy=False)

    # Xác định các rows và windows hợp lệ (không có NaN)
    valid_rows = ~np.isnan(A).any(axis=1)  # True nếu row không có NaN
    valid_win = sliding_window_view(valid_rows, window_shape=window, axis=0).all(axis=1)  # True nếu window không có NaN
    
    # Tạo sliding windows view từ A
    # X_view có shape (T-W+1, W, F)
    # Ví dụ: T=1000, W=144, F=50 -> X_view.shape = (857, 144, 50)
    # Mỗi window là một "slice" liên tiếp của W rows từ A
    # Window 0: A[0:144, :]   -> 144 rows đầu tiên, tất cả features
    # Window 1: A[1:145, :]   -> rows 1-144, tất cả features
    # Window 2: A[2:146, :]   -> rows 2-145, tất cả features
    # ...
    X_view = sliding_window_view(A, window_shape=window, axis=0)  # (T-W+1, W, F) - view, không copy

    # Lấy mẫu theo stride (ví dụ stride=3: lấy window 0, 3, 6, 9...)
    # Sau stride sampling, số windows giảm đi
    # Ví dụ: 857 windows -> 857//3 = 285 windows (nếu stride=3)
    Xv = X_view[::stride]  # Shape: (N, W, F) với N = (T-W+1)//stride
    valid_v = valid_win[::stride]
    end_times = times[window-1::stride]  # Thời gian kết thúc của mỗi window
    # end_times[0] = thời gian của row (window-1) = thời gian kết thúc window đầu tiên

    # Chỉ giữ lại các windows hợp lệ (không có NaN)
    Xv = Xv[valid_v]
    end_times = end_times[valid_v]

    # ----- GÁN NHÃN (FIXED - tránh out-of-bounds) -----
    if anomaly_starts.size == 0:
        # Không có anomaly -> tất cả labels = 0
        y = np.zeros((len(end_times),), dtype=np.int8)
    else:
        anomaly_starts = np.sort(anomaly_starts)  # Sắp xếp để dùng searchsorted
        # Tìm vị trí chèn cho mỗi end_time trong anomaly_starts (tìm anomaly tiếp theo)
        idx = np.searchsorted(anomaly_starts, end_times, side="right")
        in_bounds = idx < anomaly_starts.size  # Kiểm tra không vượt quá mảng

        # Tạo mảng chứa anomaly tiếp theo cho mỗi window
        next_anom = np.full(end_times.shape, np.datetime64('NaT'), dtype='datetime64[ns]')
        # Chỉ gán giá trị ở những vị trí hợp lệ (tránh out-of-bounds)
        next_anom[in_bounds] = anomaly_starts[idx[in_bounds]]

        # Label = 1 nếu có anomaly trong horizon, ngược lại = 0
        y = (in_bounds & ((next_anom - end_times) <= np.timedelta64(horizon_minutes, 'm'))).astype(np.int8)

    # Tạo metadata DataFrame
    meta_df = pd.DataFrame({
        "end_time": end_times.astype('datetime64[ns]').astype('datetime64[ms]').astype(str),
        "label": y.astype(int)
    })

    # Chuyển sang contiguous array (cần thiết cho một số operations)
    # X cuối cùng có shape (N, W, F):
    # - N windows (sau khi stride và filter)
    # - Mỗi window có W timesteps (rows)
    # - Mỗi timestep có F features (đã chuẩn hóa)
    #
    # Ví dụ cụ thể:
    # X[0] -> Window đầu tiên, shape (144, 50)
    #   - 144 timesteps (rows từ DataFrame gốc)
    #   - Mỗi timestep có 50 features (sensor, wind_speed, power, ...)
    #   - Tất cả giá trị đã được chuẩn hóa (mean≈0, std≈1)
    #
    # X[1] -> Window thứ hai, shape (144, 50)
    #   - 144 timesteps tiếp theo (cách stride timesteps so với window 0)
    #   - Cùng 50 features
    X = np.ascontiguousarray(Xv)
    return X, y, meta_df

def main():
    """
    Hàm chính: Xử lý tất cả events và tạo windows cho early-warning prediction.
    
    FLOW TỔNG QUAN:
    1. Parse arguments và setup paths
    2. Đọc event_info.csv và feature_description.csv
    3. Với mỗi event:
       a. Đọc CSV data
       b. Preprocessing (normalize time, transform features)
       c. Tạo sliding windows
       d. Gán labels
       e. Lưu kết quả
    4. Tạo các file tổng hợp (manifest, feature_names, scaler_stats)
    """
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm_dir", type=str, required=True)  # Thư mục chứa dữ liệu input
    ap.add_argument("--out_dir", type=str, required=True)   # Thư mục output
    ap.add_argument("--window", type=int, default=144)      # Kích thước window (số timesteps)
    ap.add_argument("--horizon", type=int, default=36)     # Horizon (đo bằng 10-minute steps)
    ap.add_argument("--stride", type=int, default=3)       # Bước nhảy giữa các windows
    ap.add_argument("--use_status", action="store_true")   # Có dùng status_type làm feature không
    ap.add_argument("--no_save_scaler", action="store_true")  # Không lưu scaler stats
    args = ap.parse_args()

    # Setup paths
    farm_dir = Path(args.farm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_info_path = farm_dir / "event_info.csv"  # File chứa thông tin các events
    ds_dir = farm_dir / "datasets"                 # Thư mục chứa CSV files cho mỗi event
    feat_desc_path = farm_dir / "feature_description.csv"  # File mô tả các feature đặc biệt

    if not ds_dir.exists():
        raise FileNotFoundError(f"[ERR] Datasets folder not found: {ds_dir}")

    # Đọc danh sách events
    events = pd.read_csv(event_info_path, sep=";")
    events["event_start"] = pd.to_datetime(events["event_start"])

    # Đọc feature description để biết cột nào là angle/counter
    angle_bases, counter_bases = read_feature_description(feat_desc_path)
    # angle_bases: danh sách các base names của angle columns
    # counter_bases: danh sách các base names của counter columns

    # Khởi tạo các biến để lưu kết quả
    manifest_lines = []  # Danh sách các dòng cho manifest.txt
    meta_global_path = out_dir / "meta_windows.csv"  # File tổng hợp metadata
    meta_written = False  # Flag để viết header chỉ 1 lần
    scaler_all: Dict[str, Dict[str, Dict[str, float]]] = {}  # Lưu stats của scaler cho mỗi event
    feature_names_final: Optional[List[str]] = None  # Tên các features cuối cùng

    # Xử lý từng event
    for _, row in events.iterrows():
        # Lấy thông tin event
        event_id = str(row["event_id"])
        label_str = str(row["event_label"])  # "anomaly", "fault", "normal", etc.
        event_start = pd.to_datetime(row["event_start"])  # Thời điểm bắt đầu anomaly

        # Kiểm tra file CSV có tồn tại không
        csv_path = ds_dir / f"{event_id}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for event {event_id}: {csv_path}; skipping.")
            continue

        # Đọc CSV và chuẩn hóa
        df = pd.read_csv(csv_path, sep=';')
        normalize_time_column(df)  # Đảm bảo có cột "time_stamp" dạng datetime
        # df.sort_values("time_stamp", inplace=True, kind="mergesort")  # Sắp xếp theo thời gian
        df.reset_index(drop=True, inplace=True)

        # Tìm các cột feature (bắt đầu bằng NUMERIC_PREFIXES)
        feature_cols = infer_feature_columns(df)

        # Nếu có flag --use_status, thêm status_type làm one-hot encoding
        if args.use_status and "status_type" in df.columns:
            dummies = pd.get_dummies(df["status_type"].astype("category"), prefix="status")
            df = pd.concat([df, dummies], axis=1)
            feature_cols += list(dummies.columns)

        # Phân loại và transform các cột đặc biệt
        angle_cols, counter_cols = map_angle_and_counter_columns(feature_cols, angle_bases, counter_bases)
        if angle_cols:
            # Chuyển góc sang sin/cos
            angle_to_sin_cos(df, angle_cols)
            # Cập nhật danh sách feature_cols: xóa angle, thêm sin/cos
            feature_cols = [c for c in feature_cols if c not in angle_cols] + \
                           [c + "_sin" for c in angle_cols] + [c + "_cos" for c in angle_cols]
        if counter_cols:
            # Chuyển counter sang rate
            counters_to_rates(df, counter_cols)
            # Cập nhật danh sách feature_cols: xóa counter, thêm rate
            feature_cols = [c for c in feature_cols if c not in counter_cols] + \
                           [c + "_rate" for c in counter_cols]

        # Chuyển tất cả features sang float32
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32", copy=False)

        # Interpolation: điền các giá trị NaN bằng interpolation theo thời gian
        df.set_index("time_stamp", inplace=True)
        df[feature_cols] = df[feature_cols].interpolate(method="time", limit_direction="both")
        df.reset_index(inplace=True)

        # Standardization: chuẩn hóa features (mean=0, std=1)
        train_mask = discover_train_mask(df)  # Tìm train data (nếu có)
        stats = standardize_inplace(df, feature_cols, train_mask)  # Chuẩn hóa và lưu stats
        if not args.no_save_scaler:
            scaler_all[event_id] = stats  # Lưu stats để dùng sau

        # Chuyển đổi horizon từ 10-minute steps sang minutes
        horizon_minutes = int(args.horizon) * 10

        # Tạo mảng anomaly_starts: nếu event là anomaly/fault thì có 1 điểm, ngược lại rỗng
        anomaly_starts = np.array(
            [np.datetime64(event_start)] if label_str in {"anomaly", "fault", "1", "true"} else [],
            dtype='datetime64[ns]'
        )

        # Tạo sliding windows và gán labels
        X, y, meta_df = build_windows_vectorized(
            df, feature_cols, window=args.window, stride=args.stride,
            anomaly_starts=anomaly_starts, horizon_minutes=horizon_minutes
        )

        # Lưu kết quả cho event này
        X_path = out_dir / f"X_{event_id}.npy"  # Features windows
        y_path = out_dir / f"y_{event_id}.npy"  # Labels
        meta_path = out_dir / f"meta_{event_id}.csv"  # Metadata

        np.save(X_path, X)
        np.save(y_path, y.astype("int8", copy=False))
        meta_df.to_csv(meta_path, index=False)

        # Thêm vào file metadata tổng hợp (append mode)
        meta_df.assign(event_id=event_id).to_csv(
            meta_global_path, mode="a", header=not meta_written, index=False
        )
        meta_written = True  # Đã viết header rồi, lần sau không viết nữa

        # Thêm vào manifest
        manifest_lines.append(f"{event_id},{X_path.name},{y_path.name},{meta_path.name}")

        # Lưu feature names (chỉ lần đầu, giả sử tất cả events có cùng features)
        if feature_names_final is None and X.shape[0] > 0:
            feature_names_final = list(feature_cols)

        print(f"[OK] Event {event_id}: {X.shape[0]} windows; +{int(y.sum())} positives.")

    # Lưu các file tổng hợp
    (out_dir / "manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")
    # manifest.txt: danh sách tất cả events và file paths
    
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as fjs:
        json.dump(feature_names_final or [], fjs, ensure_ascii=False, indent=2)
    # feature_names.json: tên các features cuối cùng (sau khi transform)
    
    if not args.no_save_scaler:
        with open(out_dir / "scaler_stats.json", "w", encoding="utf-8") as fjs:
            json.dump(scaler_all, fjs, ensure_ascii=False, indent=2)
        # scaler_stats.json: mean và std của mỗi feature cho mỗi event (để dùng khi inference)

    print("\n=== DONE (Fast, Fixed) ===")
    print(f"Output dir: {out_dir}")
    print("Artifacts per event: X_<id>.npy, y_<id>.npy, meta_<id>.csv")
    print("Global artifacts: manifest.txt, meta_windows.csv, feature_names.json" +
          (", scaler_stats.json" if not args.no_save_scaler else ""))

if __name__ == "__main__":
    main()
