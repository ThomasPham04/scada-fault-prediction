
import argparse
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

NUMERIC_PREFIXES = ["sensor_", "wind_speed_", "power_", "reactive_power_"]

def read_feature_description(fd_path: Path) -> Tuple[Set[str], Set[str]]:
    if not fd_path.exists():
        warnings.warn(f"[WARN] feature_description.csv not found at {fd_path}; assuming no special types.")
        return set(), set()
    df = pd.read_csv(fd_path, sep=";")
    # df.columns = [c.replace("statistics_type", "statistic_type") for c in df.columns]

    def pick_set(flag_col: str) -> Set[str]:
        if flag_col not in df.columns:
            return set()
        mask = df[flag_col].astype(str).str.lower().isin(["true", "1", "yes"])
        if "sensor_name" in df.columns:
            return set(df.loc[mask, "sensor_name"].astype(str).str.lower())
        if {"sensor", "id"}.issubset(set(df.columns)):
            return set((df.loc[mask, "sensor"].astype(str).str.lower() + "_" +
                        df.loc[mask, "id"].astype(str).str.lower()))
        return set()

    return pick_set("is_angle"), pick_set("is_counter")

def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if any(str(c).startswith(p) for p in NUMERIC_PREFIXES)]

def map_angle_and_counter_columns(feature_cols: List[str],
                                  angle_basenames: Set[str],
                                  counter_basenames: Set[str]) -> Tuple[List[str], List[str]]:
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
    for col in angle_cols:
        rad = np.deg2rad(pd.to_numeric(df[col], errors="coerce").astype("float32", copy=False))
        df[col + "_sin"] = np.sin(rad, dtype="float32")
        df[col + "_cos"] = np.cos(rad, dtype="float32")
    df.drop(columns=angle_cols, inplace=True)

def counters_to_rates(df: pd.DataFrame, counter_cols: List[str]) -> None:
    for col in counter_cols:
        s = pd.to_numeric(df[col], errors="coerce").astype("float32", copy=False)
        diff = s.diff().clip(lower=0).fillna(0.0).astype("float32", copy=False)
        df[col + "_rate"] = diff
    df.drop(columns=counter_cols, inplace=True)

def standardize_inplace(df: pd.DataFrame, feature_cols: List[str],
                        train_mask: Optional[np.ndarray]) -> Dict[str, Dict[str, float]]:
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
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        # Verify datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["time_stamp"]):
            raise TypeError(f"[ERR] time_stamp column is not datetime type after conversion. Type: {df['time_stamp'].dtype}")
        return
    # for cand in ["time", "timestamp", "date", "datetime"]:
    #     if cand in df.columns:
    #         df.rename(columns={cand: "time_stamp"}, inplace=True)
    #         df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    #         return
    raise ValueError("[ERR] No time column found (time_stamp/time/timestamp/date/datetime).")

def discover_train_mask(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "train_test" in df.columns:
        return df["train_test"].astype(str).eq("train").to_numpy()
    return None

def build_windows_vectorized(df: pd.DataFrame,
                             feature_cols: List[str],
                             window: int,
                             stride: int,
                             anomaly_starts: np.ndarray,
                             horizon_minutes: int,
                             selected_features: List[str] = None):
    if selected_features is not None:
        use_cols = [c for c in feature_cols if c in selected_features]
    else:
        use_cols = feature_cols
    
    T = len(df)
    if T < window:
        return np.zeros((0, window, len(use_cols)), np.float32), np.zeros((0,), np.int8), pd.DataFrame([])

    if not pd.api.types.is_datetime64_any_dtype(df["time_stamp"]):
        raise TypeError(f"[ERR] time_stamp column is not datetime type. Actual type: {df['time_stamp'].dtype}")
    
    times = pd.to_datetime(df["time_stamp"]).to_numpy('datetime64[ns]')
    A = df[use_cols].to_numpy(dtype="float32", copy=False)

    valid_rows = ~np.isnan(A).any(axis=1)  # True nếu row không có NaN
    valid_win = sliding_window_view(valid_rows, window_shape=window, axis=0).all(axis=1)  

    X_view = sliding_window_view(A, window_shape=window, axis=0)  

    Xv = X_view[::stride]  
    valid_v = valid_win[::stride]
    end_times = times[window-1::stride]  

    # Chỉ giữ lại các windows hợp lệ (không có NaN)
    Xv = Xv[valid_v]
    end_times = end_times[valid_v]

    if anomaly_starts.size == 0:
        y = np.zeros((len(end_times),), dtype=np.int8)
    else:
        anomaly_starts = np.sort(anomaly_starts)
        idx = np.searchsorted(anomaly_starts, end_times, side="right")
        in_bounds = idx < anomaly_starts.size 
        next_anom = np.full(end_times.shape, np.datetime64('NaT'), dtype='datetime64[ns]')
        next_anom[in_bounds] = anomaly_starts[idx[in_bounds]]

        # Label = 1 nếu có anomaly trong horizon, ngược lại = 0
        y = (in_bounds & ((next_anom - end_times) <= np.timedelta64(horizon_minutes, 'm'))).astype(np.int8)

    # Tạo metadata DataFrame
    meta_df = pd.DataFrame({
        "end_time": end_times.astype('datetime64[ns]').astype('datetime64[ms]').astype(str),
        "label": y.astype(int)
    })  
    X = np.ascontiguousarray(Xv)
    return X, y, meta_df

def upsample_anomalies(X: np.ndarray, y: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Upsample anomaly windows by the given factor."""
    if factor <= 1:
        return X, y
    anomaly_indices = np.where(y == 1)[0]
    if len(anomaly_indices) == 0:
        return X, y
    X_anom = X[anomaly_indices]
    y_anom = y[anomaly_indices]
    X_upsampled = np.repeat(X_anom, factor, axis=0)
    y_upsampled = np.repeat(y_anom, factor, axis=0)
    X_new = np.concatenate([X, X_upsampled], axis=0)
    y_new = np.concatenate([y, y_upsampled], axis=0)
    return X_new, y_new

def print_label_distribution(y, name="y"):
    labels, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"Label distribution for {name}:")
    for lbl, cnt in zip(labels, counts):
        print(f"  Label {lbl}: {cnt} ({cnt/total*100:.2f}%)")
    print(f"  Total: {total}\n")


def compute_pearson_correlation_features(X_all: np.ndarray, feature_cols: List[str], 
                                         output_path: str = "pearson_correlation_features.csv") -> pd.DataFrame:
    print("[INFO] Computing Pearson correlation between features...")
    
    # Concatenate if list
    if isinstance(X_all, list):
        X_all = np.concatenate(X_all, axis=0)
    
    # Aggregate across window dimension: (N, F, W) -> (N, F)
    feat_values = X_all.mean(axis=2)  # Mean across window
    
    # Create DataFrame and compute correlation (simple!)
    df_features = pd.DataFrame(feat_values, columns=feature_cols)
    corr_matrix = df_features[feature_cols].corr(method='pearson')
    
    # Save correlation matrix
    corr_matrix.to_csv(output_path)
    print(f"[INFO] Saved Pearson correlation matrix to {output_path}")
    
    return corr_matrix


def remove_redundant_features(corr_matrix: pd.DataFrame, target_correlations: Dict[str, float],
                               threshold: float = 0.85) -> List[str]:

    print(f"[INFO] Removing redundant features (threshold: |corr| >= {threshold})...")
    
    feature_cols = corr_matrix.index.tolist()
    to_remove = set()
    redundant_pairs = []
    
    # Find highly correlated pairs
    for i, feat1 in enumerate(feature_cols):
        if feat1 in to_remove:
            continue
        
        for feat2 in feature_cols[i+1:]:
            if feat2 in to_remove:
                continue
            
            corr_value = corr_matrix.loc[feat1, feat2]
            if abs(corr_value) >= threshold:
                # Both features are highly correlated - remove the one with lower target correlation
                target_corr1 = abs(target_correlations.get(feat1, 0.0))
                target_corr2 = abs(target_correlations.get(feat2, 0.0))
                
                redundant_pairs.append({
                    "feature1": feat1,
                    "feature2": feat2,
                    "pearson_corr": corr_value,
                    "target_corr1": target_corr1,
                    "target_corr2": target_corr2,
                    "removed": feat1 if target_corr1 < target_corr2 else feat2
                })
                
                # Remove the feature with lower correlation to target
                if target_corr1 < target_corr2:
                    to_remove.add(feat1)
                else:
                    to_remove.add(feat2)
    
    # Return features without redundant ones
    selected_features = [f for f in feature_cols if f not in to_remove]
    
    print(f"[INFO] Found {len(redundant_pairs)} redundant pairs")
    print(f"[INFO] Removed {len(to_remove)} redundant features")
    print(f"[INFO] Remaining: {len(selected_features)} features (from {len(feature_cols)})")
    
    # Save redundant pairs report
    if redundant_pairs:
        df_redundant = pd.DataFrame(redundant_pairs)
        df_redundant = df_redundant.sort_values("pearson_corr", key=lambda x: x.abs(), ascending=False)
        df_redundant.to_csv("redundant_features.csv", index=False)
        print(f"[INFO] Saved redundant pairs to redundant_features.csv")
    
    return selected_features


def compute_point_biserial(X_all: np.ndarray, y_all: np.ndarray, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute point-biserial correlation between features and target variable.
    
    Note: This function expects X_all and y_all to contain smart-sampled data:
    - MUST include: ALL windows labeled 1 (fault/anomaly)
    - MUST include: Windows right before anomaly (early warning period)
    - OK to include: Some normal windows before/after the event
    - SHOULD NOT include: Months of unrelated normal data far from faults
    
    Returns:
        df_corr_global: DataFrame with features, correlations, and p-values (sorted by |corr|)
        target_correlations: Dict mapping feature name to its correlation with target
        X_corr: Sampled feature data used for correlation computation (fault-focused sampling)
        y_corr: Sampled target data used for correlation computation
    """
    print("[PASS 1] Computing point-biserial correlation (feature-to-target)...")
    
    # Concatenate all data
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    n_neg = np.sum(y_all == 0)
    n_pos = np.sum(y_all == 1)
    
    if n_pos == 0:
        print("[WARN] No positive samples found, skipping correlation computation.")
        # Return empty results
        df_empty = pd.DataFrame([], columns=["feature", "corr", "pval"])
        return df_empty, {}, X_all[:0], y_all[:0]
    
    n_samples_orig = len(y_all)
    n_features = X_all.shape[1]
    window_size = X_all.shape[2]
    
    print(f"[INFO] Class imbalance: {n_neg} negatives, {n_pos} positives (ratio: {n_neg/n_pos:.2f}:1)")
    print(f"[INFO] Total samples: {n_samples_orig}, Features: {n_features}, Window: {window_size}")
    
    from scipy.stats import pointbiserialr
    
    max_samples_for_corr = 50000
    
    if n_samples_orig <= max_samples_for_corr:
        X_corr = X_all
        y_corr = y_all
        print(f"[INFO] Using all {n_samples_orig} samples for correlation")
    else:
        np.random.seed(42)
        pos_indices = np.where(y_all == 1)[0]
        neg_indices = np.where(y_all == 0)[0]
        
        target_pos = min(n_pos, max_samples_for_corr // 2)
        target_neg = min(n_neg, max_samples_for_corr // 2)
        
        if len(pos_indices) <= target_pos:
            sampled_pos = pos_indices
        else:
            sampled_pos = np.random.choice(pos_indices, size=target_pos, replace=False)
        
        if len(neg_indices) <= target_neg:
            sampled_neg = neg_indices
        else:
            sampled_neg = np.random.choice(neg_indices, size=target_neg, replace=False)
        
        sampled_indices = np.concatenate([sampled_pos, sampled_neg])
        X_corr = X_all[sampled_indices]
        y_corr = y_all[sampled_indices]
        
        print(f"[INFO] Using stratified sampling: {len(sampled_pos)} positives, {len(sampled_neg)} negatives (total: {len(sampled_indices)})")
        print_label_distribution(y_corr, name="y_corr (sampled)")
    
    # Compute point-biserial correlations
    corr_list = []
    target_correlations = {}  # Store for redundancy removal
    
    print(f"[INFO] Computing point-biserial correlations for {n_features} features...")
    for i in range(n_features):
        feat_values = X_corr[:, i, :].mean(axis=1)
        corr, pval = pointbiserialr(feat_values, y_corr)
        if corr is None or np.isnan(corr):
            corr = 0.0
        
        corr_list.append((feature_cols[i], corr, pval))
        target_correlations[feature_cols[i]] = corr
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_features} features...")

    df_corr_global = pd.DataFrame(corr_list, columns=["feature", "corr", "pval"])
    df_corr_global["corr"] = df_corr_global["corr"].fillna(0)
    df_corr_global = df_corr_global.sort_values("corr", key=lambda x: x.abs(), ascending=False)
    
    df_corr_global.to_csv("corr_global.csv", index=False)
    print(f"[INFO] Saved point-biserial correlations to corr_global.csv")
    
    return df_corr_global, target_correlations, X_corr, y_corr


def select_top_features(df_corr_global: pd.DataFrame, target_correlations: Dict[str, float],
                       X_corr: np.ndarray, feature_cols: List[str], K_top: int,
                       remove_redundant: bool = False, redundancy_threshold: float = 0.85) -> List[str]:
    
    top_k_features = df_corr_global["feature"].head(K_top).tolist()
    print(f"\n{'='*60}")
    print(f"Point-biserial correlation Select top {K_top} most relevant features")
    print(f"{'='*60}")
    print(f"Selected {len(top_k_features)} features from point-biserial correlation")
    
    if not remove_redundant:
        
        return top_k_features
    
    
    print(f"\n{'='*60}")
    print(f"Pearson correlation → Remove redundant features from top {len(top_k_features)}")
    print(f"{'='*60}")
    print(f"[Computing Pearson correlation matrix for top {len(top_k_features)} features...")
    
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    top_k_indices = [feature_to_idx[f] for f in top_k_features if f in feature_to_idx]
    
    if len(top_k_indices) != len(top_k_features):
        missing = set(top_k_features) - set(feature_cols)
        print(f"[WARN] Some features not found in feature_cols: {missing}")
    
    X_top_k = X_corr[:, top_k_indices, :]  
    
    
    pearson_corr_matrix = compute_pearson_correlation_features(
        X_all=[X_top_k], 
        feature_cols=top_k_features,
        output_path="pearson_correlation_features_topk.csv"
    )
    
    # Create target_correlations dict for top K features only
    top_k_target_correlations = {f: target_correlations.get(f, 0.0) for f in top_k_features}
    
    # Remove redundant features from top K
    print(f"Removing redundant features (threshold: |corr| >= {redundancy_threshold})...")
    features_after_redundancy = remove_redundant_features(
        corr_matrix=pearson_corr_matrix,
        target_correlations=top_k_target_correlations,
        threshold=redundancy_threshold
    )
    
    removed_count = len(top_k_features) - len(features_after_redundancy)
    print(f" Redundancy removal complete: {len(features_after_redundancy)} features remaining (removed {removed_count} redundant features)")
    print(f"{'='*60}\n")
    
    return features_after_redundancy


def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm_dir", type=str, required=True)  # Thư mục chứa dữ liệu input
    ap.add_argument("--out_dir", type=str, required=True)   # Thư mục output
    ap.add_argument("--window", type=int, default=144)      # Kích thước window (số timesteps)
    ap.add_argument("--horizon", type=int, default=36)     # Horizon (đo bằng 10-minute steps)
    ap.add_argument("--stride", type=int, default=3)       # Bước nhảy giữa các windows
    ap.add_argument("--use_status", action="store_true")   # Có dùng status_type làm feature không
    ap.add_argument("--no_save_scaler", action="store_true")  # Không lưu scaler stats
    ap.add_argument("--add_diff", action="store_true",
                    help="Append first-order differences for numeric features.")
    ap.add_argument("--rolling_window", type=int, default=0,
                    help="If >0, append rolling mean/std features using this many rows.")
    ap.add_argument("--compute_corr_only", action="store_true",
                    help="Run pass 1: Compute correlation only, save selected_features.json")
    ap.add_argument("--top_features", type=int, default=50,
                    help="Number of top features to select based on correlation (default: 50)")
    ap.add_argument("--compute_pearson_corr", action="store_true",
                    help="Compute Pearson correlation between features (feature-to-feature), saves to pearson_correlation_features.csv")
    ap.add_argument("--remove_redundant", action="store_true",
                    help="Remove redundant features based on Pearson correlation before selecting top K features")
    ap.add_argument("--redundancy_threshold", type=float, default=0.85,
                    help="Pearson correlation threshold for redundancy removal (default: 0.85). Features with |corr| >= threshold are considered redundant")
    ap.add_argument("--use_selected_features", type=str, default=None,
                    help="Run pass 2: Only use selected features from JSON file")
    ap.add_argument("--compress", action="store_true",
                    help="Use compressed .npz format to save arrays (saves disk space)")
    ap.add_argument("--use_float16", action="store_true",
                    help="Use float16 instead of float32 (saves 50%% disk space, slight precision loss)")
    args = ap.parse_args()

    # Setup paths
    farm_dir = Path(args.farm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    selected_features = None
    if args.use_selected_features is not None:
        tentative_path = Path(args.use_selected_features)
        if tentative_path.is_file():
            json_path = tentative_path
        else:
            json_path = out_dir / args.use_selected_features
        with open(json_path, "r", encoding="utf-8") as fjs:
            selected_features = json.load(fjs)
        print(f"[INFO] Using {len(selected_features)} selected features from {json_path}")

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
    meta_rows_collected = []  # Collect all metadata to write at the end (overwrite, not append)
    scaler_all: Dict[str, Dict[str, Dict[str, float]]] = {}  # Lưu stats của scaler cho mỗi event
    feature_names_final: Optional[List[str]] = None  # Tên các features cuối cùng
    
    # For memory-efficient correlation: use incremental sampling instead of storing all arrays
    max_samples_in_memory = 50000  # Maximum samples to keep in memory for correlation
    sampled_X_list = []
    sampled_y_list = []
    total_samples_collected = 0
    
    # For Pearson correlation: also collect all feature data
    sampled_X_list_pearson = [] if args.compute_pearson_corr else None

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
        usecols = guess_usecols(csv_path)
        try:
            df = pd.read_csv(csv_path, sep=";", usecols=usecols)
        except Exception:
            df = pd.read_csv(csv_path, sep=";")
        normalize_time_column(df)  # Đảm bảo có cột "time_stamp" dạng datetime
        
        # Verify time_stamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["time_stamp"]):
            raise TypeError(f"[ERR] Event {event_id}: time_stamp is not datetime type. Actual type: {df['time_stamp'].dtype}")
        print(f"[INFO] Event {event_id}: time_stamp column verified as datetime type (dtype: {df['time_stamp'].dtype})")
        
        df.sort_values("time_stamp", inplace=True, kind="mergesort")
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

        # Feature engineering: first-order differences và rolling stats
        base_features = list(feature_cols)

        if args.add_diff:
            diff_df = df[base_features].diff().fillna(0.0)
            diff_cols = []
            for c in base_features:
                new_col = f"{c}_diff"
                df[new_col] = diff_df[c].astype("float32", copy=False)
                diff_cols.append(new_col)
            feature_cols += diff_cols

        if args.rolling_window and args.rolling_window > 1:
            roll = int(args.rolling_window)
            roll_mean = df[base_features].rolling(window=roll, min_periods=1).mean()
            roll_std = df[base_features].rolling(window=roll, min_periods=1).std(ddof=0).fillna(0.0)
            roll_cols = []
            for c in base_features:
                mean_col = f"{c}_roll{roll}_mean"
                std_col = f"{c}_roll{roll}_std"
                df[mean_col] = roll_mean[c].astype("float32", copy=False)
                df[std_col] = roll_std[c].astype("float32", copy=False)
                roll_cols.extend([mean_col, std_col])
            feature_cols += roll_cols

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
            anomaly_starts=anomaly_starts, horizon_minutes=horizon_minutes,
            selected_features=selected_features
        )
        
        if X.shape[0] > 0 and args.compute_corr_only:
            # Smart sampling strategy for correlation computation
            n_samples_this_event = X.shape[0]
            remaining_slots = max_samples_in_memory - total_samples_collected
            
            if remaining_slots > 0:
                pos_indices = np.where(y == 1)[0]
                neg_indices = np.where(y == 0)[0]
                n_pos = len(pos_indices)
                n_neg = len(neg_indices)
                
                is_fault_event = (label_str in {"anomaly", "fault", "1", "true"})
                
                if is_fault_event and n_pos > 0:
                    # FAULT EVENT: Prioritize windows around the fault
                    # 1. MUST include: ALL positive windows (fault/anomaly windows) - highest priority!
                    # If we're low on memory, still prioritize positives even if it means taking fewer negatives
                    
                    if remaining_slots >= len(pos_indices):
                        # Can fit ALL positives: take them all
                        sampled_pos = pos_indices
                        remaining_after_pos = remaining_slots - len(sampled_pos)
                    else:
                        # Cannot fit ALL positives: take as many as we can (prioritize positives!)
                        n_pos_to_take = min(n_pos, remaining_slots)
                        np.random.seed(42 + int(event_id))
                        sampled_pos = np.random.choice(pos_indices, size=n_pos_to_take, replace=False)
                        remaining_after_pos = 0
                        print(f"[WARN] Event {event_id}: Only sampling {n_pos_to_take}/{n_pos} positives due to memory limit")
                    
                    # 2. Prioritize: Negative windows close to fault (pre-fault / early warning)
                    # Only if we have remaining slots after taking positives
                    sampled_neg = np.array([], dtype=int)
                    if remaining_after_pos > 0:
                        # Calculate time distance from fault for each window
                        end_times = pd.to_datetime(meta_df["end_time"]).values
                        event_start_dt64 = np.datetime64(event_start)  # Convert pandas Timestamp to numpy datetime64
                        time_diffs = np.abs((end_times - event_start_dt64).astype('timedelta64[h]').astype(float))  # hours
                        
                        # Pre-fault period: windows within 48 hours BEFORE fault (but labeled 0 = early warning)
                        pre_fault_hours = 48
                        pre_fault_mask = (y == 0) & (end_times < event_start_dt64) & (time_diffs <= pre_fault_hours)
                        pre_fault_indices = np.where(pre_fault_mask)[0]
                        
                        # Nearby normal windows: within 72 hours of fault (before or after)
                        nearby_hours = 72
                        nearby_normal_mask = (y == 0) & (time_diffs <= nearby_hours)
                        nearby_normal_indices = np.where(nearby_normal_mask)[0]
                        
                        # Priority order: pre-fault > nearby normal > other negatives
                        neg_to_sample = []
                        
                        # First, try to include pre-fault windows (early warning)
                        if len(pre_fault_indices) > 0:
                            n_pre_fault = min(len(pre_fault_indices), remaining_after_pos // 2)
                            if n_pre_fault > 0:
                                np.random.seed(42 + int(event_id))
                                sampled_pre_fault = np.random.choice(pre_fault_indices, size=n_pre_fault, replace=False)
                                neg_to_sample.extend(sampled_pre_fault)
                                remaining_after_pos -= n_pre_fault
                        
                        # Then, include nearby normal windows
                        if remaining_after_pos > 0:
                            remaining_nearby = [idx for idx in nearby_normal_indices if idx not in neg_to_sample]
                            if len(remaining_nearby) > 0:
                                n_nearby = min(len(remaining_nearby), remaining_after_pos)
                                np.random.seed(42 + int(event_id) + 1000)
                                sampled_nearby = np.random.choice(remaining_nearby, size=n_nearby, replace=False)
                                neg_to_sample.extend(sampled_nearby)
                                remaining_after_pos -= len(sampled_nearby)
                        
                        # If still have slots, sample other negatives from this event
                        if remaining_after_pos > 0:
                            other_neg_indices = [idx for idx in neg_indices if idx not in neg_to_sample]
                            if len(other_neg_indices) > 0:
                                n_other = min(len(other_neg_indices), remaining_after_pos)
                                np.random.seed(42 + int(event_id) + 2000)
                                sampled_other = np.random.choice(other_neg_indices, size=n_other, replace=False)
                                neg_to_sample.extend(sampled_other)
                        
                        sampled_neg = np.array(neg_to_sample) if neg_to_sample else np.array([], dtype=int)
                    
                    sampled_indices = np.concatenate([sampled_pos, sampled_neg])
                    
                elif n_pos > 0 and n_neg > 0:
                    # NORMAL EVENT with some positives (rare): Use stratified sampling
                    target_pos = min(n_pos, remaining_slots // 2)
                    target_neg = min(n_neg, remaining_slots - target_pos)
                    
                    np.random.seed(42 + int(event_id))
                    if len(pos_indices) > target_pos:
                        sampled_pos = np.random.choice(pos_indices, size=target_pos, replace=False)
                    else:
                        sampled_pos = pos_indices
                    
                    if len(neg_indices) > target_neg:
                        sampled_neg = np.random.choice(neg_indices, size=target_neg, replace=False)
                    else:
                        sampled_neg = neg_indices
                    
                    sampled_indices = np.concatenate([sampled_pos, sampled_neg])
                else:
                    # Only positives or only negatives: take what we can
                    all_indices = np.concatenate([pos_indices, neg_indices])
                    n_to_sample = min(len(all_indices), remaining_slots)
                    np.random.seed(42 + int(event_id))
                    sampled_indices = np.random.choice(all_indices, size=n_to_sample, replace=False)
                
                # Apply sampling
                if len(sampled_indices) > 0:
                    X_sampled = X[sampled_indices]
                    y_sampled = y[sampled_indices]
                    
                    sampled_X_list.append(X_sampled)
                    sampled_y_list.append(y_sampled)
                    total_samples_collected += len(sampled_indices)
                    
                    pos_sampled = np.sum(y_sampled == 1)
                    neg_sampled = np.sum(y_sampled == 0)
                    print(f"[INFO] Event {event_id}: Sampled {len(sampled_indices)}/{n_samples_this_event} windows "
                          f"(+{pos_sampled}, -{neg_sampled}, total: {total_samples_collected})")
                else:
                    print(f"[INFO] Event {event_id}: No windows selected")
            else:
                print(f"[INFO] Event {event_id}: Skipping (memory limit reached: {total_samples_collected} samples)")

        # Lưu kết quả cho event này (skip .npy files if only computing correlations to save memory/disk)
        if not args.compute_corr_only:
            X_path = out_dir / f"X_{event_id}.npy"  # Features windows
            y_path = out_dir / f"y_{event_id}.npy"  # Labels
            
            # Convert to float16 if requested (saves 50% disk space)
            X_to_save = X.astype("float16", copy=False) if args.use_float16 else X
            
            if args.compress:
                # Use compressed format
                np.savez_compressed(X_path.with_suffix('.npz'), X=X_to_save)
                np.savez_compressed(y_path.with_suffix('.npz'), y=y.astype("int8", copy=False))
            else:
                np.save(X_path, X_to_save)
                np.save(y_path, y.astype("int8", copy=False))
        
        meta_path = out_dir / f"meta_{event_id}.csv"  # Metadata
        meta_df.to_csv(meta_path, index=False)

        # Collect metadata to write at the end (overwrite, not append)
        meta_rows_collected.append(meta_df.assign(event_id=event_id))

        # Thêm vào manifest (skip if only computing correlations)
        if not args.compute_corr_only:
            ext = ".npz" if args.compress else ".npy"
            X_path = out_dir / f"X_{event_id}{ext}"
            y_path = out_dir / f"y_{event_id}{ext}"
            manifest_lines.append(f"{event_id},{X_path.name},{y_path.name},{meta_path.name}")

        # Lưu feature names (chỉ lần đầu, giả sử tất cả events có cùng features)
        if feature_names_final is None and X.shape[0] > 0:
            feature_names_final = list(feature_cols)

        print(f"[OK] Event {event_id}: {X.shape[0]} windows; +{int(y.sum())} positives.")

    # Lưu các file tổng hợp
    (out_dir / "manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")
    # manifest.txt: danh sách tất cả events và file paths
    
    # Write collected metadata (overwrite, not append)
    if meta_rows_collected:
        pd.concat(meta_rows_collected, ignore_index=True).to_csv(meta_global_path, index=False)
    # meta_windows.csv: tổng hợp metadata cho tất cả events
    
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as fjs:
        json.dump(feature_names_final or [], fjs, ensure_ascii=False, indent=2)
    # feature_names.json: tên các features cuối cùng (sau khi transform)
    
    if not args.no_save_scaler:
        with open(out_dir / "scaler_stats.json", "w", encoding="utf-8") as fjs:
            json.dump(scaler_all, fjs, ensure_ascii=False, indent=2)
        # scaler_stats.json: mean và std của mỗi feature cho mỗi event (để dùng khi inference)
    
    if args.compute_corr_only:
        # Step 1: Compute point-biserial correlation (feature-to-target)
        df_corr_global, target_correlations, X_corr, y_corr = compute_point_biserial(
            X_all=sampled_X_list, 
            y_all=sampled_y_list, 
            feature_cols=feature_names_final or []
        )
        
        # Step 2: Select top K features (with optional redundancy removal)
        top_features = select_top_features(
            df_corr_global=df_corr_global,
            target_correlations=target_correlations,
            X_corr=X_corr,
            feature_cols=feature_names_final or [],
            K_top=args.top_features,
            remove_redundant=args.remove_redundant,
            redundancy_threshold=args.redundancy_threshold
        )
        
        # Save selected features
        with open(out_dir / "selected_features.json", "w", encoding="utf-8") as fjs:
            json.dump(top_features, fjs, ensure_ascii=False, indent=2)
        print(f"[PASS 1 DONE] Top features saved to {out_dir / 'selected_features.json'}")
        
        # Compute Pearson correlation between features if requested (standalone analysis)
        if args.compute_pearson_corr:
            pearson_output = out_dir / "pearson_correlation_features.csv"
            compute_pearson_correlation_features(
                X_all=sampled_X_list,
                feature_cols=feature_names_final or [],
                output_path=str(pearson_output)
            )
        
        return

    print("\n=== DONE (Fast, Fixed) ===")
    print(f"Output dir: {out_dir}")
    print("Artifacts per event: X_<id>.npy, y_<id>.npy, meta_<id>.csv")
    print("Global artifacts: manifest.txt, meta_windows.csv, feature_names.json" +
          (", scaler_stats.json" if not args.no_save_scaler else ""))

if __name__ == "__main__":
    main()
