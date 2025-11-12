
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

NUMERIC_PREFIXES = ["sensor_", "wind_speed_", "power_", "reactive_power_"]

def read_feature_description(fd_path: Path) -> Tuple[Set[str], Set[str]]:
    if not fd_path.exists():
        warnings.warn(f"[WARN] feature_description.csv not found at {fd_path}; assuming no special types.")
        return set(), set()
    df = pd.read_csv(fd_path, sep=";")
    df.columns = [c.replace("statistics_type", "statistic_type") for c in df.columns]

    def pick_set(flag_col: str) -> Set[str]:
        if flag_col not in df.columns:
            return set()
        mask = (
            df[flag_col]
            .astype(str).str.str
            .isin(["true", "1", "yes"])
        )
        if "sensor_name" in df.columns:
            return set(df.loc[mask, "sensor_name"].astype(str).str.str)
        if {"sensor", "id"}.issubset(set(df.columns)):
            return set((df.loc[mask, "sensor"].astype(str).str.str + "_" +
                        df.loc[mask, "id"].astype(str).str.str))
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
    df.columns = [c for c in df.columns]
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        return
    for cand in ["time", "timestamp", "date", "datetime"]:
        if cand in df.columns:
            df.rename(columns={cand: "time_stamp"}, inplace=True)
            df["time_stamp"] = pd.to_datetime(df["time_stamp"])
            return
    raise ValueError("[ERR] No time column found (time_stamp/time/timestamp/date/datetime).")

def discover_train_mask(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "train_test" in df.columns:
        return df["train_test"].astype(str).str.eq("train").to_numpy()
    return None

def build_windows_vectorized(df: pd.DataFrame,
                             feature_cols: List[str],
                             window: int,
                             stride: int,
                             anomaly_starts: np.ndarray,
                             horizon_minutes: int):
    T = len(df)
    if T < window:
        return np.zeros((0, window, len(feature_cols)), np.float32), np.zeros((0,), np.int8), pd.DataFrame([])

    times = pd.to_datetime(df["time_stamp"]).to_numpy('datetime64[ns]')
    A = df[feature_cols].to_numpy(dtype="float32", copy=False)

    valid_rows = ~np.isnan(A).any(axis=1)
    valid_win = sliding_window_view(valid_rows, window_shape=window, axis=0).all(axis=1)
    X_view = sliding_window_view(A, window_shape=window, axis=0)  # (T-W+1, W, F)

    # stride sampling
    Xv = X_view[::stride]
    valid_v = valid_win[::stride]
    end_times = times[window-1::stride]

    # keep only valid windows (no NaNs)
    Xv = Xv[valid_v]
    end_times = end_times[valid_v]

    # ----- FIXED LABELING (avoid out-of-bounds) -----
    if anomaly_starts.size == 0:
        y = np.zeros((len(end_times),), dtype=np.int8)
    else:
        anomaly_starts = np.sort(anomaly_starts)  # ensure sorted
        idx = np.searchsorted(anomaly_starts, end_times, side="right")
        in_bounds = idx < anomaly_starts.size

        next_anom = np.full(end_times.shape, np.datetime64('NaT'), dtype='datetime64[ns]')
        # assign only where valid to avoid OOB
        next_anom[in_bounds] = anomaly_starts[idx[in_bounds]]

        # horizon in minutes
        y = (in_bounds & ((next_anom - end_times) <= np.timedelta64(horizon_minutes, 'm'))).astype(np.int8)

    meta_df = pd.DataFrame({
        "end_time": end_times.astype('datetime64[ns]').astype('datetime64[ms]').astype(str),
        "label": y.astype(int)
    })

    X = np.ascontiguousarray(Xv)
    return X, y, meta_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--window", type=int, default=144)
    ap.add_argument("--horizon", type=int, default=36)  # horizon measured in 10-minute steps in raw problem
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--use_status", action="store_true")
    ap.add_argument("--no_save_scaler", action="store_true")
    args = ap.parse_args()

    farm_dir = Path(args.farm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_info_path = farm_dir / "event_info.csv"
    ds_dir = farm_dir / "datasets"
    feat_desc_path = farm_dir / "feature_description.csv"

    if not ds_dir.exists():
        raise FileNotFoundError(f"[ERR] Datasets folder not found: {ds_dir}")

    events = pd.read_csv(event_info_path, sep=";")
    events["event_start"] = pd.to_datetime(events["event_start"])

    angle_bases, counter_bases = read_feature_description(feat_desc_path)

    manifest_lines = []
    meta_global_path = out_dir / "meta_windows.csv"
    meta_written = False
    scaler_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    feature_names_final: Optional[List[str]] = None

    for _, row in events.iterrows():
        event_id = str(row["event_id"])
        label_str = str(row["event_label"])
        event_start = pd.to_datetime(row["event_start"])

        csv_path = ds_dir / f"{event_id}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for event {event_id}: {csv_path}; skipping.")
            continue

        usecols = guess_usecols(csv_path)
        try:
            df = pd.read_csv(csv_path, sep=";", usecols=usecols)
        except Exception:
            df = pd.read_csv(csv_path, sep=";")

        normalize_time_column(df)
        df.sort_values("time_stamp", inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

        feature_cols = infer_feature_columns(df)

        if args.use_status and "status_type" in df.columns:
            dummies = pd.get_dummies(df["status_type"].astype("category"), prefix="status")
            df = pd.concat([df, dummies], axis=1)
            feature_cols += list(dummies.columns)

        angle_cols, counter_cols = map_angle_and_counter_columns(feature_cols, angle_bases, counter_bases)
        if angle_cols:
            angle_to_sin_cos(df, angle_cols)
            feature_cols = [c for c in feature_cols if c not in angle_cols] + \
                           [c + "_sin" for c in angle_cols] + [c + "_cos" for c in angle_cols]
        if counter_cols:
            counters_to_rates(df, counter_cols)
            feature_cols = [c for c in feature_cols if c not in counter_cols] + \
                           [c + "_rate" for c in counter_cols]

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32", copy=False)

        df.set_index("time_stamp", inplace=True)
        df[feature_cols] = df[feature_cols].interpolate(method="time", limit_direction="both")
        df.reset_index(inplace=True)

        train_mask = discover_train_mask(df)
        stats = standardize_inplace(df, feature_cols, train_mask)
        if not args.no_save_scaler:
            scaler_all[event_id] = stats

        # horizon is specified in 10-minute steps -> convert to minutes
        horizon_minutes = int(args.horizon) * 10

        anomaly_starts = np.array(
            [np.datetime64(event_start)] if label_str in {"anomaly", "fault", "1", "true"} else [],
            dtype='datetime64[ns]'
        )

        X, y, meta_df = build_windows_vectorized(
            df, feature_cols, window=args.window, stride=args.stride,
            anomaly_starts=anomaly_starts, horizon_minutes=horizon_minutes
        )

        X_path = out_dir / f"X_{event_id}.npy"
        y_path = out_dir / f"y_{event_id}.npy"
        meta_path = out_dir / f"meta_{event_id}.csv"

        np.save(X_path, X)
        np.save(y_path, y.astype("int8", copy=False))
        meta_df.to_csv(meta_path, index=False)

        meta_df.assign(event_id=event_id).to_csv(
            meta_global_path, mode="a", header=not meta_written, index=False
        )
        meta_written = True

        manifest_lines.append(f"{event_id},{X_path.name},{y_path.name},{meta_path.name}")

        if feature_names_final is None and X.shape[0] > 0:
            feature_names_final = list(feature_cols)

        print(f"[OK] Event {event_id}: {X.shape[0]} windows; +{int(y.sum())} positives.")

    (out_dir / "manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as fjs:
        json.dump(feature_names_final or [], fjs, ensure_ascii=False, indent=2)
    if not args.no_save_scaler:
        with open(out_dir / "scaler_stats.json", "w", encoding="utf-8") as fjs:
            json.dump(scaler_all, fjs, ensure_ascii=False, indent=2)

    print("\n=== DONE (Fast, Fixed) ===")
    print(f"Output dir: {out_dir}")
    print("Artifacts per event: X_<id>.npy, y_<id>.npy, meta_<id>.csv")
    print("Global artifacts: manifest.txt, meta_windows.csv, feature_names.json" +
          (", scaler_stats.json" if not args.no_save_scaler else ""))

if __name__ == "__main__":
    main()
