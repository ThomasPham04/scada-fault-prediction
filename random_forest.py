#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Model cho SCADA Early-Warning Prediction

Flow:
1. Load tất cả X_*.npy và y_*.npy từ thư mục preprocessed
2. Gộp tất cả windows từ các events lại
3. Flatten windows: (N, W, F) -> (N, W*F)
4. Train/Val/Test split (stratified)
5. Train Random Forest với class weights
6. Evaluate trên validation và test sets
7. Lưu model và metrics
"""

import numpy as np
from pathlib import Path
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle
from glob import glob

def compute_pr_auc(y_true, y_prob):
    """
    Tính Precision-Recall AUC.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        float: PR-AUC score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(np.trapz(precision, recall))

def load_all_windows(data_dir: Path):
    """
    Load tất cả windows từ các event files.
    
    Flow:
    1. Tìm tất cả X_*.npy và y_*.npy files
    2. Load và gộp lại thành X và y tổng hợp
    3. Kiểm tra shape consistency
    
    Returns:
        X: numpy array (N_total, W, F) - tất cả windows gộp lại
        y: numpy array (N_total,) - tất cả labels gộp lại
    """
    data_dir = Path(data_dir)
    
    # Tìm tất cả X files
    X_files = sorted(glob(str(data_dir / "X_*.npy")))
    y_files = sorted(glob(str(data_dir / "y_*.npy")))
    
    if len(X_files) == 0:
        # Thử load từ manifest.txt
        manifest_path = data_dir / "manifest.txt"
        if manifest_path.exists():
            print(f"[INFO] Đọc từ manifest: {manifest_path}")
            X_files = []
            y_files = []
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        event_id, x_file, y_file = parts[0], parts[1], parts[2]
                        X_files.append(data_dir / x_file)
                        y_files.append(data_dir / y_file)
        else:
            raise FileNotFoundError(
                f"[ERROR] Không tìm thấy X_*.npy files trong {data_dir}\n"
                f"Và cũng không có manifest.txt. Hãy kiểm tra lại thư mục input."
            )
    
    if len(X_files) != len(y_files):
        raise ValueError(
            f"[ERROR] Số lượng X files ({len(X_files)}) khác với y files ({len(y_files)})"
        )
    
    print(f"[INFO] Tìm thấy {len(X_files)} event files")
    
    # Load và gộp tất cả windows
    X_list = []
    y_list = []
    
    for i, (x_file, y_file) in enumerate(zip(X_files, y_files)):
        X_i = np.load(x_file)
        y_i = np.load(y_file)
        
        # Kiểm tra shape consistency
        if len(X_i) != len(y_i):
            print(f"[WARN] Event {i}: X.shape[0]={len(X_i)} khác y.shape[0]={len(y_i)}, bỏ qua")
            continue
        
        X_list.append(X_i)
        y_list.append(y_i)
        print(f"  [{i+1}/{len(X_files)}] {Path(x_file).name}: {X_i.shape[0]} windows, "
              f"+{int(y_i.sum())} positives")
    
    if len(X_list) == 0:
        raise ValueError("[ERROR] Không có event nào hợp lệ để load")
    
    # Gộp tất cả lại
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"\n[INFO] Tổng hợp:")
    print(f"  X shape: {X.shape} (N={X.shape[0]}, W={X.shape[1]}, F={X.shape[2]})")
    print(f"  y shape: {y.shape}")
    
    return X, y

def main():
    """
    Hàm chính: Train Random Forest model cho early-warning prediction.
    
    Flow:
    1. Parse arguments
    2. Load tất cả windows từ preprocessed data
    3. Flatten windows: (N, W, F) -> (N, W*F)
    4. Train/Val/Test split (stratified)
    5. Train Random Forest với class weights
    6. Evaluate trên validation và test
    7. Tìm best threshold cho F1 score
    8. Lưu model và metrics
    """
    ap = argparse.ArgumentParser(
        description="Train Random Forest cho SCADA Early-Warning Prediction"
    )
    ap.add_argument("--in_dir", type=str, required=True,
                    help="Thư mục chứa preprocessed data (X_*.npy, y_*.npy)")
    ap.add_argument("--n_estimators", type=int, default=500,
                    help="Số cây trong Random Forest (default: 500)")
    ap.add_argument("--max_depth", type=int, default=None,
                    help="Max depth của cây (None = không giới hạn)")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Tỷ lệ test set (default: 0.2)")
    ap.add_argument("--val_size", type=float, default=0.1,
                    help="Tỷ lệ validation set (default: 0.1)")
    ap.add_argument("--random_state", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--save_model", action="store_true",
                    help="Lưu trained model (rf_model.pkl)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"[ERROR] Thư mục không tồn tại: {in_dir}")

    print("=" * 60)
    print("RANDOM FOREST TRAINING - SCADA Early-Warning")
    print("=" * 60)
    
    # Load tất cả windows
    print("\n[1/6] Loading data...")
    X, y = load_all_windows(in_dir)
    
    # Kiểm tra class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\nClass distribution (overall): {class_dist}")
    
    # Kiểm tra nếu chỉ có 1 class
    if len(unique) == 1:
        print(f"\n[ERROR] y chỉ có 1 class duy nhất: {int(unique[0])}")
        print("→ Không thể train Random Forest cho bài toán phân loại 0/1.")
        print("Hãy kiểm tra lại bước preprocess (window, horizon, stride) để tạo được ít nhất vài label = 1.")
        return
    
    # Flatten windows: (N, W, F) -> (N, W*F)
    print("\n[2/6] Flattening windows...")
    N, W, F = X.shape
    X_flat = X.reshape(N, W * F)
    print(f"  X_flat shape: {X_flat.shape} (từ {X.shape})")
    print(f"  Mỗi sample có {W * F} features (W={W} timesteps × F={F} features)")

    # Train/Val/Test split (stratified để đảm bảo có positive trong mỗi split)
    print("\n[3/6] Splitting data (stratified)...")
    total_test_size = args.test_size + args.val_size
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_flat, y, 
        test_size=total_test_size, 
        random_state=args.random_state, 
        stratify=y
    )
    
    # Split tmp thành val và test
    val_ratio = args.val_size / total_test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, 
        test_size=(1 - val_ratio), 
        random_state=args.random_state, 
        stratify=y_tmp
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # Kiểm tra class distribution trong train set
    u_tr, c_tr = np.unique(y_train, return_counts=True)
    print(f"  Class distribution (train): {dict(zip(u_tr, c_tr))}")
    
    if len(u_tr) < 2:
        print("\n[ERROR] Tập train sau split vẫn chỉ có 1 class.")
        print("→ Thường do số positive quá ít (vd chỉ 1–2 positive trên toàn dataset).")
        print("Bạn cần tăng số positive (đổi horizon, gộp thêm farm, giảm window, v.v.).")
        return

    # Tính class weights (balanced)
    print("\n[4/6] Computing class weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Class weights: {class_weight}")

    # Train Random Forest
    print("\n[5/6] Training Random Forest...")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  max_depth: {args.max_depth}")
    print(f"  class_weight: {class_weight}")
    
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=args.random_state,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    print("  ✓ Training completed!")

    # Evaluate trên validation set
    print("\n[6/6] Evaluating model...")
    
    # Validation set
    y_val_prob = rf.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    
    val_P = precision_score(y_val, y_val_pred, zero_division=0)
    val_R = recall_score(y_val, y_val_pred, zero_division=0)
    val_F1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_pr_auc = compute_pr_auc(y_val, y_val_prob)
    
    print(f"\n--- VALIDATION SET ---")
    print(f"Precision: {val_P:.4f}")
    print(f"Recall:    {val_R:.4f}")
    print(f"F1-score:   {val_F1:.4f}")
    print(f"PR-AUC:     {val_pr_auc:.4f}")
    
    # Test set
    y_test_prob = rf.predict_proba(X_test)[:, 1]
    
    # Default threshold @0.5
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    test_P = precision_score(y_test, y_test_pred, zero_division=0)
    test_R = recall_score(y_test, y_test_pred, zero_division=0)
    test_F1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_pr_auc = compute_pr_auc(y_test, y_test_prob)
    
    # Tìm best threshold cho F1 score trên validation set
    print("\n--- Finding best threshold (on validation set) ---")
    thresholds = np.linspace(0, 1, 200)
    f1s = []
    for t in thresholds:
        pred_t = (y_val_prob >= t).astype(int)
        f1s.append(f1_score(y_val, pred_t, zero_division=0))
    
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])
    best_f1_val = float(f1s[best_idx])
    
    print(f"Best threshold: {best_t:.4f} (F1={best_f1_val:.4f} on validation)")
    
    # Evaluate test set với best threshold
    y_test_pred_best = (y_test_prob >= best_t).astype(int)
    best_P = precision_score(y_test, y_test_pred_best, zero_division=0)
    best_R = recall_score(y_test, y_test_pred_best, zero_division=0)
    best_F1 = f1_score(y_test, y_test_pred_best, zero_division=0)
    
    print(f"\n--- TEST SET ---")
    print(f"Default @0.5:   P={test_P:.4f} R={test_R:.4f} F1={test_F1:.4f} PR-AUC={test_pr_auc:.4f}")
    print(f"Best @{best_t:.4f}: P={best_P:.4f} R={best_R:.4f} F1={best_F1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred_best)
    print(f"\nConfusion Matrix (best threshold):")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"        Positive   {cm[1,0]:6d}   {cm[1,1]:6d}")

    # Lưu kết quả
    results = {
        "model_config": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "class_weight": class_weight,
            "random_state": args.random_state
        },
        "data_info": {
            "total_samples": int(N),
            "window_size": int(W),
            "n_features": int(F),
            "flattened_features": int(W * F),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0])
        },
        "validation": {
            "precision": float(val_P),
            "recall": float(val_R),
            "f1": float(val_F1),
            "pr_auc": float(val_pr_auc)
        },
        "test_default": {
            "threshold": 0.5,
            "precision": float(test_P),
            "recall": float(test_R),
            "f1": float(test_F1),
            "pr_auc": float(test_pr_auc)
        },
        "test_best_f1": {
            "threshold": best_t,
            "precision": float(best_P),
            "recall": float(best_R),
            "f1": float(best_F1)
        }
    }
    
    metrics_path = in_dir / "rf_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved metrics: {metrics_path}")
    
    # Lưu model nếu có flag
    if args.save_model:
        model_path = in_dir / "rf_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(rf, f)
        print(f"✓ Saved model: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
