#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple XGBoost baseline for SCADA fault prediction.

This script flattens each (window, feature) tensor into a tabular vector and
trains an XGBoost binary classifier with event-aware splits aligned with the
sequence models. It relies on the manifest.txt file created by the data
preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def read_manifest(in_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    man_path = in_dir / "manifest.txt"
    if not man_path.exists():
        raise FileNotFoundError(f"manifest.txt not found at {man_path}")
    for line in man_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        event_id, x_name, y_name, meta_name = line.strip().split(",")
        rows.append(
            {
                "event_id": str(event_id),
                "x_path": str((in_dir / x_name).as_posix()),
                "y_path": str((in_dir / y_name).as_posix()),
                "meta_path": str((in_dir / meta_name).as_posix()),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Manifest is empty.")
    return df


def train_val_test_split_event(
    manifest_df: pd.DataFrame,
    seed: int,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    if len(ratios) != 3:
        raise ValueError("ratios must have exactly 3 elements.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Sum of ratios must be positive.")
    ratios = tuple(max(r, 0.0) / total for r in ratios)

    evt_stats = []
    for eid, grp in manifest_df.groupby("event_id"):
        pos, total_samples = 0, 0
        for _, r in grp.iterrows():
            y = np.load(r["y_path"], mmap_mode="r")
            pos += int((y == 1).sum())
            total_samples += int(y.shape[0])
        evt_stats.append({"event_id": str(eid), "positives": pos, "total": total_samples})

    pos_events = [e for e in evt_stats if e["positives"] > 0]
    neg_events = [e for e in evt_stats if e["positives"] == 0]

    def calc_targets(n: int) -> List[int]:
        raw = [r * n for r in ratios]
        targets = [int(x) for x in raw]
        remainder = n - sum(targets)
        fracs = [x - int(x) for x in raw]
        while remainder > 0:
            idx = max(range(len(targets)), key=lambda i: fracs[i])
            targets[idx] += 1
            fracs[idx] = 0.0
            remainder -= 1
        return targets

    def distribute(stats: List[Dict[str, int]], targets: Sequence[int], weight_key: str):
        buckets = [[] for _ in targets]
        loads = [0.0 for _ in targets]
        stats = stats.copy()
        rng.shuffle(stats)
        for st in stats:
            choices = [i for i, cap in enumerate(targets) if cap > 0 and len(buckets[i]) < cap]
            if not choices:
                choices = list(range(len(targets)))
            idx = min(choices, key=lambda i: (loads[i], len(buckets[i])))
            buckets[idx].append(st)
            loads[idx] += max(1, st[weight_key])
        return buckets

    pos_targets = calc_targets(len(pos_events))
    neg_targets = calc_targets(len(neg_events))
    pos_buckets = distribute(pos_events, pos_targets, "positives")
    neg_buckets = distribute(neg_events, neg_targets, "total")

    final = []
    for i in range(3):
        bucket = [s["event_id"] for s in pos_buckets[i]] + [s["event_id"] for s in neg_buckets[i]]
        rng.shuffle(bucket)
        final.append(bucket)
    return tuple(final)  # type: ignore[return-value]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def flatten_shards(manifest_df: pd.DataFrame, events: Iterable[str], 
                   feature_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    selected = manifest_df[manifest_df["event_id"].isin(set(str(e) for e in events))]
    if selected.empty:
        raise RuntimeError("No files matched the provided events.")

    feat_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    for _, row in selected.iterrows():
        x = np.load(row["x_path"], mmap_mode="r")
        y = np.load(row["y_path"], mmap_mode="r")
        if x.shape[0] == 0:
            continue
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch between {row['x_path']} and {row['y_path']}.")
        
        # Apply feature mask if provided (before flattening)
        if feature_mask is not None:
            # x shape is (N, W, F) or (N, F, W)
            # Need to determine which dimension is features
            if len(x.shape) == 3:
                n_samples, dim1, dim2 = x.shape
                # Check which dimension matches feature mask size
                if dim1 == len(feature_mask):
                    # (N, F, W) - features are dim1
                    x = x[:, feature_mask, :]
                elif dim2 == len(feature_mask):
                    # (N, W, F) - features are dim2
                    x = x[:, :, feature_mask]
                else:
                    raise ValueError(f"Feature mask size {len(feature_mask)} doesn't match data shape {x.shape}")
        
        feat_parts.append(np.asarray(x.reshape(x.shape[0], -1), dtype=np.float32))
        label_parts.append(np.asarray(y.reshape(-1), dtype=np.float32))

    if not feat_parts:
        raise RuntimeError("Selected events yielded zero samples.")

    X = np.concatenate(feat_parts, axis=0)
    y = np.concatenate(label_parts, axis=0)
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate_scores(scores: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    preds = (scores >= 0.5).astype(int)
    p, r, f1 = precision_recall_f1(targets.astype(int), preds)
    best = (0.0, 0.0, 0.0, 0.5)
    for t in np.linspace(0.0, 1.0, 200):
        cand = (scores >= t).astype(int)
        pb, rb, f1b = precision_recall_f1(targets.astype(int), cand)
        if f1b > best[0]:
            best = (f1b, pb, rb, t)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "best_f1": float(best[0]),
        "best_p": float(best[1]),
        "best_r": float(best[2]),
        "best_t": float(best[3]),
    }


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return neg / max(pos, 1.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train an XGBoost baseline on SCADA shards.")
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing manifest + shards.")
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=30)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--tree_method", type=str, default="hist")
    ap.add_argument("--use_gpu", action="store_true", help="Force gpu_hist tree method.")
    ap.add_argument(
        "--scale_pos_weight",
        type=float,
        default=None,
        help="Override class weight ratio. Defaults to neg/pos from training split.",
    )
    ap.add_argument(
        "--selected_features",
        type=str,
        default=None,
        help="Path to JSON file with list of selected feature names to use",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    seed_everything(args.seed)
    manifest_df = read_manifest(in_dir)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train_events, val_events, test_events = train_val_test_split_event(manifest_df, args.seed, ratios=ratios)

    # Load feature selection if provided
    feature_mask = None
    feature_names = None
    if args.selected_features:
        selected_path = Path(args.selected_features)
        if not selected_path.is_absolute():
            selected_path = in_dir / selected_path
        if selected_path.exists():
            with open(selected_path, "r", encoding="utf-8") as f:
                selected_features = json.load(f)
            print(f"[INFO] Using {len(selected_features)} selected features from {selected_path}")
            
            # Load feature names for filtering
            feature_names_path = in_dir / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
                selected_set = set(selected_features)
                feature_mask = np.array([f in selected_set for f in feature_names], dtype=bool)
                print(f"[INFO] Feature mask: {feature_mask.sum()}/{len(feature_names)} features selected")
            else:
                print("[WARN] feature_names.json not found, cannot apply feature selection")
        else:
            print(f"[WARN] Selected features file not found: {selected_path}")

    print(
        f"Loading train events ({len(train_events)}), "
        f"val events ({len(val_events)}), test events ({len(test_events)})."
    )
    X_train, y_train = flatten_shards(manifest_df, train_events, feature_mask=feature_mask)
    X_val, y_val = flatten_shards(manifest_df, val_events, feature_mask=feature_mask)
    X_test, y_test = flatten_shards(manifest_df, test_events, feature_mask=feature_mask)

    print(f"Train samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")

    scale_pos_weight = args.scale_pos_weight
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    tree_method = "gpu_hist" if args.use_gpu else args.tree_method
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=args.n_jobs,
        scale_pos_weight=scale_pos_weight,
        tree_method=tree_method,
        random_state=args.seed,
        verbosity=1,
    )

    callbacks: List[xgb.callback.TrainingCallback] = []
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=args.early_stopping_rounds,
                save_best=True,
                data_name="validation_1",  # corresponds to the validation split in eval_set
                maximize=False,
            )
        )
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_train, y_train), (X_val, y_val)],
        "verbose": True,
    }
    if callbacks:
        try:
            model.fit(callbacks=callbacks, **fit_kwargs)
        except TypeError:
            print("Current xgboost version lacks callback support in sklearn API; continuing without early stopping.")
            model.fit(**fit_kwargs)
    else:
        model.fit(**fit_kwargs)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    val_metrics = evaluate_scores(val_scores, y_val)
    test_metrics = evaluate_scores(test_scores, y_test)

    print(
        f"[Validation] F1={val_metrics['f1']:.4f} bestF1={val_metrics['best_f1']:.4f} "
        f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}"
    )
    print(
        f"[Test]       F1={test_metrics['f1']:.4f} bestF1={test_metrics['best_f1']:.4f} "
        f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}"
    )

    summary = {
        "splits": {"train_events": train_events, "val_events": val_events, "test_events": test_events},
        "features": {"input_dim": int(X_train.shape[1])},
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "xgboost_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "gamma": args.gamma,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "tree_method": tree_method,
            "scale_pos_weight": scale_pos_weight,
        },
        "metrics": {"val": val_metrics, "test": test_metrics},
    }

    out_path = in_dir / "xgboost_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved metrics to", out_path)


if __name__ == "__main__":
    main()

