#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm_early_warning_sharded.py  (FAST VERSION)

- Train LSTM trên các shard SCADA early-warning (manifest.txt).
- Memory-friendly (mmap), event-based split.
- WeightedRandomSampler + pos_weight.
- Val/Test có PR-AUC + best-F1; Train chỉ log loss (nhanh hơn nhiều).

Usage
python train_lstm_early_warning_sharded.py --in_dir "Dataset/processed" --epochs 8 --batch_size 256 --hidden 64 --layers 1 \
  --lr 1e-3 --dropout 0.1 --num_workers 4 --seed 42 --weighted_sampler
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm


# -------------------------
# Data utilities
# -------------------------

def read_manifest(in_dir: Path) -> pd.DataFrame:
    rows = []
    man_path = in_dir / "manifest.txt"
    if not man_path.exists():
        raise FileNotFoundError(f"manifest.txt not found at {man_path}")
    for line in man_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        event_id, x_name, y_name, meta_name = line.strip().split(",")
        rows.append({
            "event_id": str(event_id),
            "x_path": str((in_dir / x_name).as_posix()),
            "y_path": str((in_dir / y_name).as_posix()),
            "meta_path": str((in_dir / meta_name).as_posix()),
        })
    return pd.DataFrame(rows)


def train_val_test_split_event(manifest_df: pd.DataFrame, seed: int,
                               ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)):
    rng = random.Random(seed)
    if len(ratios) != 3:
        raise ValueError("ratios must have exactly 3 values (train/val/test).")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("Sum of ratios must be positive.")
    ratios = tuple(max(r, 0.0) / ratio_sum for r in ratios)

    # Đếm positives per event
    evt_stats = []
    for eid, grp in manifest_df.groupby("event_id"):
        pos = 0
        total = 0
        for _, r in grp.iterrows():
            y = load_array(r["y_path"], mmap_mode="r")
            pos += int((y == 1).sum())
            total += int(y.shape[0])
        evt_stats.append({"event_id": str(eid), "positives": pos, "total": total})

    def calc_targets(total_events: int):
        if total_events == 0:
            return [0, 0, 0]
        raw = [r * total_events for r in ratios]
        targets = [int(x) for x in raw]
        remainder = total_events - sum(targets)
        fracs = [x - int(x) for x in raw]
        # phân bổ phần dư theo phần thập phân lớn nhất
        while remainder > 0:
            idx = max(range(len(targets)), key=lambda i: (fracs[i], -i))
            targets[idx] += 1
            fracs[idx] = 0.0
            remainder -= 1
        # đảm bảo không bucket nào 0 nếu có đủ sự kiện
        for i in range(len(targets)):
            if total_events >= len(targets) and targets[i] == 0:
                donor = max(range(len(targets)), key=lambda j: targets[j])
                if targets[donor] > 1:
                    targets[donor] -= 1
                    targets[i] = 1
        return targets

    pos_stats = [s for s in evt_stats if s["positives"] > 0]
    neg_stats = [s for s in evt_stats if s["positives"] == 0]

    def distribute(stats_list, targets, weight_key):
        buckets = [[] for _ in targets]
        loads = [0.0 for _ in targets]
        stats_list = stats_list.copy()
        rng.shuffle(stats_list)
        for st in stats_list:
            available = [i for i in range(len(targets)) if targets[i] > 0 and len(buckets[i]) < targets[i]]
            if not available:
                available = list(range(len(targets)))
            idx = min(available, key=lambda i: (loads[i], len(buckets[i])))
            buckets[idx].append(st)
            loads[idx] += weight_key(st)
        return buckets

    pos_targets = calc_targets(len(pos_stats))
    neg_targets = calc_targets(len(neg_stats))

    pos_buckets = distribute(pos_stats, pos_targets, lambda s: max(1, s["positives"]))
    neg_buckets = distribute(neg_stats, neg_targets, lambda s: max(1, s["total"]))

    # đảm bảo mỗi bucket có ≥1 positive nếu có thể
    if pos_stats:
        for i in range(len(pos_buckets)):
            if pos_buckets[i]:
                continue
            donor_idx = max(range(len(pos_buckets)), key=lambda j: len(pos_buckets[j]))
            if len(pos_buckets[donor_idx]) > 1:
                pos_buckets[i].append(pos_buckets[donor_idx].pop())

    final_buckets = []
    for i in range(len(ratios)):
        bucket_events = [s["event_id"] for s in pos_buckets[i]] + \
                        [s["event_id"] for s in neg_buckets[i]]
        rng.shuffle(bucket_events)
        final_buckets.append(bucket_events)

    train_e, val_e, test_e = final_buckets

    # nếu bucket rỗng, mượn bớt từ bucket lớn nhất
    for bucket in (train_e, val_e, test_e):
        if len(bucket) == 0:
            src = max((train_e, val_e, test_e), key=len)
            if len(src) > 1:
                bucket.append(src.pop())

    # Validate that each split has enough positive samples
    def count_positives(events: List[str]) -> int:
        total_pos = 0
        for eid in events:
            for _, r in manifest_df[manifest_df["event_id"] == eid].iterrows():
                y = load_array(r["y_path"], mmap_mode="r")
                total_pos += int((y == 1).sum())
        return total_pos

    train_pos = count_positives(train_e)
    val_pos = count_positives(val_e)
    test_pos = count_positives(test_e)
    total_pos = train_pos + val_pos + test_pos

    # Minimum positive samples per split (at least 10, or 1% of total positives, whichever is larger)
    min_pos_per_split = max(10, int(total_pos * 0.01)) if total_pos > 0 else 0

    # Warn if any split has too few positives
    if total_pos > 0:
        splits_info = [
            ("Train", train_e, train_pos),
            ("Val", val_e, val_pos),
            ("Test", test_e, test_pos)
        ]
        for split_name, events, pos_count in splits_info:
            if pos_count < min_pos_per_split:
                print(f"[WARN] {split_name} split has only {pos_count} positive samples (min recommended: {min_pos_per_split})")
                print(f"       Events in {split_name}: {events}")
        print(f"[INFO] Positive samples per split: Train={train_pos}, Val={val_pos}, Test={test_pos} (Total: {total_pos})")
    
    return train_e, val_e, test_e


def load_array(path: str, mmap_mode: str = "r"):
    """Load numpy array from .npy or .npz file."""
    path_obj = Path(path)
    # If path already has .npz extension, load directly
    if path_obj.suffix == ".npz":
        data = np.load(str(path_obj), mmap_mode=None)  # npz doesn't support mmap
        key = "X" if "X_" in path_obj.name else "y"
        return data[key]
    # Otherwise try .npz first (compressed), then .npy
    npz_path = path_obj.with_suffix(".npz")
    if npz_path.exists():
        data = np.load(str(npz_path), mmap_mode=None)
        key = "X" if "X_" in path_obj.name else "y"
        return data[key]
    else:
        # Standard .npy format
        return np.load(path, mmap_mode=mmap_mode)

class ShardDataset(Dataset):
    """
    Lazily loads shards, keeps a small cache of opened arrays.
    index_map: list of (shard_idx, local_idx) covering only selected event_ids.
    """
    def __init__(self, manifest_df: pd.DataFrame, selected_events: List[str], 
                 feature_names: Optional[List[str]] = None, selected_features: Optional[List[str]] = None):
        self.mani = manifest_df.reset_index(drop=True)
        self.sel_idx = self.mani[self.mani["event_id"].isin(selected_events)].index.tolist()
        if len(self.sel_idx) == 0:
            raise RuntimeError("No shards matched the selected events.")

        # Feature selection: if selected_features provided, create feature mask
        self.feature_mask = None
        if selected_features is not None and feature_names is not None:
            selected_set = set(selected_features)
            self.feature_mask = np.array([f in selected_set for f in feature_names], dtype=bool)
            print(f"[INFO] Using {self.feature_mask.sum()}/{len(feature_names)} selected features")

        # load sizes and build index map
        self.shard_sizes = []
        for i in self.sel_idx:
            y = load_array(self.mani.loc[i, "y_path"], mmap_mode="r")
            self.shard_sizes.append(int(y.shape[0]))

        self.index_map = []
        for s_i, shard_idx in enumerate(self.sel_idx):
            size = self.shard_sizes[s_i]
            self.index_map.extend([(shard_idx, j) for j in range(size)])

        self._x_cache: Dict[int, Any] = {}
        self._y_cache: Dict[int, Any] = {}

        # infer dims từ shard đầu tiên không rỗng
        first_nonempty = None
        for shard_idx in self.sel_idx:
            x = load_array(self.mani.loc[shard_idx, "x_path"], mmap_mode="r")
            if x.shape[0] > 0:
                first_nonempty = x
                break
        if first_nonempty is None:
            raise RuntimeError("All selected shards are empty.")
        
        # Data shape from preprocessing: (N, W, F) where W is window size and F is num features
        print(f"[DEBUG] First nonempty data shape: {first_nonempty.shape}")
        
        if len(first_nonempty.shape) != 3:
            raise RuntimeError(f"Expected 3D array shape (N, W, F), got {first_nonempty.shape}")
        
        n_samples, dim1, dim2 = first_nonempty.shape
        expected_feature_counts: List[int] = []
        if feature_names:
            expected_feature_counts.append(len(feature_names))
        if selected_features:
            expected_feature_counts.append(len(selected_features))
        expected_feature_counts = sorted(set(expected_feature_counts))

        def matches_feature_dim(dim_val: int) -> bool:
            return any(dim_val == cnt for cnt in expected_feature_counts)

        matches_dim1 = matches_feature_dim(dim1)
        matches_dim2 = matches_feature_dim(dim2)

        if matches_dim2 and not matches_dim1:
            # Standard format: (N, W, F)
            self.W = dim1
            actual_num_features = dim2
            self._needs_transpose = False
        elif matches_dim1 and not matches_dim2:
            # Transposed format: (N, F, W)
            self.W = dim2
            actual_num_features = dim1
            self._needs_transpose = True
            print(f"[WARN] Data appears transposed: interpreting shape {first_nonempty.shape} as (N, F, W). Will transpose during loading.")
        elif matches_dim1 and matches_dim2:
            # Both axes match expected feature count (rare, e.g., window size == feature count)
            self.W = dim1
            actual_num_features = dim2
            self._needs_transpose = False
            print(f"[WARN] Both axes match expected feature count(s) {expected_feature_counts}. Defaulting to (N, W, F) interpretation.")
        else:
            # Fallback heuristics when metadata is missing or inconsistent
            if not expected_feature_counts:
                print("[WARN] Could not find feature metadata to determine feature dimension. Falling back to heuristic inference.")
            else:
                print(
                    "[WARN] Unable to match feature dimension from metadata "
                    f"(expected {expected_feature_counts}, got dims {dim1} and {dim2}). "
                    "Inferring orientation heuristically."
                )

            if dim1 == dim2:
                self.W = dim1
                actual_num_features = dim2
                self._needs_transpose = False
            elif dim1 > dim2:
                # Assume dim1 is window (most common case when window > features)
                self.W = dim1
                actual_num_features = dim2
                self._needs_transpose = False
            else:
                # Assume dim2 is window -> data stored as (N, F, W)
                self.W = dim2
                actual_num_features = dim1
                self._needs_transpose = True
                print(f"[WARN] Falling back to treat shape {first_nonempty.shape} as (N, F, W). Please verify preprocessing settings.")
        
        print(f"[INFO] Detected: window_size={self.W}, num_features={actual_num_features}, transpose={self._needs_transpose}")
        
        # If features were already filtered during preprocessing, disable mask
        if self.feature_mask is not None:
            expected_num_features = int(self.feature_mask.sum())
            total_feature_count = len(feature_names) if feature_names is not None else None

            if actual_num_features == expected_num_features:
                # Features already filtered - no need for mask
                print(f"[INFO] Data already has {actual_num_features} features (features were filtered during preprocessing). Disabling mask.")
                self.feature_mask = None
                self.F = actual_num_features
            elif total_feature_count is not None and actual_num_features == total_feature_count:
                # Data has all features - use mask to select
                self.F = expected_num_features
            else:
                total_feat_msg = total_feature_count if total_feature_count is not None else "unknown"
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"FEATURE COUNT MISMATCH ERROR\n"
                    f"{'='*60}\n"
                    f"Data has: {actual_num_features} features\n"
                    f"Selected features expects: {expected_num_features} features\n"
                    f"All features: {total_feat_msg}\n"
                    f"Data shape: {first_nonempty.shape}\n\n"
                    f"SOLUTION:\n"
                    f"  Ensure preprocessing was run with --use_selected_features when providing a selected feature list,\n"
                    f"  or regenerate the dataset so that feature dimensions align.\n"
                    f"{'='*60}"
                )
        else:
            self.F = actual_num_features

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_idx, j = self.index_map[idx]
        if shard_idx not in self._x_cache:
            self._x_cache[shard_idx] = load_array(self.mani.loc[shard_idx, "x_path"], mmap_mode="r")
            self._y_cache[shard_idx] = load_array(self.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        X_shard = self._x_cache[shard_idx]
        y_shard = self._y_cache[shard_idx]

        # tránh copy thừa: asarray + from_numpy
        x_np = np.asarray(X_shard[j], dtype=np.float32, copy=True)  # (W, F) or (F, W)
        
        # Transpose if needed (data stored as (F, W) instead of (W, F))
        if getattr(self, '_needs_transpose', False):
            x_np = x_np.T  # (F, W) -> (W, F)
        
        # Apply feature mask if available (mask is None if features already filtered)
        if self.feature_mask is not None:
            x_np = x_np[:, self.feature_mask]  # (W, F_selected)
        
        x = torch.from_numpy(x_np)        # (W, F)
        y = torch.tensor(float(y_shard[j]), dtype=torch.float32)
        return x, y

# -------------------------
# Model
# -------------------------

class LSTMBinary(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x):  # x: (B, W, F)
        h, _ = self.lstm(x)                  # (B, W, H[*2])
        last = h[:, -1, :]                   # (B, H[*2])
        logit = self.head(last).squeeze(-1)  # (B,)
        return logit

# -------------------------
# Metrics (numpy-based)
# -------------------------

def sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -20.0, 20.0, out=np.empty_like(z, dtype=np.float64))
    return 1.0 / (1.0 + np.exp(-z))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def pr_auc_from_scores(y_true: np.ndarray, scores: np.ndarray, num_thresh: int = 200) -> float:
    if scores.size == 0:
        return 0.0
    smin, smax = float(scores.min()), float(scores.max())
    if smin == smax:
        y_pred = (scores >= 0.0).astype(int)
        p, r, _ = precision_recall_f1(y_true, y_pred)
        return p * r
    ths = np.linspace(smin, smax, num_thresh)
    precs, recs = [], []
    for t in ths:
        y_pred = (scores >= t).astype(int)
        p, r, _ = precision_recall_f1(y_true, y_pred)
        precs.append(p)
        recs.append(r)
    recs = np.array(recs)
    precs = np.array(precs)
    order = np.argsort(recs)
    recs, precs = recs[order], precs[order]
    auc = 0.0
    for i in range(1, len(recs)):
        auc += (recs[i] - recs[i - 1]) * precs[i]
    return float(auc)

# -------------------------
# Training
# -------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(ds: ShardDataset) -> float:
    pos = 0
    total = 0
    seen = set()
    for shard_idx in ds.sel_idx:
        if shard_idx in seen:
            continue
        y = load_array(ds.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        pos += int((y == 1).sum())
        total += int(y.shape[0])
        seen.add(shard_idx)
    neg = max(1, total - pos)
    pos = max(1, pos)
    return float(neg / pos)


def run_epoch(model, loader, device, criterion, optimizer=None,
              grad_clip: float = 1.0, scaler: GradScaler = None,
              collect_logits: bool = False):
    """
    - Nếu optimizer is None -> eval mode, luôn collect logits.
    - Nếu optimizer not None:
        + collect_logits=False -> chỉ train loss (nhanh hơn).
        + collect_logits=True  -> vẫn collect (ít dùng).
    """
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    all_logits = [] if collect_logits or not is_train else None
    all_targets = [] if collect_logits or not is_train else None

    loop = tqdm(loader, desc="Train" if is_train else "Eval", leave=False)
    for batch_idx, (x, y) in enumerate(loop, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                with torch.amp.autocast(device_type=device.type):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)

        loss_val = float(loss.detach().cpu().item())
        total_loss += loss_val * x.size(0)
        loop.set_postfix(loss=loss_val, batches=batch_idx)

        if collect_logits or not is_train:
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    if all_logits is not None:
        logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
        targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    else:
        logits_np = np.array([])
        targets_np = np.array([])

    return total_loss / max(1, len(loader.dataset)), logits_np, targets_np


def evaluate_from_logits(logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    scores = sigmoid_np(logits)
    y_hat = (scores >= 0.5).astype(int)
    p, r, f1 = precision_recall_f1(targets.astype(int), y_hat)

    ths = np.linspace(0.0, 1.0, 200)
    best = (0.0, 0.0, 0.0, 0.5)  # f1, p, r, t
    for t in ths:
        yb = (scores >= t).astype(int)
        pb, rb, f1b = precision_recall_f1(targets.astype(int), yb)
        if f1b > best[0]:
            best = (f1b, pb, rb, t)

    pr_auc = pr_auc_from_scores(targets.astype(int), scores, num_thresh=200)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "best_f1": float(best[0]),
        "best_p": float(best[1]),
        "best_r": float(best[2]),
        "best_t": float(best[3]),
    }


def build_weighted_sampler(ds: ShardDataset, pos_mult: float = 10.0):
    # gán weight cao hơn cho các sample dương tính
    weights = np.zeros(len(ds), dtype=np.float32)
    for i, (shard_idx, j) in enumerate(ds.index_map):
        if shard_idx not in ds._y_cache:
            ds._y_cache[shard_idx] = load_array(ds.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        y = ds._y_cache[shard_idx][j]
        weights[i] = pos_mult if y == 1 else 1.0
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(ds),
        replacement=True,
    )
    return sampler

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weighted_sampler", action="store_true",
                    help="Use WeightedRandomSampler on train to balance classes")
    ap.add_argument("--train_ratio", type=float, default=0.70,
                    help="Proportion of events allocated to train split (before normalization).")
    ap.add_argument("--val_ratio", type=float, default=0.15,
                    help="Proportion of events allocated to validation split.")
    ap.add_argument("--test_ratio", type=float, default=0.15,
                    help="Proportion of events allocated to test split.")
    ap.add_argument("--selected_features", type=str, default=None,
                    help="Path to JSON file with list of selected feature names to use")
    ap.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience: stop if validation F1 doesn't improve for N epochs")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    seed_everything(args.seed)

    manifest_df = read_manifest(in_dir)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train_e, val_e, test_e = train_val_test_split_event(manifest_df, seed=args.seed, ratios=ratios)

    # Load feature names and selected features if provided
    feature_names = None
    selected_features = None
    feature_names_path = in_dir / "feature_names.json"
    if feature_names_path.exists():
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    
    if args.selected_features:
        selected_path = Path(args.selected_features)
        if not selected_path.is_absolute():
            selected_path = in_dir / selected_path
        if selected_path.exists():
            with open(selected_path, "r", encoding="utf-8") as f:
                selected_features = json.load(f)
            print(f"[INFO] Using {len(selected_features)} selected features from {selected_path}")

    ds_train = ShardDataset(manifest_df, train_e, feature_names=feature_names, selected_features=selected_features)
    ds_val = ShardDataset(manifest_df, val_e, feature_names=feature_names, selected_features=selected_features)
    ds_test = ShardDataset(manifest_df, test_e, feature_names=feature_names, selected_features=selected_features)

    device = torch.device(args.device)

    # DataLoaders
    pin_memory = (device.type == "cuda")
    if args.weighted_sampler:
        sampler = build_weighted_sampler(ds_train, pos_mult=10.0)
        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


    model = LSTMBinary(
        in_features=ds_train.F,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    # pos_weight to address imbalance
    pos_weight_val = compute_pos_weight(ds_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed precision scaler (bật khi dùng CUDA)
    use_amp = (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Train: không collect logits để tiết kiệm thời gian + RAM
        tr_loss, _, _ = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            grad_clip=1.0,
            scaler=scaler,
            collect_logits=False,
        )
        # Val: collect logits để tính metric
        val_loss, val_logits, val_targets = run_epoch(
            model,
            val_loader,
            device,
            criterion,
            optimizer=None,
            grad_clip=None,
            scaler=None,
            collect_logits=True,
        )

        va_metrics = evaluate_from_logits(val_logits, val_targets)

        # Early stopping: track best validation F1
        improved = False
        if va_metrics["best_f1"] > best_val_f1:
            best_val_f1 = va_metrics["best_f1"]
            best_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                          for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
            improved = True
        else:
            patience_counter += 1

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"F1={va_metrics['f1']:.4f} "
            f"bestF1={va_metrics['best_f1']:.4f} "
            f"p={va_metrics['precision']:.4f} "
            f"r={va_metrics['recall']:.4f} "
            f"prAUC={va_metrics['pr_auc']:.4f}"
            + (f" *" if improved else "")
        )

        # Early stopping check
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Best F1={best_val_f1:.4f} at epoch {best_epoch}")
            break

    # Evaluate on test with best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_logits, te_targets = run_epoch(
        model,
        test_loader,
        device,
        criterion,
        optimizer=None,
        grad_clip=None,
        scaler=None,
        collect_logits=True,
    )
    te_metrics = evaluate_from_logits(te_logits, te_targets)

    out = {
        "val_best_f1": float(best_val_f1),
        "test_default": {
            "precision": float(te_metrics["precision"]),
            "recall": float(te_metrics["recall"]),
            "f1": float(te_metrics["f1"]),
            "pr_auc": float(te_metrics["pr_auc"]),
        },
        "test_best_f1": {
            "threshold": float(te_metrics["best_t"]),
            "precision": float(te_metrics["best_p"]),
            "recall": float(te_metrics["best_r"]),
            "f1": float(te_metrics["best_f1"]),
        },
        "split": {
            "train_events": train_e,
            "val_events": val_e,
            "test_events": test_e,
        },
        "model": {
            "in_features": ds_train.F,
            "window": ds_train.W,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
        },
        "loss": {"pos_weight": float(pos_weight_val)},
    }
    with open(in_dir / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    torch.save(model.state_dict(), in_dir / "lstm_early_warning.pt")
    print("\nSaved metrics to", (in_dir / "lstm_metrics.json"))
    print("Saved model weights to", (in_dir / "lstm_early_warning.pt"))


if __name__ == "__main__":
    main()
