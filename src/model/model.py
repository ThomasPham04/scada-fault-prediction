
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm_early_warning_sharded.py

Train an LSTM on SCADA early-warning shards produced by
scada_preprocess_early_warning_sharded(_fast[_fix]).py.

Key features
- Reads shards via manifest.txt, uses np.load(..., mmap_mode='r') lazily.
- No giant X.npy in RAM; per-batch loading only.QZ
- Splits by event_id -> Train/Val/Test (70/15/15) with reproducible seed.
- Handles extreme class imbalance with BCEWithLogitsLoss(pos_weight).
- Reports default (0.5) metrics + best-F1 threshold and PR-AUC (numpy implementation).

Usage
python train_lstm_early_warning_sharded.py \
  --in_dir "Dataset/processed" \
  --epochs 8 --batch_size 256 --hidden 64 --layers 1 \
  --lr 1e-3 --dropout 0.1 --num_workers 2 --seed 42
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

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

def train_val_test_split_event(manifest_df: pd.DataFrame, seed: int):
    rng = random.Random(seed)

    # Đếm positives per event bằng cách đọc y_*.npy
    evt_stats = []
    for eid, grp in manifest_df.groupby("event_id"):
        pos = 0
        total = 0
        for _, r in grp.iterrows():
            y = np.load(r["y_path"], mmap_mode="r")
            pos += int((y == 1).sum()); total += int(y.shape[0])
        evt_stats.append((str(eid), pos, total))

    pos_events = [e for e,p,_ in evt_stats if p > 0]
    neg_events = [e for e,p,_ in evt_stats if p == 0]
    rng.shuffle(pos_events); rng.shuffle(neg_events)

    def split(lst, r_train=0.70, r_val=0.15):
        n=len(lst); n_tr=max(0, int(round(r_train*n))); n_va=max(0, int(round(r_val*n)))
        tr=lst[:n_tr]; va=lst[n_tr:n_tr+n_va]; te=lst[n_tr+n_va:]; return tr,va,te

    p_tr,p_va,p_te = split(pos_events)
    n_tr,n_va,n_te = split(neg_events)

    train_e = p_tr + n_tr
    val_e   = p_va + n_va
    test_e  = p_te + n_te

    # đảm bảo mỗi bucket có ≥1 positive nếu có thể
    def ensure_has_pos(bucket):
        if not any(e in pos_events for e in bucket) and len(pos_events)>0:
            # mượn 1 pos từ bucket khác còn dôi
            for src in (train_e, val_e, test_e):
                if src is bucket: 
                    continue
                for e in list(src):
                    if e in pos_events and (len(src) > 1):
                        src.remove(e); bucket.append(e); return
    ensure_has_pos(train_e); ensure_has_pos(val_e); ensure_has_pos(test_e)

    # nếu bucket rỗng, mượn bớt từ bucket lớn nhất
    for bucket in (train_e, val_e, test_e):
        if len(bucket) == 0:
            src = max((train_e, val_e, test_e), key=len)
            if len(src) > 1:
                bucket.append(src.pop())

    return train_e, val_e, test_e

class ShardDataset(Dataset):
    """
    Lazily loads shards, keeps a small cache of opened arrays.
    index_map: list of (shard_idx, local_idx) covering only selected event_ids.
    """
    def __init__(self, manifest_df: pd.DataFrame, selected_events: List[str]):
        self.mani = manifest_df.reset_index(drop=True)
        self.sel_idx = self.mani[self.mani["event_id"].isin(selected_events)].index.tolist()
        if len(self.sel_idx) == 0:
            raise RuntimeError("No shards matched the selected events.")
        # load sizes and build index map
        self.shard_sizes = []
        for i in self.sel_idx:
            y = np.load(self.mani.loc[i, "y_path"], mmap_mode="r")
            self.shard_sizes.append(int(y.shape[0]))
        self.index_map = []
        for s_i, shard_idx in enumerate(self.sel_idx):
            size = self.shard_sizes[s_i]
            self.index_map.extend([(shard_idx, j) for j in range(size)])
        self._x_cache: Dict[int, Any] = {}
        self._y_cache: Dict[int, Any] = {}

        # infer dims from first non-empty shard
        feat_json = self.mani["x_path"].iloc[0]
        first_nonempty = None
        for s_i, shard_idx in enumerate(self.sel_idx):
            x = np.load(self.mani.loc[shard_idx, "x_path"], mmap_mode="r")
            if x.shape[0] > 0:
                first_nonempty = x
                break
        if first_nonempty is None:
            raise RuntimeError("All selected shards are empty.")
        self.W = first_nonempty.shape[1]
        self.F = first_nonempty.shape[2]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_idx, j = self.index_map[idx]
        if shard_idx not in self._x_cache:
            self._x_cache[shard_idx] = np.load(self.mani.loc[shard_idx, "x_path"], mmap_mode="r")
            self._y_cache[shard_idx] = np.load(self.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        X_shard = self._x_cache[shard_idx]
        y_shard = self._y_cache[shard_idx]
        x = np.array(X_shard[j], dtype=np.float32, copy=True)  # copy -> writable
        y = float(y_shard[j])
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# -------------------------
# Model
# -------------------------

class LSTMBinary(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0, bidirectional: bool=False):
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
        h, _ = self.lstm(x)            # (B, W, H[*2])
        last = h[:, -1, :]             # (B, H[*2])
        logit = self.head(last).squeeze(-1)  # (B,)
        return logit

# -------------------------
# Metrics (numpy-based)
# -------------------------

def sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1

def pr_auc_from_scores(y_true: np.ndarray, scores: np.ndarray, num_thresh: int = 200) -> float:
    # approximate PR-AUC by scanning thresholds over score range
    if scores.size == 0:
        return 0.0
    smin, smax = float(scores.min()), float(scores.max())
    if smin == smax:
        # constant scores -> degenerate curve
        y_pred = (scores >= 0.0).astype(int)
        p, r, _ = precision_recall_f1(y_true, y_pred)
        return p * r
    ths = np.linspace(smin, smax, num_thresh)
    precs, recs = [], []
    for t in ths:
        y_pred = (scores >= t).astype(int)
        p, r, _ = precision_recall_f1(y_true, y_pred)
        precs.append(p); recs.append(r)
    # sort by recall and integrate
    recs = np.array(recs)
    precs = np.array(precs)
    order = np.argsort(recs)
    recs, precs = recs[order], precs[order]
    # Riemann sum
    auc = 0.0
    for i in range(1, len(recs)):
        auc += (recs[i] - recs[i-1]) * precs[i]
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
    # scan y lazily
    pos = 0
    total = 0
    seen = set()
    for shard_idx in ds.sel_idx:
        if shard_idx in seen:
            continue
        y = np.load(ds.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        pos += int((y == 1).sum())
        total += int(y.shape[0])
        seen.add(shard_idx)
    neg = max(1, total - pos)
    pos = max(1, pos)
    return float(neg / pos)

def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip: float = 1.0):
    model.train(mode=optimizer is not None)
    total_loss = 0.0
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.detach().cpu().item()) * x.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    return total_loss / max(1, len(loader.dataset)), logits, targets

def evaluate_from_logits(logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    scores = sigmoid_np(logits)
    y_hat = (scores >= 0.5).astype(int)
    p, r, f1 = precision_recall_f1(targets.astype(int), y_hat)
    # best-F1 search on 200 thresholds between 0 and 1
    ths = np.linspace(0.0, 1.0, 200)
    best = (0.0, 0.0, 0.0, 0.5)  # f1, p, r, t
    for t in ths:
        yb = (scores >= t).astype(int)
        pb, rb, f1b = precision_recall_f1(targets.astype(int), yb)
        if f1b > best[0]:
            best = (f1b, pb, rb, t)
    pr_auc = pr_auc_from_scores(targets.astype(int), scores, num_thresh=200)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1), "pr_auc": float(pr_auc),
        "best_f1": float(best[0]), "best_p": float(best[1]), "best_r": float(best[2]), "best_t": float(best[3])
    }

def build_weighted_sampler(ds: Dataset, pos_mult: float = 10.0):
    # gán weight cao hơn cho các sample dương tính
    weights = np.zeros(len(ds), dtype=np.float32)
    for i, (shard_idx, j) in enumerate(ds.index_map):  # index_map đã có trong ShardDataset
        if shard_idx not in ds._y_cache:
            ds._y_cache[shard_idx] = np.load(ds.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        y = ds._y_cache[shard_idx][j]
        weights[i] = pos_mult if y == 1 else 1.0
    sampler = WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(ds), replacement=True)
    return sampler
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
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    seed_everything(args.seed)

    manifest_df = read_manifest(in_dir)
    train_e, val_e, test_e = train_val_test_split_event(manifest_df, seed=args.seed)

    ds_train = ShardDataset(manifest_df, train_e)
    if args.weighted_sampler:
        sampler = build_weighted_sampler(ds_train, pos_mult=3.0)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    ds_val   = ShardDataset(manifest_df, val_e)
    ds_test  = ShardDataset(manifest_df, test_e)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = LSTMBinary(in_features=ds_train.F, hidden=args.hidden, layers=args.layers,
                       dropout=args.dropout, bidirectional=args.bidirectional).to(args.device)

    # pos_weight to address imbalance
    pos_weight_val = compute_pos_weight(ds_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_logits, tr_targets = run_epoch(model, train_loader, args.device, criterion, optimizer)
        val_loss, val_logits, val_targets = run_epoch(model, val_loader, args.device, criterion, optimizer=None)

        tr_metrics = evaluate_from_logits(tr_logits, tr_targets)
        va_metrics = evaluate_from_logits(val_logits, val_targets)

        if va_metrics["best_f1"] > best_val_f1:
            best_val_f1 = va_metrics["best_f1"]
            best_state = {k: v.cpu() if hasattr(v, "is_cuda") else v for k, v in model.state_dict().items()}

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={tr_loss:.4f} F1={tr_metrics['f1']:.4f} "
              f"val_loss={val_loss:.4f} F1={va_metrics['f1']:.4f} bestF1={va_metrics['best_f1']:.4f} "
              f"p={va_metrics['precision']:.4f} r={va_metrics['recall']:.4f} prAUC={va_metrics['pr_auc']:.4f}")

    # Evaluate on test with best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_logits, te_targets = run_epoch(model, test_loader, args.device, criterion, optimizer=None)
    te_metrics = evaluate_from_logits(te_logits, te_targets)

    # Save metrics & model
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
            "test_events": test_e
        },
        "model": {
            "in_features": ds_train.F, "window": ds_train.W,
            "hidden": args.hidden, "layers": args.layers,
            "dropout": args.dropout, "bidirectional": args.bidirectional
        },
        "loss": {"pos_weight": float(pos_weight_val)}
    }
    with open(in_dir / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    torch.save(model.state_dict(), in_dir / "lstm_early_warning.pt")
    print("\nSaved metrics to", (in_dir / "lstm_metrics.json"))
    print("Saved model weights to", (in_dir / "lstm_early_warning.pt"))

if __name__ == "__main__":
    main()
