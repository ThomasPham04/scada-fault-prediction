#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRU trainer for SCADA early-fault prediction using shard-based datasets.

Usage example:
python src/model/gru.py --in_dir output_144_2016_A --epochs 12 --batch_size 256 \
    --hidden 128 --layers 2 --dropout 0.2 --lr 5e-4 --weight_decay 1e-4 \
    --num_workers 2 --weighted_sampler --train_ratio 0.6 --val_ratio 0.2 --test_ratio 0.2
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import GradScaler, autocast


# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------

def read_manifest(in_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    man_path = in_dir / "manifest.txt"
    if not man_path.exists():
        raise FileNotFoundError(f"manifest.txt not found at {man_path}")
    for line in man_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        event_id, x_name, y_name, meta_name = line.strip().split(",")
        rows.append({
            "event_id": event_id,
            "x_path": str((in_dir / x_name).as_posix()),
            "y_path": str((in_dir / y_name).as_posix()),
            "meta_path": str((in_dir / meta_name).as_posix()),
        })
    return pd.DataFrame(rows)


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
        pos, total = 0, 0
        for _, r in grp.iterrows():
            y = np.load(r["y_path"], mmap_mode="r")
            pos += int((y == 1).sum())
            total += int(y.shape[0])
        evt_stats.append({"event_id": str(eid), "positives": pos, "total": total})

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

    def distribute(stats: List[Dict[str, Any]], targets: Sequence[int], weight_key: str):
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
        bucket = [s["event_id"] for s in pos_buckets[i]] + \
                 [s["event_id"] for s in neg_buckets[i]]
        rng.shuffle(bucket)
        final.append(bucket)
    return tuple(final)  # type: ignore[return-value]


class ShardDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        selected_events: Iterable[str],
        feature_names: List[str] | None = None,
        selected_features: List[str] | None = None,
    ):
        self.mani = manifest_df.reset_index(drop=True)
        sel_events = set(str(e) for e in selected_events)
        self.sel_idx = self.mani[self.mani["event_id"].isin(sel_events)].index.tolist()
        if not self.sel_idx:
            raise RuntimeError("No shards matched the selected events.")
        self.feature_mask: np.ndarray | None = None
        if feature_names is not None and selected_features is not None:
            selected_set = set(selected_features)
            self.feature_mask = np.array([f in selected_set for f in feature_names], dtype=bool)
            print(f"[INFO] Using {self.feature_mask.sum()}/{len(feature_names)} selected features")
        self.index_map: List[Tuple[int, int]] = []
        self._x_cache: Dict[int, np.memmap] = {}
        self._y_cache: Dict[int, np.memmap] = {}
        for shard_idx in self.sel_idx:
            y = np.load(self.mani.loc[shard_idx, "y_path"], mmap_mode="r")
            self.index_map.extend([(shard_idx, i) for i in range(y.shape[0])])

        first = None
        for shard_idx in self.sel_idx:
            x = np.load(self.mani.loc[shard_idx, "x_path"], mmap_mode="r")
            if x.shape[0] > 0:
                first = x
                break
        if first is None:
            raise RuntimeError("All selected shards are empty.")
        self.window = first.shape[1]
        self.features = first.shape[2]
        if self.feature_mask is not None:
            if len(self.feature_mask) != self.features:
                print("[WARN] feature mask length does not match data dimension; disabling mask.")
                self.feature_mask = None
            else:
                self.features = int(self.feature_mask.sum())

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self.index_map[idx]
        if shard_idx not in self._x_cache:
            self._x_cache[shard_idx] = np.load(self.mani.loc[shard_idx, "x_path"], mmap_mode="r")
            self._y_cache[shard_idx] = np.load(self.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        X_shard = self._x_cache[shard_idx]
        y_shard = self._y_cache[shard_idx]
        x_np = np.asarray(X_shard[local_idx], dtype=np.float32, copy=True)
        if self.feature_mask is not None:
            x_np = x_np[:, self.feature_mask]
        y = float(y_shard[local_idx])
        return torch.from_numpy(x_np), torch.tensor(y, dtype=torch.float32)


class GRUClassifier(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


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
        y = np.load(ds.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        pos += int((y == 1).sum())
        total += int(y.shape[0])
        seen.add(shard_idx)
    neg = max(1, total - pos)
    pos = max(1, pos)
    return float(neg / pos)


def build_weighted_sampler(ds: ShardDataset, pos_mult: float = 10.0):
    weights = np.zeros(len(ds), dtype=np.float32)
    for i, (shard_idx, local_idx) in enumerate(ds.index_map):
        if shard_idx not in ds._y_cache:
            ds._y_cache[shard_idx] = np.load(ds.mani.loc[shard_idx, "y_path"], mmap_mode="r")
        weights[i] = pos_mult if ds._y_cache[shard_idx][local_idx] == 1 else 1.0
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(ds),
        replacement=True,
    )


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


def evaluate_from_logits(logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    scores = sigmoid_np(logits)
    default_preds = (scores >= 0.5).astype(int)
    p, r, f1 = precision_recall_f1(targets.astype(int), default_preds)
    best = (0.0, 0.0, 0.0, 0.5)
    for t in np.linspace(0.0, 1.0, 200):
        preds = (scores >= t).astype(int)
        pb, rb, f1b = precision_recall_f1(targets.astype(int), preds)
        if f1b > best[0]:
            best = (f1b, pb, rb, t)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "best_f1": best[0],
        "best_p": best[1],
        "best_r": best[2],
        "best_t": best[3],
    }


def run_epoch(model, loader, device, criterion, optimizer=None,
              scaler: GradScaler | None = None, grad_clip: float | None = 1.0,
              collect_logits: bool = False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    logits_buf, targets_buf = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                with autocast(device_type=device.type):
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

        total_loss += float(loss.detach().cpu().item()) * x.size(0)
        if collect_logits or not is_train:
            logits_buf.append(logits.detach().cpu().numpy())
            targets_buf.append(y.detach().cpu().numpy())

    if logits_buf:
        logits_np = np.concatenate(logits_buf, axis=0)
        targets_np = np.concatenate(targets_buf, axis=0)
    else:
        logits_np = np.array([])
        targets_np = np.array([])
    return total_loss / max(1, len(loader.dataset)), logits_np, targets_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--weighted_sampler", action="store_true")
    ap.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience: stop if no improvement for N epochs")
    ap.add_argument("--early_stop_metric", type=str, default="f1", choices=["f1", "loss"],
                    help="Metric to monitor for early stopping: 'f1' (best_f1) or 'loss' (val_loss)")
    ap.add_argument("--selected_features", type=str, default=None,
                    help="Optional JSON file with list of selected features to use")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    seed_everything(args.seed)
    manifest_df = read_manifest(in_dir)
    feature_names = None
    selected_features = None
    feature_names_path = in_dir / "feature_names.json"
    if feature_names_path.exists():
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    if args.selected_features:
        sel_path = Path(args.selected_features)
        if not sel_path.is_absolute():
            sel_path = in_dir / sel_path
        if not sel_path.exists():
            raise FileNotFoundError(sel_path)
        with open(sel_path, "r", encoding="utf-8") as f:
            selected_features = json.load(f)
        print(f"[INFO] Using {len(selected_features)} selected features from {sel_path}")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train_events, val_events, test_events = train_val_test_split_event(manifest_df, args.seed, ratios=ratios)

    ds_train = ShardDataset(manifest_df, train_events, feature_names=feature_names,
                            selected_features=selected_features)
    ds_val = ShardDataset(manifest_df, val_events, feature_names=feature_names,
                          selected_features=selected_features)
    ds_test = ShardDataset(manifest_df, test_events, feature_names=feature_names,
                           selected_features=selected_features)

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    if args.weighted_sampler:
        sampler = build_weighted_sampler(ds_train)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = GRUClassifier(
        in_features=ds_train.features,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    pos_weight = compute_pos_weight(ds_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type if use_amp else "cpu", enabled=use_amp)

    best_state = None
    best_val = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, device, criterion,
                                  optimizer=optimizer, scaler=scaler, grad_clip=1.0, collect_logits=False)
        val_loss, val_logits, val_targets = run_epoch(model, val_loader, device, criterion,
                                                      optimizer=None, scaler=None, grad_clip=None, collect_logits=True)
        metrics = evaluate_from_logits(val_logits, val_targets)
        
        # Check for improvement based on selected metric
        improved = False
        if args.early_stop_metric == "f1":
            if metrics["best_f1"] > best_val:
                best_val = metrics["best_f1"]
                best_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in model.state_dict().items()}
                improved = True
        else:  # early_stop_metric == "loss"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val = metrics["best_f1"]  # Still track best F1 for reporting
                best_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in model.state_dict().items()}
                improved = True
        
        # Early stopping logic
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"bestF1={metrics['best_f1']:.4f} "
            f"p={metrics['precision']:.4f} "
            f"r={metrics['recall']:.4f}"
        )
        
        if patience_counter >= args.patience:
            early_stopped = True
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping training.")
            print(f"Best validation {args.early_stop_metric}: {best_val if args.early_stop_metric == 'f1' else best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_logits, test_targets = run_epoch(model, test_loader, device, criterion,
                                                     optimizer=None, scaler=None, grad_clip=None, collect_logits=True)
    test_metrics = evaluate_from_logits(test_logits, test_targets)

    summary = {
        "val_best_f1": best_val,
        "test_metrics": test_metrics,
        "splits": {
            "train_events": train_events,
            "val_events": val_events,
            "test_events": test_events,
        },
        "model": {
            "in_features": ds_train.features,
            "window": ds_train.window,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
        },
        "loss": {"pos_weight": pos_weight},
        "training": {
            "total_epochs": epoch,
            "early_stopped": early_stopped,
            "patience": args.patience,
            "early_stop_metric": args.early_stop_metric,
        },
    }
    (in_dir / "gru_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), in_dir / "gru_early_warning.pt")
    print("Saved metrics to", in_dir / "gru_metrics.json")
    print("Saved model weights to", in_dir / "gru_early_warning.pt")


if __name__ == "__main__":
    main()
