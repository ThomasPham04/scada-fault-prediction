"""
eda.py — data_pipeline.eda
Exploratory Data Analysis on a combined SCADA CSV.

Produces statistical summaries and visualizations covering:
  - missing value counts per column
  - per-feature skewness, kurtosis, and Kolmogorov-Smirnov normality test
  - Spearman correlation of each feature with the binary fault label
  - Spearman correlation matrix across features
  - representative time series with fault intervals shaded
  - boxplot grid of feature distributions (Normal vs Fault)
  - per-asset fault counts
  - class balance

Usage (CLI):
    python src/main.py eda --csv "Dataset/processed/Wind Farm A/combined.csv"

Usage (programmatic):
    from data_pipeline.eda import EDAReport
    EDAReport(csv_path="...", output_dir="...").run()
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, kurtosis, skew

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from training.hyperparameters.feature_screening_defaults import (
    DEFAULT_EDA_MAX_FEATURES,
    DEFAULT_EDA_MAX_MISSING_PCT,
    DEFAULT_EDA_MIN_CORR,
    DEFAULT_EDA_REDUNDANCY_THRESHOLD,
)
from training.hyperparameters.sequence_defaults import (
    PREDICTION_HORIZON_STEPS as DEFAULT_EDA_HORIZON_STEPS,
    SEQUENCE_LENGTH as DEFAULT_EDA_WINDOW_STEPS,
    STRIDE as DEFAULT_EDA_STRIDE_STEPS,
)

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


META_COLUMNS = {
    "time_stamp", "asset_id", "sequence_id", "train_test", "status_type_id", "label",
}
def _resolve_feature_columns(df: pd.DataFrame, feature_file: str | None) -> list[str]:
    """Return numeric feature columns, optionally filtered by an external feature list."""
    if feature_file:
        feat_df = pd.read_csv(feature_file)
        col = "final_feature" if "final_feature" in feat_df.columns else feat_df.columns[0]
        requested = feat_df[col].astype(str).tolist()
        return [c for c in requested if c in df.columns and c not in META_COLUMNS]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in META_COLUMNS]


def _set_boxplot_labels(ax, data, labels_box) -> None:
    """Compatibility wrapper: matplotlib 3.9+ uses `tick_labels`, older uses `labels`."""
    try:
        ax.boxplot(data, tick_labels=labels_box, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels_box, showfliers=False)


def _annotate_bars(ax, bars, total: int) -> None:
    for bar in bars:
        value = int(bar.get_height())
        pct = value / max(total, 1) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _find_event_info_path(csv_path: Path) -> Path | None:
    candidates = [
        csv_path.parent.parent / "raw" / "Wind Farm A" / "event_info.csv",
        csv_path.parent.parent.parent / "raw" / "Wind Farm A" / "event_info.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@dataclass
class EDAReport:
    """Exploratory data analysis runner for combined SCADA CSVs."""

    csv_path: str | Path
    output_dir: str | Path
    feature_file: str | None = None
    max_features: int = DEFAULT_EDA_MAX_FEATURES
    sample_asset: int | None = None
    select_features: bool = False
    min_corr: float = DEFAULT_EDA_MIN_CORR
    max_missing_pct: float = DEFAULT_EDA_MAX_MISSING_PCT
    redundancy_threshold: float = DEFAULT_EDA_REDUNDANCY_THRESHOLD

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        self.output_dir = Path(self.output_dir)
        self.df: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.summary: dict = {}

    def load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Combined CSV not found: {self.csv_path}")
        print(f"Loading combined CSV: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"  Rows: {len(self.df):,}  Columns: {self.df.shape[1]}")

        if "label" not in self.df.columns:
            raise ValueError("Combined CSV must contain a 'label' column.")

        self.feature_cols = _resolve_feature_columns(self.df, self.feature_file)
        if not self.feature_cols:
            raise ValueError("No numeric feature columns found after metadata filtering.")
        print(f"  Features: {len(self.feature_cols)}")

        if self.sample_asset is None and "asset_id" in self.df.columns:
            self.sample_asset = int(self.df["asset_id"].iloc[0])

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Statistical tables
    # ------------------------------------------------------------------

    def compute_missing_values(self) -> pd.DataFrame:
        miss_count = self.df.isna().sum()
        miss_pct = (miss_count / len(self.df) * 100).round(3)
        out = pd.DataFrame({
            "column": miss_count.index,
            "missing_count": miss_count.values.astype(int),
            "missing_pct": miss_pct.values,
        }).sort_values("missing_count", ascending=False).reset_index(drop=True)
        out.to_csv(self.output_dir / "missing_values.csv", index=False)
        return out

    def compute_feature_statistics(self) -> pd.DataFrame:
        rows = []
        for col in self.feature_cols:
            values = self.df[col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                rows.append({
                    "feature": col, "n": 0,
                    "mean": None, "std": None, "min": None, "max": None, "median": None,
                    "skewness": None, "kurtosis": None,
                    "ks_pvalue": None, "normal": None,
                })
                continue

            mean_v = float(np.mean(values))
            std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            min_v = float(np.min(values))
            max_v = float(np.max(values))
            median_v = float(np.median(values))

            if len(values) < 8:
                skew_v = kurt_v = ks_p = None
            elif std_v == 0:
                skew_v, kurt_v, ks_p = 0.0, 0.0, None
            else:
                try:
                    skew_v = float(skew(values))
                except Exception:
                    skew_v = None
                try:
                    kurt_v = float(kurtosis(values, fisher=True))
                except Exception:
                    kurt_v = None
                try:
                    _, p = kstest(values, "norm", args=(mean_v, std_v))
                    ks_p = float(p)
                except Exception:
                    ks_p = None

            rows.append({
                "feature": col,
                "n": int(len(values)),
                "mean": mean_v,
                "std": std_v,
                "min": min_v,
                "max": max_v,
                "median": median_v,
                "skewness": skew_v,
                "kurtosis": kurt_v,
                "ks_pvalue": ks_p,
                "normal": bool(ks_p > 0.05) if ks_p is not None else None,
            })

        out = pd.DataFrame(rows)
        out["abs_skewness"] = out["skewness"].abs()
        out = out.sort_values("abs_skewness", ascending=False, na_position="last").drop(columns=["abs_skewness"]).reset_index(drop=True)
        out.to_csv(self.output_dir / "feature_statistics.csv", index=False)
        return out

    def compute_correlation_with_label(self) -> pd.DataFrame:
        if self.df["label"].isna().any():
            raise ValueError("'label' column contains NaN values — clean before running EDA.")
        labels = self.df["label"].astype(int)
        if labels.nunique() < 2:
            print("  [WARN] label column has only one unique value — correlations will be NaN.")

        rows = []
        for col in self.feature_cols:
            x = self.df[col]
            valid = x.notna()
            x_valid = x[valid]
            y_valid = labels[valid]
            if len(x_valid) < 8 or x_valid.nunique() < 2 or y_valid.nunique() < 2:
                rows.append({"feature": col, "spearman_corr": None, "n": int(len(x_valid))})
                continue
            corr = x_valid.corr(y_valid, method="spearman")
            rows.append({
                "feature": col,
                "spearman_corr": float(corr) if pd.notna(corr) else None,
                "n": int(len(x_valid)),
            })

        out = pd.DataFrame(rows)
        out["abs_corr"] = out["spearman_corr"].abs()
        out = out.sort_values("abs_corr", ascending=False, na_position="last").reset_index(drop=True)
        out.to_csv(self.output_dir / "correlation_with_label.csv", index=False)
        return out

    def compute_per_asset_summary(self) -> pd.DataFrame:
        if "asset_id" not in self.df.columns:
            return pd.DataFrame()
        agg_kwargs = {
            "total_rows": ("label", "size"),
            "fault_rows": ("label", "sum"),
        }
        if "sequence_id" in self.df.columns:
            agg_kwargs["n_events"] = ("sequence_id", "nunique")
        grouped = self.df.groupby("asset_id").agg(**agg_kwargs).reset_index()
        grouped["fault_pct"] = (grouped["fault_rows"] / grouped["total_rows"] * 100).round(3)
        grouped.to_csv(self.output_dir / "per_asset_summary.csv", index=False)
        return grouped

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_missing_values_bar(self, missing_df: pd.DataFrame) -> None:
        top = missing_df[missing_df["missing_count"] > 0].head(self.max_features)
        if top.empty:
            print("  No missing values — skipping missing_values_bar.png")
            return
        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.25)))
        ax.barh(top["column"][::-1], top["missing_pct"][::-1], color="#E15759")
        ax.set_xlabel("Missing %")
        ax.set_title("Missing values by column (top by count)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "missing_values_bar.png", dpi=160)
        plt.close(fig)

    def plot_correlation_matrix(self, corr_df: pd.DataFrame) -> None:
        by_corr = corr_df.dropna(subset=["abs_corr"]).head(self.max_features)["feature"].tolist()
        ranked = len(by_corr) >= 2
        top_features = by_corr if ranked else self.feature_cols[: self.max_features]
        if len(top_features) < 2:
            print("  Not enough features for correlation matrix.")
            return
        sub = self.df[top_features]
        corr_matrix = sub.corr(method="spearman")

        side = max(6, min(20, 0.45 * len(top_features) + 4))
        fig, ax = plt.subplots(figsize=(side, side - 1))
        if _HAS_SEABORN:
            sns.heatmap(
                corr_matrix, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=False, square=True, cbar_kws={"shrink": 0.8},
            )
        else:
            im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(top_features)))
            ax.set_xticklabels(top_features, rotation=90)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            fig.colorbar(im, ax=ax, shrink=0.8)
        title_suffix = "by |corr with label|" if ranked else "(all features)"
        ax.set_title(f"Spearman correlation — top {len(top_features)} features {title_suffix}")
        fig.tight_layout()
        fig.savefig(self.output_dir / "correlation_matrix.png", dpi=160)
        plt.close(fig)

    def plot_feature_distributions(self, corr_df: pd.DataFrame) -> None:
        top_features = corr_df.dropna(subset=["abs_corr"]).head(self.max_features)["feature"].tolist()
        if not top_features:
            top_features = self.feature_cols[: self.max_features]
        if not top_features:
            return

        n = len(top_features)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4))
        axes = np.atleast_2d(axes).flatten()

        sample_n = min(50_000, len(self.df))
        sample_df = self.df.sample(n=sample_n, random_state=0) if sample_n < len(self.df) else self.df

        for i, col in enumerate(top_features):
            ax = axes[i]
            normal = sample_df.loc[sample_df["label"] == 0, col].dropna().values
            fault = sample_df.loc[sample_df["label"] == 1, col].dropna().values
            data = [normal, fault] if len(fault) else [normal]
            labels_box = ["Normal", "Fault"] if len(fault) else ["Normal"]
            _set_boxplot_labels(ax, data, labels_box)
            ax.set_title(col, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.25)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Feature distributions: Normal vs Fault rows (sample)", fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / "feature_distributions.png", dpi=160)
        plt.close(fig)

    def _shade_fault_intervals(self, ax, x, labels) -> None:
        """Shade contiguous fault regions on a time-axis plot."""
        labels = np.asarray(labels)
        if labels.sum() == 0:
            return
        in_fault = False
        start_idx = 0
        any_shaded = False
        for i, lab in enumerate(labels):
            if lab == 1 and not in_fault:
                start_idx = i
                in_fault = True
            elif lab == 0 and in_fault:
                xs = x.iloc[start_idx] if hasattr(x, "iloc") else x[start_idx]
                xe = x.iloc[i] if hasattr(x, "iloc") else x[i]
                ax.axvspan(xs, xe, color="red", alpha=0.2,
                           label="Fault interval" if not any_shaded else None)
                any_shaded = True
                in_fault = False
        if in_fault:
            xs = x.iloc[start_idx] if hasattr(x, "iloc") else x[start_idx]
            xe = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
            ax.axvspan(xs, xe, color="red", alpha=0.2,
                       label="Fault interval" if not any_shaded else None)

    def plot_time_series_overview(self) -> None:
        if "asset_id" not in self.df.columns:
            return
        asset_df = self.df[self.df["asset_id"] == self.sample_asset]
        if asset_df.empty:
            return

        if "sequence_id" in asset_df.columns:
            counts = asset_df.groupby("sequence_id")["label"].agg(["sum", "size"])
            fault_events = counts[counts["sum"] > 0]
            if not fault_events.empty:
                chosen_event = int(fault_events["size"].idxmax())
            else:
                chosen_event = int(counts["size"].idxmax())
            event_df = asset_df[asset_df["sequence_id"] == chosen_event].copy()
            title = f"Asset {self.sample_asset} — Event {chosen_event}"
        else:
            event_df = asset_df.copy()
            title = f"Asset {self.sample_asset}"

        priority = ["active_power", "power", "wind_speed", "rotor_speed", "generator"]
        chosen_feature = None
        for kw in priority:
            for col in self.feature_cols:
                if kw in col.lower():
                    chosen_feature = col
                    break
            if chosen_feature:
                break
        if chosen_feature is None:
            chosen_feature = self.feature_cols[0]

        if "time_stamp" in event_df.columns:
            event_df["time_stamp"] = pd.to_datetime(event_df["time_stamp"], errors="coerce")
            event_df = event_df.sort_values("time_stamp").reset_index(drop=True)
            x = event_df["time_stamp"]
        else:
            event_df = event_df.reset_index(drop=True)
            x = pd.Series(np.arange(len(event_df)))

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(x, event_df[chosen_feature], color="#1f77b4", linewidth=1, alpha=0.9,
                label=chosen_feature)
        self._shade_fault_intervals(ax, x, event_df["label"].to_numpy())
        ax.set_xlabel("Time")
        ax.set_ylabel(chosen_feature)
        ax.set_title(f"{title} — {chosen_feature}")
        ax.grid(alpha=0.25)
        ax.legend()
        if "time_stamp" in event_df.columns:
            fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / "time_series_overview.png", dpi=160)
        plt.close(fig)

    def plot_label_balance(self) -> None:
        row_counts = self.df["label"].astype(int).value_counts().reindex([0, 1], fill_value=0)

        window_counts = pd.Series({0: 0, 1: 0}, dtype=int)
        if {"asset_id", "sequence_id", "time_stamp"}.issubset(self.df.columns):
            required_steps = DEFAULT_EDA_WINDOW_STEPS + DEFAULT_EDA_HORIZON_STEPS
            for _, group in self.df.groupby(["asset_id", "sequence_id"], sort=False):
                group = group.sort_values("time_stamp", kind="mergesort")
                labels = group["label"].to_numpy(dtype=np.int8)
                max_start = len(labels) - required_steps + 1
                if max_start <= 0:
                    continue
                for start in range(0, max_start, DEFAULT_EDA_STRIDE_STEPS):
                    end = start + DEFAULT_EDA_WINDOW_STEPS
                    horizon_end = end + DEFAULT_EDA_HORIZON_STEPS
                    target = int(labels[end:horizon_end].max())
                    window_counts.loc[target] += 1

        event_info_path = _find_event_info_path(self.csv_path)
        event_fault_counts = None
        if event_info_path is not None:
            event_info = pd.read_csv(event_info_path, sep=None, engine="python")
            if {"event_label", "event_description"}.issubset(event_info.columns):
                fault_events = event_info[
                    event_info["event_label"].astype(str).str.lower().eq("anomaly")
                ]
                event_fault_counts = (
                    fault_events["event_description"]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .sort_values(ascending=False)
                )

        if event_fault_counts is not None and not event_fault_counts.empty:
            event_counts = event_fault_counts
        elif {"sequence_id", "label"}.issubset(self.df.columns):
            group_cols = ["sequence_id"]
            if "asset_id" in self.df.columns:
                group_cols = ["asset_id", "sequence_id"]
            event_labels = (
                self.df.groupby(group_cols, sort=False)["label"]
                .max()
                .astype(int)
            )
            event_counts = event_labels.value_counts().reindex([0, 1], fill_value=0)
        else:
            event_counts = pd.Series({0: 0, 1: 0}, dtype=int)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        panels = [
            (
                axes[0],
                row_counts,
                "Row-level label balance",
                "Number of 10-minute SCADA samples",
            ),
            (
                axes[1],
                window_counts,
                "Window-level label balance",
                "Number of sliding windows",
            ),
            (
                axes[2],
                event_counts,
                "Event-level fault distribution",
                "Number of fault events",
            ),
        ]

        for panel_idx, (ax, counts, title, ylabel) in enumerate(panels):
            if panel_idx == 2 and event_fault_counts is not None and not event_fault_counts.empty:
                labels = counts.index.astype(str).tolist()
                values = [int(v) for v in counts.tolist()]
                bars = ax.bar(labels, values, color="#E15759")
                ax.tick_params(axis="x", rotation=20, labelsize=8)
            else:
                labels = ["Normal", "Fault"]
                values = [int(counts.loc[0]), int(counts.loc[1])]
                bars = ax.bar(labels, values, color=["#4E79A7", "#E15759"])
            _annotate_bars(ax, bars, sum(values))
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
            ax.margins(y=0.18)
            ax.ticklabel_format(axis="y", style="plain")

        fig.suptitle("Label balance across rows, sliding windows, and events", fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / "label_balance.png", dpi=160)
        plt.close(fig)

    def plot_per_asset_summary(self, summary_df: pd.DataFrame) -> None:
        if summary_df.empty:
            return
        df = summary_df.sort_values("asset_id").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(df) + 2), 5))
        x = np.arange(len(df))
        normal_rows = df["total_rows"] - df["fault_rows"]
        ax.bar(x, normal_rows, color="#4E79A7", label="Normal")
        ax.bar(x, df["fault_rows"], bottom=normal_rows, color="#E15759", label="Fault")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(a)) for a in df["asset_id"]])
        ax.set_xlabel("Asset ID")
        ax.set_ylabel("Row count")
        ax.set_title("Rows per asset (Normal stacked with Fault)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "per_asset_summary.png", dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Feature selection (EDA-driven)
    # ------------------------------------------------------------------

    def run_feature_selection(
        self,
        missing_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        corr_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select features from the full EDA statistics using four sequential filters:
          1. Drop constant features (std == 0)
          2. Drop features with missing_pct > max_missing_pct
          3. Drop features with |Spearman rho| < min_corr
          4. Remove redundant pairs: among features with |inter-corr| > redundancy_threshold,
             keep the one with higher |label correlation|

        Outputs:
          - feature_selection_audit.csv  — every feature with kept flag and reason
          - eda_selected_features.csv    — kept feature names only

        Returns:
            audit DataFrame (feature, kept, reason)
        """
        missing_lookup = dict(zip(missing_df["column"], missing_df["missing_pct"]))
        std_lookup = dict(zip(stats_df["feature"], stats_df["std"]))
        corr_lookup = {
            row["feature"]: abs(row["spearman_corr"])
            for _, row in corr_df.iterrows()
            if pd.notna(row["spearman_corr"])
        }

        audit: dict[str, dict] = {f: {"kept": True, "reason": "passed all criteria"}
                                   for f in self.feature_cols}

        for feat in self.feature_cols:
            std_v = std_lookup.get(feat)
            if std_v is not None and std_v == 0:
                audit[feat] = {"kept": False, "reason": "constant feature (std=0)"}

        for feat in self.feature_cols:
            if not audit[feat]["kept"]:
                continue
            miss_pct = missing_lookup.get(feat, 0.0)
            if miss_pct > self.max_missing_pct:
                audit[feat] = {"kept": False, "reason": f"too many missing: {miss_pct:.1f}%"}

        for feat in self.feature_cols:
            if not audit[feat]["kept"]:
                continue
            abs_corr = corr_lookup.get(feat)
            if abs_corr is None:
                audit[feat] = {"kept": False, "reason": "correlation could not be computed"}
            elif abs_corr < self.min_corr:
                audit[feat] = {"kept": False, "reason": f"low label correlation: {abs_corr:.4f}"}

        kept = [f for f in self.feature_cols if audit[f]["kept"]]
        kept_sorted = sorted(kept, key=lambda f: corr_lookup.get(f, 0.0), reverse=True)

        if len(kept_sorted) >= 2:
            try:
                inter_corr = self.df[kept_sorted].corr(method="spearman").abs()
                redundant: set[str] = set()
                for feat in kept_sorted:
                    if feat in redundant:
                        continue
                    for other in kept_sorted:
                        if other == feat or other in redundant:
                            continue
                        if inter_corr.loc[feat, other] > self.redundancy_threshold:
                            audit[other] = {
                                "kept": False,
                                "reason": (
                                    f"redundant with {feat} "
                                    f"(inter-corr={inter_corr.loc[feat, other]:.3f})"
                                ),
                            }
                            redundant.add(other)
            except Exception as exc:
                print(f"  [WARN] Redundancy check failed: {exc}")

        audit_df = pd.DataFrame(
            [{"feature": f, "kept": d["kept"], "reason": d["reason"]}
             for f, d in audit.items()]
        )
        audit_df.to_csv(self.output_dir / "feature_selection_audit.csv", index=False)

        selected = audit_df[audit_df["kept"]]["feature"].tolist()
        pd.DataFrame({"feature": selected}).to_csv(
            self.output_dir / "eda_selected_features.csv", index=False
        )

        print(
            f"\n  Feature selection: {len(selected)}/{len(self.feature_cols)} features kept"
        )
        print(f"    min_corr={self.min_corr}, max_missing={self.max_missing_pct}%, "
              f"redundancy_threshold={self.redundancy_threshold}")
        drop_reasons = audit_df[~audit_df["kept"]]["reason"].str.split(":").str[0]
        for reason, count in drop_reasons.value_counts().items():
            print(f"    Dropped — {reason}: {count}")
        if selected:
            print(f"    Selected: {', '.join(selected)}")

        return audit_df

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------

    def write_summary_json(
        self,
        missing_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        corr_df: pd.DataFrame,
        per_asset_df: pd.DataFrame,
    ) -> None:
        n_rows = int(len(self.df))
        n_fault = int(self.df["label"].sum())
        n_train = (
            int((self.df["train_test"] == "train").sum())
            if "train_test" in self.df.columns
            else None
        )

        top_missing = (
            missing_df[missing_df["missing_count"] > 0]
            .head(5)[["column", "missing_count", "missing_pct"]]
        )
        top_skew = stats_df.dropna(subset=["skewness"]).head(5)[["feature", "skewness", "kurtosis"]]
        top_corr = corr_df.dropna(subset=["abs_corr"]).head(5)[["feature", "spearman_corr"]]
        non_normal_count = int((stats_df["normal"] == False).sum())  # noqa: E712

        summary = {
            "csv_path": str(self.csv_path),
            "output_dir": str(self.output_dir),
            "n_rows": n_rows,
            "n_features": len(self.feature_cols),
            "n_assets": (
                int(self.df["asset_id"].nunique())
                if "asset_id" in self.df.columns
                else None
            ),
            "fault_row_count": n_fault,
            "fault_row_pct": round(n_fault / max(n_rows, 1) * 100, 4),
            "train_row_count": n_train,
            "prediction_row_count": (n_rows - n_train) if n_train is not None else None,
            "non_normal_features_count": non_normal_count,
            "top_missing_columns": top_missing.to_dict(orient="records"),
            "top_skewed_features": top_skew.to_dict(orient="records"),
            "top_correlated_features": top_corr.to_dict(orient="records"),
            "feature_columns": self.feature_cols,
        }
        with (self.output_dir / "eda_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        self.summary = summary

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> dict:
        self.load()
        print(f"\nOutput dir: {self.output_dir}\n")

        print("Computing missing values...")
        missing_df = self.compute_missing_values()

        print("Computing feature statistics (skewness, kurtosis, KS-test)...")
        stats_df = self.compute_feature_statistics()

        print("Computing Spearman correlation with label...")
        corr_df = self.compute_correlation_with_label()

        print("Computing per-asset summary...")
        per_asset_df = self.compute_per_asset_summary()

        print("Plotting missing values bar...")
        self.plot_missing_values_bar(missing_df)

        print("Plotting correlation matrix...")
        self.plot_correlation_matrix(corr_df)

        print("Plotting feature distributions...")
        self.plot_feature_distributions(corr_df)

        print("Plotting time series overview...")
        self.plot_time_series_overview()

        print("Plotting label balance...")
        self.plot_label_balance()

        print("Plotting per-asset summary...")
        self.plot_per_asset_summary(per_asset_df)

        if self.select_features:
            print("Running EDA-based feature selection...")
            self.run_feature_selection(missing_df, stats_df, corr_df)

        print("Writing summary JSON...")
        self.write_summary_json(missing_df, stats_df, corr_df, per_asset_df)

        print(f"\nEDA complete. Outputs saved to: {self.output_dir}")
        return self.summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA on a combined SCADA CSV.")
    parser.add_argument("--csv", required=True, type=str, metavar="PATH")
    parser.add_argument("--output-dir", type=str, default=None, metavar="DIR")
    parser.add_argument("--feature-file", type=str, default=None, metavar="PATH")
    parser.add_argument("--max-features", type=int, default=DEFAULT_EDA_MAX_FEATURES, metavar="N")
    parser.add_argument("--sample-asset", type=int, default=None, metavar="ID")
    parser.add_argument("--select-features", action="store_true",
                        help="Run EDA-based feature selection and output selected features.")
    parser.add_argument("--min-corr", type=float, default=DEFAULT_EDA_MIN_CORR, metavar="F",
                        help="Minimum |Spearman rho| with label to keep a feature (default 0.02).")
    parser.add_argument("--max-missing-pct", type=float, default=DEFAULT_EDA_MAX_MISSING_PCT, metavar="F",
                        help="Drop features with missing %% above this threshold (default 80).")
    parser.add_argument("--redundancy-threshold", type=float, default=DEFAULT_EDA_REDUNDANCY_THRESHOLD, metavar="F",
                        help="Inter-feature |Spearman rho| above which weaker feature is dropped (default 0.90).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from config import RESULTS_DIR
        farm_name = csv_path.parent.name or "default"
        output_dir = Path(RESULTS_DIR) / "eda" / farm_name

    EDAReport(
        csv_path=csv_path,
        output_dir=output_dir,
        feature_file=args.feature_file,
        max_features=args.max_features,
        sample_asset=args.sample_asset,
        select_features=args.select_features,
        min_corr=args.min_corr,
        max_missing_pct=args.max_missing_pct,
        redundancy_threshold=args.redundancy_threshold,
    ).run()


if __name__ == "__main__":
    main()
