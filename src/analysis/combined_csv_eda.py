"""Paper-style EDA for the combined SCADA CSV.

The script is intentionally artifact-oriented: it reads one combined CSV,
computes the tables needed for the report, and writes plots that can be reused
without opening a notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from config import PROCESSED_DATA_DIR, RESULTS_DIR
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_DIR = str(REPO_ROOT / "Dataset" / "processed")
    RESULTS_DIR = str(REPO_ROOT / "results")

from training.hyperparameters.feature_screening_defaults import (
    DEFAULT_ALPHA,
    DEFAULT_COMBINED_CSV_SEP,
    DEFAULT_TARGET_COL,
    DROP_COLUMNS,
)
from training.hyperparameters.sequence_defaults import RANDOM_SEED


DEFAULT_CSV = Path(PROCESSED_DATA_DIR) / "combined_dataset.csv"
DEFAULT_OUTPUT_DIR = Path(RESULTS_DIR) / "eda_combined_csv"
DEFAULT_SELECTED_FEATURE_FILE = (
    Path(RESULTS_DIR) / "feature_screening_combined_csv" / "final_selected_features.csv"
)
DEFAULT_METADATA_COLUMNS = set(DROP_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EDA tables and plots for a combined SCADA CSV.",
    )
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sep", type=str, default=DEFAULT_COMBINED_CSV_SEP)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--time-col", type=str, default="time_stamp")
    parser.add_argument("--selected-feature-file", type=str, default=str(DEFAULT_SELECTED_FEATURE_FILE))
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--plot-rows", type=int, default=5_000)
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def read_csv(csv_path: Path, sep: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Combined CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep=None, engine="python", low_memory=False)
    return df


def numeric_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        col
        for col in numeric_cols
        if col not in DEFAULT_METADATA_COLUMNS and col != target_col
    ]


def sample_df(df: pd.DataFrame, sample_rows: int, random_state: int) -> pd.DataFrame:
    if sample_rows <= 0 or len(df) <= sample_rows:
        return df
    return df.sample(n=sample_rows, random_state=random_state)


def selected_features_from_file(path: Path, available: Iterable[str]) -> list[str]:
    available_set = set(available)
    if not path.exists():
        return []

    selected = pd.read_csv(path)
    if selected.empty:
        return []
    feature_col = "final_feature" if "final_feature" in selected.columns else selected.columns[0]
    return [
        str(feature).strip()
        for feature in selected[feature_col].dropna()
        if str(feature).strip() in available_set
    ]


def save_dataset_profile(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    output_dir: Path,
) -> dict:
    target = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series(dtype=float)
    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_features": int(len(feature_cols)),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_col": target_col if target_col in df.columns else None,
        "target_non_null": int(target.notna().sum()),
        "target_positive_count": int((target == 1).sum()) if not target.empty else None,
        "target_positive_ratio": float((target == 1).mean()) if not target.empty else None,
    }
    pd.DataFrame([profile]).to_csv(output_dir / "dataset_profile.csv", index=False)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return profile


def save_missing_values(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100.0),
            "non_missing_count": df.notna().sum().values,
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])
    missing.to_csv(output_dir / "missing_values.csv", index=False)
    return missing


def save_numeric_summary(df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    summary = df[feature_cols].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    summary.insert(0, "feature", summary.index)
    summary.reset_index(drop=True).to_csv(output_dir / "numeric_summary.csv", index=False)
    return summary


def save_distribution_tests(
    df_sample: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    alpha: float,
) -> pd.DataFrame:
    records = []
    for col in feature_cols:
        series = pd.to_numeric(df_sample[col], errors="coerce").dropna()
        n = int(len(series))
        if n < 5 or series.nunique(dropna=True) < 2:
            records.append(
                {
                    "feature": col,
                    "n": n,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                    "ks_stat": np.nan,
                    "ks_p_value": np.nan,
                    "normally_distributed": False,
                }
            )
            continue

        z = (series - series.mean()) / series.std(ddof=0)
        z = z.replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 5:
            ks_stat, ks_p = np.nan, np.nan
        else:
            ks_stat, ks_p = stats.kstest(z.to_numpy(dtype=float), "norm")

        records.append(
            {
                "feature": col,
                "n": n,
                "skewness": float(stats.skew(series, nan_policy="omit")),
                "kurtosis": float(stats.kurtosis(series, nan_policy="omit")),
                "ks_stat": float(ks_stat) if not np.isnan(ks_stat) else np.nan,
                "ks_p_value": float(ks_p) if not np.isnan(ks_p) else np.nan,
                "normally_distributed": bool(ks_p >= alpha) if not np.isnan(ks_p) else False,
            }
        )

    tests = pd.DataFrame(records).sort_values("feature")
    tests.to_csv(output_dir / "skew_kurtosis_ks.csv", index=False)
    return tests


def save_spearman_outputs(
    df_sample: pd.DataFrame,
    feature_cols: list[str],
    selected_features: list[str],
    target_col: str,
    output_dir: Path,
    top_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_cols = feature_cols + ([target_col] if target_col in df_sample.columns else [])
    corr_input = df_sample[corr_cols].apply(pd.to_numeric, errors="coerce")
    spearman_full = corr_input.corr(method="spearman")
    spearman_full.to_csv(output_dir / "spearman_full.csv", index_label="feature")

    target_corr = pd.DataFrame()
    if target_col in spearman_full.columns:
        target_corr = (
            spearman_full[target_col]
            .drop(labels=[target_col], errors="ignore")
            .rename("spearman_rho_to_target")
            .to_frame()
        )
        target_corr["abs_spearman_rho_to_target"] = target_corr["spearman_rho_to_target"].abs()
        target_corr = target_corr.sort_values("abs_spearman_rho_to_target", ascending=False)
        target_corr.to_csv(output_dir / "spearman_target.csv", index_label="feature")

    selected = selected_features
    if not selected:
        selected = target_corr.head(top_features).index.tolist() if not target_corr.empty else feature_cols[:top_features]

    selected_corr_cols = selected + ([target_col] if target_col in df_sample.columns else [])
    spearman_selected = corr_input[selected_corr_cols].corr(method="spearman")
    spearman_selected.to_csv(output_dir / "spearman_selected.csv", index_label="feature")
    return spearman_full, spearman_selected


def plot_missing_values(missing: pd.DataFrame, output_dir: Path, top_features: int) -> None:
    top = missing[missing["missing_count"] > 0].head(top_features)
    if top.empty:
        top = missing.head(min(top_features, len(missing)))

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(top))))
    ax.barh(top["column"], top["missing_pct"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Column")
    ax.set_title("Missing Values by Column")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "missing_values_bar.png", dpi=180)
    plt.close(fig)


def plot_distribution_boxplots(
    df_sample: pd.DataFrame,
    features: list[str],
    output_dir: Path,
) -> None:
    if not features:
        return
    plot_df = df_sample[features].apply(pd.to_numeric, errors="coerce")
    standardized = (plot_df - plot_df.median()) / plot_df.std(ddof=0).replace(0, np.nan)
    standardized = standardized.replace([np.inf, -np.inf], np.nan)

    fig, ax = plt.subplots(figsize=(max(12, 0.5 * len(features)), 7))
    ax.boxplot(
        [standardized[col].dropna().values for col in features],
        tick_labels=features,
        showfliers=False,
    )
    ax.set_ylabel("Robust standardized value")
    ax.set_title("Feature Distribution Boxplots")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distribution_boxplots.png", dpi=180)
    plt.close(fig)


def plot_spearman_heatmap(corr: pd.DataFrame, output_dir: Path) -> None:
    if corr.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(corr)), max(7, 0.45 * len(corr))))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=75, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Spearman Correlation Heatmap")
    plt.colorbar(image, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output_dir / "spearman_heatmap.png", dpi=180)
    plt.close(fig)


def plot_fault_timeline(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    feature: str | None,
    output_dir: Path,
    plot_rows: int,
) -> None:
    if target_col not in df.columns:
        return

    cols = [target_col]
    if time_col in df.columns:
        cols.append(time_col)
    if feature and feature in df.columns:
        cols.append(feature)

    plot_df = df[cols].copy()
    if time_col in plot_df.columns:
        plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
        plot_df = plot_df.sort_values(time_col)

    if plot_rows > 0 and len(plot_df) > plot_rows:
        step = max(1, len(plot_df) // plot_rows)
        plot_df = plot_df.iloc[::step].head(plot_rows)

    x = plot_df[time_col] if time_col in plot_df.columns else np.arange(len(plot_df))
    target = pd.to_numeric(plot_df[target_col], errors="coerce")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(x, target, color="#D62728", linewidth=1.0, label=target_col)
    ax1.set_ylabel(target_col)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.25)

    if feature and feature in plot_df.columns:
        ax2 = ax1.twinx()
        values = pd.to_numeric(plot_df[feature], errors="coerce")
        values = (values - values.mean()) / values.std(ddof=0)
        ax2.plot(x, values, color="#4C78A8", linewidth=0.8, alpha=0.8, label=feature)
        ax2.set_ylabel(f"{feature} z-score")

    ax1.set_title("Fault Timeline Example")
    fig.tight_layout()
    fig.savefig(output_dir / "fault_timeline_examples.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EDA] Reading: {csv_path}")
    df = read_csv(csv_path, args.sep)
    feature_cols = numeric_feature_columns(df, args.target_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    selected_features = selected_features_from_file(Path(args.selected_feature_file), feature_cols)
    sample = sample_df(df, args.sample_rows, args.random_state)

    profile = save_dataset_profile(df, feature_cols, args.target_col, output_dir)
    missing = save_missing_values(df, output_dir)
    save_numeric_summary(df, feature_cols, output_dir)
    save_distribution_tests(sample, feature_cols, output_dir, args.alpha)
    _, spearman_selected = save_spearman_outputs(
        sample,
        feature_cols,
        selected_features,
        args.target_col,
        output_dir,
        args.top_features,
    )

    plot_features = selected_features[: args.top_features]
    if not plot_features:
        target_corr = pd.read_csv(output_dir / "spearman_target.csv", index_col=0)
        plot_features = target_corr.head(args.top_features).index.tolist()

    plot_missing_values(missing, output_dir, args.top_features)
    plot_distribution_boxplots(sample, plot_features, output_dir)
    plot_spearman_heatmap(spearman_selected, output_dir)
    timeline_feature = plot_features[0] if plot_features else None
    plot_fault_timeline(
        df,
        args.target_col,
        args.time_col,
        timeline_feature,
        output_dir,
        args.plot_rows,
    )

    print("[EDA] Complete")
    print(f"[EDA] Rows: {profile['rows']:,}")
    print(f"[EDA] Numeric features: {profile['numeric_features']}")
    print(f"[EDA] Outputs: {output_dir}")


if __name__ == "__main__":
    run(parse_args())
