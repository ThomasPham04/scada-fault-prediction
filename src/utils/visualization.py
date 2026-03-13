"""
Visualizer — utils.visualization
Shared plotting helpers consolidated from train_lstm.py, train_xgboost.py,
train_random_forest.py, and evaluate_lstm.py.
"""

from __future__ import annotations
import os
import numpy as np


class Visualizer:
    """
    Centralises all Matplotlib-based plots used across training and evaluation.

    All methods save figures to disk and close them. They import matplotlib
    locally so that importing this module does not force a display dependency.
    """

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------

    def plot_training_history(self, history, save_path: str) -> None:
        """
        Save Loss (MSE) and MAE training curves for a Keras history object.

        Args:
            history: Keras History object from model.fit().
            save_path: Full path for the output PNG.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, metric, ylabel in zip(
            axes,
            [("loss", "val_loss"), ("mae", "val_mae")],
            ["Loss (MSE)", "MAE"],
        ):
            ax.plot(history.history[metric[0]], label="Train", linewidth=2)
            ax.plot(history.history[metric[1]], label="Val",   linewidth=2)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(
                f"Training and Validation {ylabel}", fontsize=14, fontweight="bold"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    # Error distributions (LSTM)
    # ------------------------------------------------------------------

    def plot_error_distributions(self, errors: dict, save_path: str) -> None:
        """
        Plot MSE and MAE distributions for train and val splits.

        Args:
            errors: Dict with 'train' and 'val' keys from evaluate_train_val().
            save_path: Full path for the output PNG.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for split in ["train", "val"]:
            mse = errors[split]["mse_per_sample"]
            mae = errors[split]["mae_per_sample"]
            axes[0].hist(
                mse, bins=50, alpha=0.6,
                label=f"{split.capitalize()} (μ={mse.mean():.4f})"
            )
            axes[1].hist(
                mae, bins=50, alpha=0.6,
                label=f"{split.capitalize()} (μ={mae.mean():.4f})"
            )
        for ax, xlabel, title in zip(
            axes,
            ["MSE", "MAE"],
            ["MSE Distribution", "MAE Distribution"],
        ):
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Error distributions saved to: {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    # Feature importance (tree models)
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        model,
        output_dir: str,
        title: str = "Feature Importances",
        filename: str = "feature_importance.png",
        top_k: int = 30,
    ) -> None:
        """
        Save a bar chart of the top-k feature importances.

        Args:
            model: Fitted sklearn/XGBoost model with feature_importances_.
            output_dir: Directory to save the PNG.
            title: Chart title.
            filename: Output filename.
            top_k: Number of top features to display.
        """
        try:
            import matplotlib.pyplot as plt
            importance = model.feature_importances_
            k = min(top_k, len(importance))
            top_idx = np.argsort(importance)[::-1][:k]

            plt.figure(figsize=(12, 6))
            plt.bar(range(k), importance[top_idx])
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.title(title)
            plt.tight_layout()
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Feature importance saved to: {path}")
        except Exception as e:
            print(f"[WARN] Could not plot feature importance: {e}")

    # ------------------------------------------------------------------
    # Event-level scatter (LSTM evaluation)
    # ------------------------------------------------------------------

    def plot_event_scores(
        self,
        plot_records: list,
        upper_threshold: float,
        save_path: str | None = None,
    ) -> None:
        """
        Scatter plot of per-event p95 MAE versus global and adaptive thresholds.

        Args:
            plot_records: List of dicts with keys event_id, true_label,
                detected, p95, mean, adaptive_threshold.
            upper_threshold: Global threshold line.
            save_path: If given, save to file instead of showing interactively.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        event_ids = [r["event_id"]           for r in plot_records]
        p95_vals  = [r["p95"]                for r in plot_records]
        adap_vals = [r["adaptive_threshold"] for r in plot_records]
        labels    = [r["true_label"]         for r in plot_records]
        colors    = ["red" if l == "anomaly" else "blue" for l in labels]

        plt.figure(figsize=(14, 5))
        plt.scatter(event_ids, p95_vals, c=colors, s=90, alpha=0.8, label="_nolegend_")
        plt.plot(
            event_ids, adap_vals,
            color="green", linestyle=":", marker="o", linewidth=2,
            label="Adaptive Threshold",
        )
        plt.axhline(
            upper_threshold, color="black", linestyle="--", linewidth=2,
            label="Global Upper Threshold",
        )
        plt.xticks(event_ids)
        plt.xlabel("Event ID")
        plt.ylabel("p95 MAE")
        plt.title("Event-level p95 MAE vs Thresholds")

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label="Anomaly",
                   markerfacecolor="red",  markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Normal",
                   markerfacecolor="blue", markersize=10),
        ]
        plt.legend(
            handles=legend_elements + plt.gca().get_legend_handles_labels()[0],
            loc="best",
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Event score plot saved to: {save_path}")
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Model comparison bar chart
    # ------------------------------------------------------------------

    def compare_models_chart(
        self,
        all_results: dict,
        output_dir: str,
        filename: str = "model_comparison.png",
    ) -> None:
        """
        Bar chart comparing multiple models across key detection metrics.

        Args:
            all_results: Dict of {model_name: metrics_dict}.
            output_dir: Directory to save the PNG.
            filename: Output filename.
        """
        try:
            import matplotlib.pyplot as plt
            display_metrics = ["recall", "precision", "far", "f1"]
            x = np.arange(len(display_metrics))
            n_models = len(all_results)
            width = 0.8 / n_models

            fig, ax = plt.subplots(figsize=(12, 5))
            for i, (model_key, res) in enumerate(all_results.items()):
                vals = []
                for m in display_metrics:
                    v = res.get(m, res.get("false_alarm_rate", 0) if m == "far" else 0)
                    vals.append(float(v) if v is not None else 0.0)
                ax.bar(x + i * width, vals, width, label=model_key, alpha=0.85)

            ax.set_xticks(x + width * (n_models - 1) / 2)
            ax.set_xticklabels(
                ["Detection Rate", "Precision", "False Alarm Rate", "F1"], fontsize=11
            )
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title(
                "Model Comparison — SCADA Fault Detection", fontsize=14, fontweight="bold"
            )
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nComparison chart saved to: {path}")
        except Exception as e:
            print(f"[WARN] Could not generate comparison chart: {e}")
