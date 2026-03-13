"""
MetricsCalculator — utils.metrics
Shared binary classification metrics used by both tree trainers and evaluators.
Consolidates duplicated evaluate_binary / find_best_threshold helpers.
"""

import numpy as np


class MetricsCalculator:
    """
    Computes binary classification metrics for anomaly detection.

    Works on sample-level probability scores produced by XGBoost /
    Random Forest, as well as the LSTM MAE anomaly signal.
    """

    def evaluate_binary(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """
        Compute classification metrics at a given decision threshold.

        Args:
            y_true: Ground-truth binary labels (0 = normal, 1 = anomaly).
            y_pred_proba: Predicted probabilities or continuous scores in [0, 1].
            threshold: Decision boundary; samples >= threshold → anomaly.

        Returns:
            Dict with keys: accuracy, precision, recall, far, f1,
                            TP, FP, TN, FN, threshold.
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        TP = int(((y_true == 1) & (y_pred == 1)).sum())
        FP = int(((y_true == 0) & (y_pred == 1)).sum())
        TN = int(((y_true == 0) & (y_pred == 0)).sum())
        FN = int(((y_true == 1) & (y_pred == 0)).sum())

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        far  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        acc  = (TP + TN) / max(TP + FP + TN + FN, 1)

        return {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "far":       far,
            "f1":        f1,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "threshold": threshold,
        }

    def find_best_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_steps: int = 200,
    ) -> float:
        """
        Grid-search the threshold that maximises F1 on a validation set.

        Args:
            y_true: Ground-truth binary labels.
            y_proba: Predicted probabilities in [0, 1].
            n_steps: Number of threshold candidates in [0.05, 0.95].

        Returns:
            Best threshold (float).
        """
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, n_steps):
            m = self.evaluate_binary(y_true, y_proba, t)
            if m["f1"] > best_f1:
                best_f1, best_t = m["f1"], float(t)
        return best_t

    def print_metrics(self, metrics: dict, label: str = "") -> None:
        """Pretty-print a metrics dict returned by evaluate_binary."""
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}Accuracy={metrics['accuracy']:.2%}  "
            f"Precision={metrics['precision']:.2%}  "
            f"Recall={metrics['recall']:.2%}  "
            f"FAR={metrics['far']:.2%}  "
            f"F1={metrics['f1']:.4f}"
        )
