"""
sequence_experiments.py - training.sequence_experiments
High-level experiment runner for supervised sequence classifiers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.hyperparameters.classifier_defaults import (
    CLASSIFIER_THRESHOLD_GRID_POINTS,
    CLASSIFIER_THRESHOLD_GRID_START,
    CLASSIFIER_THRESHOLD_GRID_STOP,
)
from training.sequence_metrics import (
    aggregate_event_scores,
    compute_class_weights,
    evaluate_at_threshold,
    pick_best_threshold,
    sweep_thresholds,
)
from training.sequence_models import build_classifier_model, classifier_callbacks
from training.sequence_plots import (
    build_prediction_frame,
    save_confusion_matrix_plot,
    save_history_csv_and_plot,
    save_metrics_bar_plot,
    save_pr_curve_plot,
    save_roc_curve_plot,
    save_threshold_csv,
    save_threshold_plot,
)
from training.sequence_utils import (
    cleanup_tf,
    history_to_frame,
    load_best_model,
    load_json,
    save_json,
    save_model_summary,
    set_random_seed,
)


def run_classifier_experiment(
    model_name: str,
    bundle: dict,
    output_dir: Path,
    random_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float | None,
    l2_strength: float,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float,
    overwrite: bool,
    save_predictions: bool,
) -> dict:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        loaded = load_json(metrics_path)
        return {
            "summary": loaded["summary"],
            "metrics": loaded,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_tf()
    set_random_seed(random_seed)

    X_train = bundle["X_train"]
    y_train = np.asarray(bundle["y_train"]).astype(np.int8)
    X_val = bundle["X_val"]
    y_val = np.asarray(bundle["y_val"]).astype(np.int8)
    X_test = bundle["X_test"]
    y_test = np.asarray(bundle["y_test"]).astype(np.int8)

    input_shape = (int(X_train.shape[1]), int(X_train.shape[2]))
    model = build_classifier_model(
        model_name,
        input_shape,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_strength=l2_strength,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    model.summary()
    save_model_summary(model, output_dir / "model_summary.txt")
    model_path = output_dir / "model.keras"

    class_weight = None if loss_name == "focal" else compute_class_weights(y_train)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=classifier_callbacks(model_path),
        verbose=1,
    )

    best_model = load_best_model(model_path)
    history_df = history_to_frame(history)
    save_history_csv_and_plot(
        history_df,
        output_dir / "history.csv",
        output_dir / "loss_history.png",
        f"{model_name.upper()} Train / Val Loss",
    )

    val_scores = best_model.predict(X_val, batch_size=batch_size, verbose=0).reshape(-1)
    test_scores = best_model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)

    threshold_grid = np.linspace(
        CLASSIFIER_THRESHOLD_GRID_START,
        CLASSIFIER_THRESHOLD_GRID_STOP,
        CLASSIFIER_THRESHOLD_GRID_POINTS,
    )
    val_meta = bundle["val_meta"]
    sweep_df = sweep_thresholds(y_val, val_scores, threshold_grid)
    threshold_source = "validation_f1_sweep"
    best_row = pick_best_threshold(sweep_df)
    best_threshold = float(best_row["threshold"])

    save_threshold_csv(sweep_df, output_dir / "threshold_sweep_val.csv")
    save_threshold_plot(
        sweep_df,
        best_threshold,
        output_dir / "threshold_sweep_val.png",
        f"{model_name.upper()} Validation Threshold Sweep",
    )

    val_metrics = evaluate_at_threshold(y_val, val_scores, best_threshold)
    test_metrics = evaluate_at_threshold(y_test, test_scores, best_threshold)

    test_meta = bundle["test_meta"]
    event_test_metrics: dict | None = None
    event_test_df = None
    if "sequence_id" in test_meta.columns and "asset_id" in test_meta.columns:
        event_test_df = aggregate_event_scores(test_meta, test_scores)
        event_test_metrics = evaluate_at_threshold(
            event_test_df["true_label"].to_numpy(dtype=np.int8),
            event_test_df["score"].to_numpy(dtype=np.float32),
            best_threshold,
        )

    if save_predictions:
        build_prediction_frame(val_meta, val_scores, best_threshold).to_csv(
            output_dir / "val_predictions.csv", index=False
        )
        build_prediction_frame(test_meta, test_scores, best_threshold).to_csv(
            output_dir / "test_predictions.csv", index=False
        )
        if event_test_df is not None:
            event_test_df["predicted_label"] = (
                event_test_df["score"] >= best_threshold
            ).astype(np.int8)
            event_test_df.to_csv(output_dir / "test_event_predictions.csv", index=False)

    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        output_dir / "confusion_matrix_test.png",
        f"{model_name.upper()} Test Confusion Matrix",
    )
    save_metrics_bar_plot(
        test_metrics,
        output_dir / "test_metrics_bar.png",
        f"{model_name.upper()} Test Metrics",
    )
    save_pr_curve_plot(
        y_test, test_scores, output_dir / "pr_curve_test.png",
        f"{model_name.upper()} Test PR Curve",
    )
    save_roc_curve_plot(
        y_test, test_scores, output_dir / "roc_curve_test.png",
        f"{model_name.upper()} Test ROC Curve",
    )

    summary = {
        "model_name": model_name,
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
    }
    if event_test_metrics is not None:
        summary["event_precision"] = event_test_metrics["precision"]
        summary["event_recall"] = event_test_metrics["recall"]
        summary["event_f1"] = event_test_metrics["f1"]
        summary["event_count"] = event_test_metrics["support"]

    payload = {
        "model_name": model_name,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "hyperparameters": {
            "learning_rate": float(learning_rate),
            "dropout_rate": None if dropout_rate is None else float(dropout_rate),
            "l2_strength": float(l2_strength),
            "loss_name": loss_name,
            "focal_gamma": float(focal_gamma),
            "focal_alpha": float(focal_alpha),
        },
        "selected_threshold": best_threshold,
        "threshold_source": threshold_source,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "event_test_metrics": event_test_metrics,
        "summary": summary,
    }
    save_json(metrics_path, payload)

    del model, best_model, history
    cleanup_tf()

    return {
        "summary": summary,
        "metrics": payload,
    }
