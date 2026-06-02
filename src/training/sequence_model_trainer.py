"""
Sequence model training orchestration.

Consumes the windowed exports produced by
data_pipeline.preprocessing.combined_sequence_pipeline and trains global
supervised sequence classifiers.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

from config import PROCESSED_DATA_DIR, RESULTS_DIR
from training.hyperparameters.classifier_defaults import (
    CLASSIFIER_LOSSES,
    DEFAULT_CLASSIFIER_BATCH_SIZE,
    DEFAULT_CLASSIFIER_DROPOUT,
    DEFAULT_CLASSIFIER_EPOCHS,
    DEFAULT_CLASSIFIER_FOCAL_ALPHA,
    DEFAULT_CLASSIFIER_FOCAL_GAMMA,
    DEFAULT_CLASSIFIER_L2,
    DEFAULT_CLASSIFIER_LEARNING_RATE,
    DEFAULT_CLASSIFIER_LOSS,
    DEFAULT_CLASSIFIER_MODELS,
    DEFAULT_SEQUENCE_WINDOWS,
)
from training.hyperparameters.sequence_defaults import RANDOM_SEED
from training.sequence_experiments import (
    run_classifier_experiment,
)
from training.sequence_io import (
    load_classifier_bundle,
)
from training.sequence_metrics import (
    classifier_comparison_frame,
)
from training.sequence_utils import (
    cleanup_tf,
    save_json,
    set_random_seed,
)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


@dataclass
class SequenceTrainingConfig:
    """Configuration for training exported sequence datasets."""

    exports_dir: Path = field(
        default_factory=lambda: Path(PROCESSED_DATA_DIR) / "sequence_exports"
    )
    results_dir: Path = field(
        default_factory=lambda: Path(RESULTS_DIR) / "sequence_training_results"
    )
    windows: list[int] = field(default_factory=lambda: list(DEFAULT_SEQUENCE_WINDOWS))
    classifier_models: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSIFIER_MODELS))
    random_seed: int = RANDOM_SEED
    overwrite: bool = False
    save_predictions: bool = True
    classifier_epochs: int = DEFAULT_CLASSIFIER_EPOCHS
    classifier_batch_size: int = DEFAULT_CLASSIFIER_BATCH_SIZE
    classifier_learning_rate: float = DEFAULT_CLASSIFIER_LEARNING_RATE
    classifier_dropout: float | None = DEFAULT_CLASSIFIER_DROPOUT
    classifier_l2: float = DEFAULT_CLASSIFIER_L2
    classifier_loss: str = DEFAULT_CLASSIFIER_LOSS
    classifier_focal_gamma: float = DEFAULT_CLASSIFIER_FOCAL_GAMMA
    classifier_focal_alpha: float = DEFAULT_CLASSIFIER_FOCAL_ALPHA

    def __post_init__(self) -> None:
        self.exports_dir = _as_path(self.exports_dir)
        self.results_dir = _as_path(self.results_dir)
        self.windows = [int(window) for window in self.windows]
        self.classifier_models = [str(model) for model in self.classifier_models]
        self.classifier_learning_rate = float(self.classifier_learning_rate)
        self.classifier_l2 = float(self.classifier_l2)
        self.classifier_loss = str(self.classifier_loss)
        self.classifier_focal_gamma = float(self.classifier_focal_gamma)
        self.classifier_focal_alpha = float(self.classifier_focal_alpha)
        if self.classifier_loss not in CLASSIFIER_LOSSES:
            raise ValueError(f"classifier_loss must be one of {CLASSIFIER_LOSSES}")


class SequenceModelTrainer:
    """Train classifiers from combined sequence exports."""

    def __init__(self, config: SequenceTrainingConfig | None = None) -> None:
        self.config = config or SequenceTrainingConfig()

    def train_window(self, window_hours: int) -> dict | None:
        """Train all configured models for one exported window size."""
        cfg = self.config
        export_window_dir = cfg.exports_dir / f"window_{window_hours}h"
        if not export_window_dir.exists():
            print(f"[SKIP] Export folder not found for {window_hours}h: {export_window_dir}")
            return None

        print("\n" + "=" * 70)
        print(f"Training Window: {window_hours}h")
        print("=" * 70)

        results_window_dir = cfg.results_dir / f"window_{window_hours}h"
        results_window_dir.mkdir(parents=True, exist_ok=True)

        classifier_export_dir = export_window_dir / "classifier"
        classifier_results_dir = results_window_dir / "classifier"
        classifier_results_dir.mkdir(parents=True, exist_ok=True)

        classifier_bundle = load_classifier_bundle(classifier_export_dir)

        classifier_rows = []
        classifier_payload = []

        for model_name in cfg.classifier_models:
            print(f"\n[Classifier] Training {model_name}")
            model_output_dir = classifier_results_dir / model_name
            result = run_classifier_experiment(
                model_name=model_name,
                bundle=classifier_bundle,
                output_dir=model_output_dir,
                random_seed=cfg.random_seed,
                epochs=cfg.classifier_epochs,
                batch_size=cfg.classifier_batch_size,
                learning_rate=cfg.classifier_learning_rate,
                dropout_rate=cfg.classifier_dropout,
                l2_strength=cfg.classifier_l2,
                loss_name=cfg.classifier_loss,
                focal_gamma=cfg.classifier_focal_gamma,
                focal_alpha=cfg.classifier_focal_alpha,
                overwrite=cfg.overwrite,
                save_predictions=cfg.save_predictions,
            )
            classifier_rows.append(result["summary"])
            classifier_payload.append(result["metrics"])

        classifier_comparison = classifier_comparison_frame(classifier_rows)
        classifier_comparison.to_csv(classifier_results_dir / "model_comparison.csv", index=False)

        best_classifier = (
            classifier_comparison.iloc[0].to_dict()
            if not classifier_comparison.empty
            else None
        )

        window_summary = {
            "window_hours": int(window_hours),
            "classifier_models": classifier_payload,
            "best_classifier": best_classifier,
        }
        save_json(results_window_dir / "window_summary.json", window_summary)

        del classifier_bundle
        cleanup_tf()
        gc.collect()

        return window_summary

    def run(self) -> dict:
        """Run sequence training for every configured window."""
        cfg = self.config
        set_random_seed(cfg.random_seed)
        cfg.results_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Sequence Model Training")
        print("=" * 70)
        print(f"Exports folder : {cfg.exports_dir}")
        print(f"Results folder : {cfg.results_dir}")
        print(f"Windows to run : {cfg.windows}")
        print(f"Classifier set : {cfg.classifier_models}")
        print(f"Classifier lr  : {cfg.classifier_learning_rate}")
        print(f"Classifier drop: {cfg.classifier_dropout}")
        print(f"Classifier L2  : {cfg.classifier_l2}")
        print(f"Classifier loss: {cfg.classifier_loss}")
        if cfg.classifier_loss == "focal":
            print(f"Focal alpha   : {cfg.classifier_focal_alpha}")
            print(f"Focal gamma   : {cfg.classifier_focal_gamma}")

        run_summary = {
            "exports_dir": str(cfg.exports_dir),
            "results_dir": str(cfg.results_dir),
            "windows_requested": cfg.windows,
            "windows_completed": [],
        }

        for window_hours in cfg.windows:
            window_summary = self.train_window(window_hours)
            if window_summary is not None:
                run_summary["windows_completed"].append(window_summary)

        save_json(cfg.results_dir / "run_summary.json", run_summary)

        print("\n" + "=" * 70)
        print("Training complete")
        print("=" * 70)
        for window_summary in run_summary["windows_completed"]:
            window_hours = window_summary["window_hours"]
            best_classifier = window_summary.get("best_classifier")
            print(f"Window {window_hours}h:")
            if best_classifier:
                print(
                    f"  Best classifier : {best_classifier['model_name']} "
                    f"(f1={best_classifier['f1']:.4f})"
                )

        print(f"Saved outputs to : {cfg.results_dir}")
        return run_summary
