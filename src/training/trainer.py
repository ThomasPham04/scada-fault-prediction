"""
Trainer — training.trainer
LSTMTrainer and TreeTrainer classes — per-asset training only.

The entry-point scripts (train_lstm.py, train_xgboost.py,
train_random_forest.py) are reduced to thin argument-parsing wrappers that
instantiate these classes.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List, Optional

import numpy as np
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import (
    MODELS_DIR,
    RESULTS_DIR,
    PER_ASSET_PROCESSED_DIR,
    BATCH_SIZE,
    EPOCHS,
    RANDOM_SEED,
    ensure_dirs,
)
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer


# ===========================================================================
# LSTM Trainer
# ===========================================================================

class LSTMTrainer:
    """
    Trains the LSTM Normal Behaviour Model in per-asset mode.

    Args:
        models_dir: Directory where .keras model files are saved.
        results_dir: Root directory for training result plots/logs.
        batch_size: Mini-batch size for model.fit().
        epochs: Maximum number of training epochs.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.models_dir  = models_dir
        self.results_dir = results_dir
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.seed        = seed
        self._vis        = Visualizer()

    def _set_seeds(self) -> None:
        """Ensure deterministic training runs."""
        import tensorflow as tf
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def train(self, model, X_train, y_train, X_val, y_val, model_path: str):
        """
        Run model.fit() with the standard LSTM callback set.

        Args:
            model: Compiled Keras model.
            X_train / y_train: Training sequences and targets.
            X_val / y_val: Validation sequences and targets.
            model_path: Where to checkpoint the best model.

        Returns:
            Keras History object.
        """
        from training.callbacks.early_stopping import get_lstm_callbacks
        print(f"\nTraining LSTM (epochs={self.epochs}, batch={self.batch_size})...")
        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
        return model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=get_lstm_callbacks(model_path),
            verbose=1,
        )

    def run_per_asset(self, asset_filter: Optional[List[int]] = None) -> None:
        """
        Train one LSTM per turbine asset.

        Args:
            asset_filter: Asset IDs to train. None = train all.
        """
        from models.architectures.lstm import LSTMModel

        self._set_seeds()
        asset_dirs = sorted(glob.glob(os.path.join(PER_ASSET_PROCESSED_DIR, "asset_*")))
        if not asset_dirs:
            print(f"\n[ERROR] No per-asset data found at: {PER_ASSET_PROCESSED_DIR}")
            print("  Run first: python -m src.training.scripts.prepare_per_asset")
            return

        if asset_filter is not None:
            filter_set = {str(a) for a in asset_filter}
            asset_dirs = [d for d in asset_dirs
                          if os.path.basename(d).replace("asset_", "") in filter_set]
            if not asset_dirs:
                print(f"[ERROR] No data found for requested assets: {asset_filter}")
                return
            print(f"Training only assets: {asset_filter}")

        results_root = os.path.join(self.results_dir, "per_asset")
        os.makedirs(results_root, exist_ok=True)

        trained = []
        for asset_dir in asset_dirs:
            asset_id = os.path.basename(asset_dir).replace("asset_", "")
            print(f"\n{'='*60}\nAsset {asset_id}\n{'='*60}")

            try:
                X_train = np.load(os.path.join(asset_dir, "X_train.npy"))
                X_val   = np.load(os.path.join(asset_dir, "X_val.npy"))
                y_train = np.load(os.path.join(asset_dir, "y_train.npy"))
                y_val   = np.load(os.path.join(asset_dir, "y_val.npy"))
            except FileNotFoundError:
                print(f"  [SKIP] Missing data files for asset {asset_id}")
                continue

            window_size = X_train.shape[1]
            n_features  = X_train.shape[2]
            print(f"  Train: {X_train.shape} | Val: {X_val.shape}")

            model_path = os.path.join(self.models_dir, f"lstm_asset_{asset_id}.keras")
            model = LSTMModel(
                input_shape=(window_size, n_features), output_dim=n_features
            ).build()
            history = self.train(model, X_train, y_train, X_val, y_val, model_path)

            result_dir = os.path.join(results_root, f"asset_{asset_id}")
            os.makedirs(result_dir, exist_ok=True)
            self._vis.plot_training_history(
                history, os.path.join(result_dir, "training_history.png")
            )
            print(f"  Model saved: {model_path}")
            trained.append(asset_id)

        print(f"\n{'='*60}")
        print(f"Per-asset training complete. Trained assets: {trained}")
        print(f"Next: python -m src.evaluation.evaluate_lstm --per_asset")


# ===========================================================================
# Tree Trainer  (XGBoost & Random Forest share the same logic)
# ===========================================================================

class TreeTrainer:
    """
    Trains XGBoost or Random Forest in per-asset mode.

    Args:
        model_type: 'xgboost' or 'random_forest'.
        models_dir: Directory where .pkl bundles are saved.
        results_dir: Root directory for result JSON and plots.
        seed: Random seed.
    """

    def __init__(
        self,
        model_type: str,
        models_dir: str  = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        seed: int        = RANDOM_SEED,
    ) -> None:
        if model_type not in ("xgboost", "random_forest"):
            raise ValueError(f"model_type must be 'xgboost' or 'random_forest', got '{model_type}'")
        self.model_type  = model_type
        self.models_dir  = models_dir
        self.results_dir = results_dir
        self.seed        = seed
        self._metrics    = MetricsCalculator()
        self._vis        = Visualizer()

    def _build_model(self, **kwargs):
        """Instantiate the configured model type."""
        if self.model_type == "xgboost":
            from models.architectures.xgboost_model import XGBoostModel
            return XGBoostModel(**kwargs).build()
        else:
            from models.architectures.random_forest import RandomForestModel
            return RandomForestModel(**kwargs).build()

    def _fit_xgboost(self, model, X_train, y_train, X_val, y_val, early_stopping_rounds: int = 30):
        """Fit XGBoost with optional early stopping."""
        import xgboost as xgb
        callbacks = []
        if early_stopping_rounds > 0:
            try:
                callbacks.append(xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds, save_best=True,
                    data_name="validation_1", maximize=False,
                ))
            except AttributeError:
                pass
        fit_kwargs = dict(
            X=X_train, y=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )
        try:
            model.fit(callbacks=callbacks, **fit_kwargs)
        except TypeError:
            model.fit(**fit_kwargs)
        return model

    def _fit_rf(self, model, X_train, y_train):
        """Fit Random Forest."""
        print(f"\nFitting Random Forest...")
        model.fit(X_train, y_train)
        return model

    def _fit(self, model, X_train, y_train, X_val, y_val, early_stopping_rounds: int = 30):
        """Dispatch fit to the correct backend."""
        if self.model_type == "xgboost":
            return self._fit_xgboost(model, X_train, y_train, X_val, y_val, early_stopping_rounds)
        return self._fit_rf(model, X_train, y_train)

    # ------------------------------------------------------------------
    # Per-asset run
    # ------------------------------------------------------------------

    def run_per_asset(self, args) -> None:
        """Train one model per turbine asset."""
        from data_pipeline.loaders.tabular_loader import TabularLoader

        asset_dirs = sorted(glob.glob(os.path.join(PER_ASSET_PROCESSED_DIR, "asset_*")))
        if not asset_dirs:
            print("[ERROR] No per-asset data found. Run prepare_per_asset first.")
            return

        assets_filter = getattr(args, "assets", None)
        if assets_filter:
            filter_set = {str(a) for a in assets_filter}
            asset_dirs = [d for d in asset_dirs if os.path.basename(d).replace("asset_", "") in filter_set]
            if not asset_dirs:
                print(f"[ERROR] No data found for requested assets: {assets_filter}")
                return
            print(f"Training only assets: {assets_filter}")

        results_root = os.path.join(self.results_dir, "per_asset")
        os.makedirs(results_root, exist_ok=True)

        loader = TabularLoader(use_stats=getattr(args, "use_stats", False))
        fn     = loader.compute_statistical_features if loader.use_stats else loader.flatten_sequences

        trained = []
        for asset_dir in asset_dirs:
            asset_id = os.path.basename(asset_dir).replace("asset_", "")
            try:
                X_train_raw = np.load(os.path.join(asset_dir, "X_train.npy"))
                y_train     = np.load(os.path.join(asset_dir, "y_train.npy"))
                X_val_raw   = np.load(os.path.join(asset_dir, "X_val.npy"))
                y_val       = np.load(os.path.join(asset_dir, "y_val.npy"))
            except FileNotFoundError:
                print(f"  [SKIP] Asset {asset_id}: missing NPY files")
                continue

            X_train = fn(X_train_raw)
            X_val   = fn(X_val_raw)
            y_train_bin = (y_train.mean(axis=1) > 0).astype(np.float32) if y_train.ndim > 1 else y_train
            y_val_bin   = (y_val.mean(axis=1) > 0).astype(np.float32) if y_val.ndim > 1 else y_val

            model_kwargs = {"random_state": args.seed, "n_estimators": args.n_estimators}
            if self.model_type == "xgboost":
                spw = loader.compute_scale_pos_weight(y_train_bin)
                model_kwargs["scale_pos_weight"] = spw

            model = self._build_model(**model_kwargs)
            model = self._fit(model, X_train, y_train_bin, X_val, y_val_bin)

            val_proba = model.predict_proba(X_val)[:, 1]
            best_th   = (
                self._metrics.find_best_threshold(y_val_bin, val_proba)
                if y_val_bin.sum() > 0 else 0.5
            )
            feature_mode = "statistical" if loader.use_stats else "raw_flatten"
            model_path   = os.path.join(self.models_dir, f"{self.model_type}_asset_{asset_id}.pkl")
            joblib.dump(
                {
                    "model": model, "threshold": best_th,
                    "feature_mode": feature_mode, "asset_id": asset_id,
                },
                model_path,
            )
            print(f"  Asset {asset_id}: model saved → {model_path}")
            trained.append(asset_id)

        print(f"\nPer-asset {self.model_type} training complete. Trained: {trained}")
        print(
            f"Next: python -m src.evaluation.evaluate_tree "
            f"--per_asset --model {self.model_type}"
        )
