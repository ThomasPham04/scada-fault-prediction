"""Defaults for the retained Random Forest and XGBoost baselines."""

from training.hyperparameters.sequence_defaults import RANDOM_SEED

# Shared tree-classifier threshold search.
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_THRESHOLD_GRID_START = 0.05
DEFAULT_THRESHOLD_GRID_STOP = 0.95
DEFAULT_THRESHOLD_GRID_STEPS = 200

# Per-event aggregation used by the retained tree evaluator.
DEFAULT_ADAPTIVE_STD_MULTIPLIER = 0.45
DEFAULT_MIN_OUTLIER_RATIO = 0.15
DEFAULT_PER_ASSET_PERCENTILE = 85
DEFAULT_EVENT_SCORE_MULTIPLIER = 1.2

RF_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "verbose": 0,
}

XGBOOST_DEFAULT_SCALE_POS_WEIGHT = 1.0
XGBOOST_DEFAULT_USE_GPU = False
XGBOOST_DEFAULTS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 1,
    "random_state": RANDOM_SEED,
}
