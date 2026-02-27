"""
XGBoost Architecture — models.architectures.xgboost_model
Factory function for the XGBoost binary classifier used in anomaly detection.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xgboost import XGBClassifier


# Default hyperparameters — override via config or train script arguments
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
    "random_state": 42,
}


def build_xgboost_model(
    scale_pos_weight: float = 1.0,
    n_estimators: int = XGBOOST_DEFAULTS["n_estimators"],
    max_depth: int = XGBOOST_DEFAULTS["max_depth"],
    learning_rate: float = XGBOOST_DEFAULTS["learning_rate"],
    subsample: float = XGBOOST_DEFAULTS["subsample"],
    colsample_bytree: float = XGBOOST_DEFAULTS["colsample_bytree"],
    min_child_weight: float = XGBOOST_DEFAULTS["min_child_weight"],
    gamma: float = XGBOOST_DEFAULTS["gamma"],
    reg_lambda: float = XGBOOST_DEFAULTS["reg_lambda"],
    reg_alpha: float = XGBOOST_DEFAULTS["reg_alpha"],
    tree_method: str = XGBOOST_DEFAULTS["tree_method"],
    n_jobs: int = XGBOOST_DEFAULTS["n_jobs"],
    random_state: int = XGBOOST_DEFAULTS["random_state"],
    use_gpu: bool = False,
) -> XGBClassifier:
    """
    Build an XGBoost binary classifier for fault detection.

    The model ingests flattened (window×feature) vectors and outputs
    a probability in [0, 1]. Threshold is tuned post-training.

    Args:
        scale_pos_weight: Ratio neg/pos for imbalance handling (auto-computed if 0).
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage factor per round.
        subsample: Row sub-sampling ratio per tree.
        colsample_bytree: Column sub-sampling ratio per tree.
        min_child_weight: Minimum sum of instance weight in a leaf.
        gamma: Minimum loss reduction required for a split.
        reg_lambda: L2 regularisation.
        reg_alpha: L1 regularisation.
        tree_method: 'hist' (CPU) or 'gpu_hist' / 'cuda' (GPU).
        n_jobs: Parallelism degree (-1 = all cores).
        random_state: RNG seed.
        use_gpu: Force GPU method regardless of tree_method argument.

    Returns:
        Configured (not yet trained) XGBClassifier
    """
    effective_tree_method = "cuda" if use_gpu else tree_method

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
        tree_method=effective_tree_method,
        random_state=random_state,
        verbosity=1,
    )

    print(f"[XGBoost] Model configured: {n_estimators} estimators, depth={max_depth}, "
          f"lr={learning_rate}, scale_pos_weight={scale_pos_weight:.3f}")
    return model
