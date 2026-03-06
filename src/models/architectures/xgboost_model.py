"""
XGBoostModel — models.architectures.xgboost_model
Factory class for the XGBoost binary classifier used in anomaly detection.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# XGBClassifier is imported lazily inside build() so that importing this
# module file does not fail when xgboost is not installed.


# Default hyperparameters — override via constructor arguments or config
XGBOOST_DEFAULTS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "gamma":            0.0,
    "reg_lambda":       1.0,
    "reg_alpha":        0.0,
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "tree_method":      "hist",
    "n_jobs":           -1,
    "verbosity":        1,
    "random_state":     42,
}


class XGBoostModel:
    """
    Configures an XGBoost binary classifier for SCADA fault detection.

    The model ingests flattened (window × feature) vectors and outputs a
    probability in [0, 1]. The classification threshold is tuned post-training
    by MetricsCalculator.find_best_threshold().

    Args:
        scale_pos_weight: Imbalance ratio (neg_count / pos_count).
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage factor per round.
        subsample: Row sub-sampling ratio per tree.
        colsample_bytree: Column sub-sampling ratio per tree.
        min_child_weight: Minimum sum of instance weight in a leaf.
        gamma: Minimum loss-reduction required for a split.
        reg_lambda: L2 regularisation weight.
        reg_alpha: L1 regularisation weight.
        tree_method: 'hist' (CPU) or 'cuda' (GPU).
        n_jobs: Parallelism degree (-1 = all cores).
        random_state: RNG seed.
        use_gpu: Override tree_method to 'cuda'.
    """

    def __init__(
        self,
        scale_pos_weight: float = 1.0,
        n_estimators:     int   = XGBOOST_DEFAULTS["n_estimators"],
        max_depth:        int   = XGBOOST_DEFAULTS["max_depth"],
        learning_rate:    float = XGBOOST_DEFAULTS["learning_rate"],
        subsample:        float = XGBOOST_DEFAULTS["subsample"],
        colsample_bytree: float = XGBOOST_DEFAULTS["colsample_bytree"],
        min_child_weight: float = XGBOOST_DEFAULTS["min_child_weight"],
        gamma:            float = XGBOOST_DEFAULTS["gamma"],
        reg_lambda:       float = XGBOOST_DEFAULTS["reg_lambda"],
        reg_alpha:        float = XGBOOST_DEFAULTS["reg_alpha"],
        tree_method:      str   = XGBOOST_DEFAULTS["tree_method"],
        n_jobs:           int   = XGBOOST_DEFAULTS["n_jobs"],
        random_state:     int   = XGBOOST_DEFAULTS["random_state"],
        use_gpu:          bool  = False,
    ) -> None:
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma            = gamma
        self.reg_lambda       = reg_lambda
        self.reg_alpha        = reg_alpha
        self.tree_method      = "cuda" if use_gpu else tree_method
        self.n_jobs           = n_jobs
        self.random_state     = random_state

    def build(self):
        """
        Instantiate and return a configured (untrained) XGBClassifier.

        Returns:
            XGBClassifier ready to be fitted.
        """
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=self.n_jobs,
            scale_pos_weight=self.scale_pos_weight,
            tree_method=self.tree_method,
            random_state=self.random_state,
            verbosity=1,
        )
        print(
            f"[XGBoost] Model configured: {self.n_estimators} estimators, "
            f"depth={self.max_depth}, lr={self.learning_rate}, "
            f"scale_pos_weight={self.scale_pos_weight:.3f}"
        )
        return model


# ---------------------------------------------------------------------------
# Backward-compatible module-level alias
# ---------------------------------------------------------------------------

def build_xgboost_model(
    scale_pos_weight: float = 1.0,
    n_estimators:     int   = XGBOOST_DEFAULTS["n_estimators"],
    max_depth:        int   = XGBOOST_DEFAULTS["max_depth"],
    learning_rate:    float = XGBOOST_DEFAULTS["learning_rate"],
    subsample:        float = XGBOOST_DEFAULTS["subsample"],
    colsample_bytree: float = XGBOOST_DEFAULTS["colsample_bytree"],
    min_child_weight: float = XGBOOST_DEFAULTS["min_child_weight"],
    gamma:            float = XGBOOST_DEFAULTS["gamma"],
    reg_lambda:       float = XGBOOST_DEFAULTS["reg_lambda"],
    reg_alpha:        float = XGBOOST_DEFAULTS["reg_alpha"],
    tree_method:      str   = XGBOOST_DEFAULTS["tree_method"],
    n_jobs:           int   = XGBOOST_DEFAULTS["n_jobs"],
    random_state:     int   = XGBOOST_DEFAULTS["random_state"],
    use_gpu:          bool  = False,
):
    """Legacy alias — wraps XGBoostModel(...).build()."""
    return XGBoostModel(
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        gamma=gamma, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        tree_method=tree_method, n_jobs=n_jobs,
        random_state=random_state, use_gpu=use_gpu,
    ).build()
