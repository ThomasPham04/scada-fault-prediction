"""
Random Forest Architecture — models.architectures.random_forest
Factory function for the Random Forest binary classifier used in anomaly detection.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.ensemble import RandomForestClassifier


# Default hyperparameters
RF_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": None,          # Grow fully by default
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",     # Standard for classification
    "class_weight": "balanced", # Handles class imbalance automatically
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 0,
}


def build_random_forest_model(
    n_estimators: int = RF_DEFAULTS["n_estimators"],
    max_depth: int = RF_DEFAULTS["max_depth"],
    min_samples_split: int = RF_DEFAULTS["min_samples_split"],
    min_samples_leaf: int = RF_DEFAULTS["min_samples_leaf"],
    max_features: str = RF_DEFAULTS["max_features"],
    class_weight: str = RF_DEFAULTS["class_weight"],
    n_jobs: int = RF_DEFAULTS["n_jobs"],
    random_state: int = RF_DEFAULTS["random_state"],
) -> RandomForestClassifier:
    """
    Build a Random Forest binary classifier for fault detection.

    The model ingests flattened (window×feature) vectors and outputs
    a probability in [0, 1]. Threshold is tuned post-training.

    Uses `class_weight='balanced'` by default, which automatically weights
    each class inversely proportional to its frequency — removing the need
    for manual scale_pos_weight tuning.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth. None = grow fully until pure leaves.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        max_features: Features to consider at each split ('sqrt', 'log2', float).
        class_weight: 'balanced', 'balanced_subsample', or dict.
        n_jobs: Parallelism (-1 = all cores).
        random_state: RNG seed.

    Returns:
        Configured (not yet trained) RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )

    depth_str = str(max_depth) if max_depth else "unlimited"
    print(f"[RandomForest] Model configured: {n_estimators} trees, depth={depth_str}, "
          f"max_features={max_features}, class_weight={class_weight}")
    return model
