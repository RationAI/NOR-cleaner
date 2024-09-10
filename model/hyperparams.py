"""
Hyperparameters for the model.
"""

# fmt: off
# Parameters for XGBoost
_XGBC_HYPERPARAMS = {
    "objective": "binary:logistic",
    "random_state": 42,
    "enable_categorical": False,

    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.1,
    "gamma": 0,
    "min_child_weight": 1,
    "subsample": 1,

    # The ratio of negative and positive values
    # "scale_pos_weight": (n_records - n_positives) / n_positives,
}
# fmt: on


def get_xgbc_hyperparams() -> dict:
    """
    Get the hyperparameters for XGBoost.
    """
    return _XGBC_HYPERPARAMS
