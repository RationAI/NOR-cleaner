"""
Load model from .json file
"""

from pathlib import Path

import xgboost as xgb
from model import ModelType


def load_model(path: Path) -> ModelType:
    """
    Load the model from the given path.

    Parameters:
        path: Path
            The path to the model
            
    Returns:
        ModelType
            The loaded model
    """

    if path.suffix != ".json":
        raise ValueError(
            f"Model must be loaded from a JSON file, got {path.name}"
        )
    
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model
