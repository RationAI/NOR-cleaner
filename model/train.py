from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from lib.column_names import TARGET_COLUMN
from lib.load_dataset import load_merged_data_from_csv

ModelType = xgb.Booster


def train(model: ModelType, path: Path) -> ModelType:
    """
    Train the model using the dataset from the given path.

    Parameters:
        model: ModelType
            The model to train

        path: str
            The path to the dataset

    Returns:
        ModelType
            The trained model
    """

    # TODO: Load raw data and preprocess it
    X, y, _, _ = load_merged_data_from_csv(path)
    model.fit(X, y)

    return model


def train_save_model(model: ModelType, data_path: str, save_path: str) -> None:
    # Train the model
    model = train(model, data_path)

    # Save the model
    model.save_model(save_path)
