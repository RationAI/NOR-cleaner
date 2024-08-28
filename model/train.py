from typing import Any
import xgboost as xgb
import numpy as np
import pandas as pd

from lib.column_names import TARGET_COLUMN
from lib.load_dataset import load_merged_data


ModelType = Any


def train(model: ModelType, path: str) -> ModelType:

    # TODO: Make loading dataset from CSV file
    # For now, load the dataset from the pickle file
    X, y, _, _ = load_merged_data(which="2019-2021")

    # # Load the dataset
    # data = pd.read_csv(path)

    # # Split the dataset into features and target
    # X = data.drop(columns=[TARGET_COLUMN])
    # y = data[TARGET_COLUMN]

    model.fit(X, y)

    return model


def train_save_model(model: ModelType, save_path: str) -> None:

    # Train the model
    model = train(model, save_path)

    # Save the model
    model.save_model(save_path)
