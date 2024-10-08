"""
Script for training the model
"""

import logging

import pandas as pd
import xgboost as xgb

import data_preparation
from lib import LOG_CONFIG_KWARGS
from lib.column_names import TARGET_COLUMN
from lib.load_dataset import load_merged_data_from_csv
from model.hyperparams import get_xgbc_hyperparams
from scripts.constants import *

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function for training the model.
    """

    # Load the data
    logger.info("Loading the data...")
    X, y, _, _ = load_merged_data_from_csv(MERGED_DATA_PATH)

    logger.info(f"Data loaded, X shape: {X.shape}, y shape: {y.shape}")

    # Train the model
    logger.info("Training the model...")

    model = xgb.XGBClassifier(**get_xgbc_hyperparams())
    model.fit(X, y)

    logger.info("Model training completed.")

    # Save the model
    logger.info(f"Saving the model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    logger.info(f"Model saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, **LOG_CONFIG_KWARGS)  # type: ignore
    main()
