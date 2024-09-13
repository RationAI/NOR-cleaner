"""
Script for predicting the target variable on the unseen data
"""

import logging

import pandas as pd
import xgboost as xgb

from data_preparation import (
    prepare_merged_data,
    preprocess_data,
    unfold_merged_data,
)
from lib import LOG_CONFIG_KWARGS
from lib.column_names import TARGET_COLUMN
from model.hyperparams import get_xgbc_hyperparams
from scripts.constants import *
from scripts.prepare import load_raw_data

logging.basicConfig(level=logging.INFO, **LOG_CONFIG_KWARGS)  # type: ignore

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function for predicting the target variable on the unseen data.
    """
    # Load the unseen data
    logger.info("Loading the unseen data...")
    data = load_raw_data(PREDICT_DATA_PATH)

    # Prepare the data
    logger.info("Preparing the data...")
    data = prepare_merged_data(preprocess_data(data))

    # TODO:
    # - predict.csv contains `y` column, which is the target variable. Remove it.
    X, _, _, _ = unfold_merged_data(data)

    # Load the model
    logger.info("Loading the model...")
    model = xgb.XGBClassifier(**get_xgbc_hyperparams())
    model.load_model(MODEL_PATH)

    # Predict the target variable
    logger.info("Predicting the target variable...")
    y_pred = model.predict(X)

    # Save the predictions
    logger.info("Saving the predictions...")
    pd.DataFrame(y_pred, columns=[TARGET_COLUMN]).to_csv(
        PREDICTIONS_PATH, index=False
    )
    logger.info(f"Predictions saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
