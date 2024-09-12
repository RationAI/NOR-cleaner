"""
Script for evaluating the model performance
"""

import logging

import pandas as pd
import xgboost as xgb

from data_preparation.column_names import RECORD_COUNT_NAME
from data_preparation.fold_unfold_merged_data import unfold_merged_data
from lib import LOG_CONFIG_KWARGS
from lib.column_names import TARGET_COLUMN
from lib.eval import (
    cross_validation_merged_df,
    process_each_records_scores,
    process_scoring_dict,
)
from lib.load_dataset import load_merged_data_from_csv
from model.hyperparams import get_xgbc_hyperparams
from scripts.constants import *

logging.basicConfig(level=logging.INFO, **LOG_CONFIG_KWARGS)  # type: ignore

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function for evaluating the model performance.
    """

    # Cross-validation
    # Load merged data
    logger.info("Loading the merged data...")
    X, y, _, patient_ids = load_merged_data_from_csv(MERGED_DATA_PATH)

    # Load the model
    logger.info("Loading the model...")
    model = xgb.XGBClassifier(**get_xgbc_hyperparams())
    # model.load_model(MODEL_PATH)

    # Evaluate the model
    logger.info("Evaluating the model...")

    scores, each_scores = cross_validation_merged_df(
        model=model,
        n=TAKE_RANGE[1],
        X_merged=X,
        y=y,
        groups=patient_ids,
        augment_data=False,
        each_record_eval=True,
        random_state=42,
        record_count_getter=lambda X: X[RECORD_COUNT_NAME, 0],
        # TODO: Save confusion matrix plot to a file
        # confusion_matrix_plot=True,
    )

    # Log the results
    # TODO: Resolve the types
    logger.info(f"Results:\n{process_scoring_dict(scores)}")  # type: ignore
    logger.info(f"Results for each record:\n{process_each_records_scores(each_scores)}")  # type: ignore


if __name__ == "__main__":
    main()
