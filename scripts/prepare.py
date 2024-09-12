"""
Script for preparing data for the model
"""

import logging
from pathlib import Path

import pandas as pd

import data_preparation
from data_preparation.data_merge import prepare_merged_data
from data_preparation.preprocess_data import preprocess_data
from lib import LOG_CONFIG_KWARGS
from lib.load_dataset import load_raw_data
from scripts.constants import *

# Set up the logger
logging.basicConfig(level=logging.INFO, **LOG_CONFIG_KWARGS)  # type: ignore


def main() -> None:
    """
    Main function for preparing the data.
    """

    # Load the raw data
    data = load_raw_data(DATASET_PATH)

    # Prepare the data
    data = preprocess_data(data, save_path=PREPARED_DATA_PATH)

    # Merge the records
    data = prepare_merged_data(data, save_path=MERGED_DATA_PATH)


if __name__ == "__main__":
    main()
