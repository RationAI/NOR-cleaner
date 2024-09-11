"""
Script for preparing data for the model
"""

import logging
from pathlib import Path

import pandas as pd

import data_preparation
from data_preparation.data_merge import prepare_merged_data
from data_preparation.data_preparation import prepare_data
from scripts.constants import *

# Set up the logger
# Format: LEVEL:__name__:TIME:MESSAGE
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(asctime)s:%(message)s"
)


def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load the raw data. The raw dataset is expected to contain the columns
    that are checked in `check_columns()` function.

    Parameters:
        path: Path
            Path to the raw data directory.

    Returns:
        pd.DataFrame:
            DataFrame of the raw data.
    """
    if path.suffix not in [".sav", ".xlsx"]:
        raise ValueError(
            f"Unsupported file format. Use `.sav` or `.xlsx`. Got suffix `{path.suffix}`."
        )

    data: pd.DataFrame | None = None
    if path.suffix == ".sav":
        data = pd.read_spss(str(path))
    else:
        data = pd.read_excel(path, engine="openpyxl")

    VYPORADENI_CATEGORY = "vyporadani_kat"
    if VYPORADENI_CATEGORY in data.columns:
        raise ValueError(
            f"The dataset contains the column {VYPORADENI_CATEGORY}. Remove it."
        )

    # Check if all necessary columns are present in the DataFrame
    data_preparation.check_columns(data)

    return data


def main() -> None:
    """
    Main function for preparing the data.
    """

    # Load the raw data
    data = load_raw_data(DATASET_PATH)

    # Prepare the data
    data = prepare_data(data, SAVE_PREPARED_DATA)

    # Merge the records
    data = prepare_merged_data(data, SAVE_MERGED_DATA)


if __name__ == "__main__":
    main()
