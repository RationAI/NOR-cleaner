"""
Script for preparing data for the model
"""

import logging
from pathlib import Path

import pandas as pd

import data_preparation
from data_preparation.data_merge import prepare_merged_data
from data_preparation.data_preparation import preprocess_data
from lib import LOG_CONFIG_KWARGS
from scripts.constants import *

# Set up the logger
logging.basicConfig(level=logging.INFO, **LOG_CONFIG_KWARGS)  # type: ignore


def load_raw_data(path: Path | str) -> pd.DataFrame:
    """
    Load the raw data.
    Supported file formats: `.sav`, `.xlsx` and `.csv`.
    The raw dataset is expected to contain the columns that are checked
    in `check_columns()` function.

    Parameters:
        path: Path
            Path to the raw data directory.

    Returns:
        pd.DataFrame:
            DataFrame of the raw data.
    """

    # Convert the path to Path object if it is a string
    if isinstance(path, str):
        path = Path(path)

    SUPPORTED_FORMATS = [".sav", ".xlsx", ".csv"]

    data: pd.DataFrame | None = None
    if path.suffix == ".sav":
        data = pd.read_spss(str(path))
    elif path.suffix == ".xlsx":
        data = pd.read_excel(path, engine="openpyxl")
    elif path.suffix == ".csv":
        data = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format. Use one of {SUPPORTED_FORMATS}. Got suffix `{path.suffix}`."
        )

    VYPORADENI_CATEGORY = "vyporadani_kat"
    if VYPORADENI_CATEGORY in data.columns:
        if data[VYPORADENI_CATEGORY].nunique() == 1:
            data.drop(columns=[VYPORADENI_CATEGORY], inplace=True)
        else:
            raise ValueError(
                f"The dataset contains the column `{VYPORADENI_CATEGORY}` with more than one unique value."
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
    data = preprocess_data(data, save_path=PREPARED_DATA_PATH)

    # Merge the records
    data = prepare_merged_data(data, save_path=MERGED_DATA_PATH)


if __name__ == "__main__":
    main()
