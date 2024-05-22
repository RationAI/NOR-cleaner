"""
File for parsing datasets into DataFrames.
"""

import pickle
import warnings
from typing import Literal

import pandas as pd

from lib.dataset_names import (
    DATA_PREPROCESSED_FILENAME,
    DATASET_LIST,
    ORIGINAL_DATASET_FILENAME,
    DatasetType,
    get_dataset_directory,
)


def _check_dataset_type(which: DatasetType) -> None:
    if which not in DATASET_LIST:
        raise ValueError(f"Invalid dataset type. Choose from: {DATASET_LIST}")


def parse_dtypes(dtypes_csv: str) -> tuple[dict[str, str], list[str]]:
    """
    Parse the dtypes file into a dictionary.

    Parameters:
        dtypes_csv: str
            Path to the dtypes file.

    Returns:
        dict[str, str]:
            Dictionary of column names and their respective data types.
        list[str]:
            List of dates columns.
    """
    dtype_df = pd.read_csv(dtypes_csv, index_col=0)
    dtype = dtype_df.to_dict()["dtype"]

    parse_dates = [k for k, v in dtype.items() if v == "datetime64[ns]"]
    dtype = {k: v for k, v in dtype.items() if v != "datetime64[ns]"}

    return dtype, parse_dates


def get_original_dataset(which: DatasetType) -> pd.DataFrame:
    """
    Parse the original dataset into a DataFrame.
    Requires to have the dtypes in data/data_dtypes.csv.

    Parameters:
        dataset_csv: str
            Path to the dataset file.

        which: DatasetType
            Which dataset to get. Defaults to "2019-2021".

    Returns:
        pd.DataFrame:
            DataFrame of the dataset.
    """
    dataset_path = get_dataset_directory(which)
    dtype, parse_dates = parse_dtypes(f"{dataset_path}/data_dtypes.csv")

    # Ignore UserWarning
    with warnings.catch_warnings(action="ignore"):
        dataset = pd.read_csv(
            f"{dataset_path}/{ORIGINAL_DATASET_FILENAME}",
            dtype=dtype,
            parse_dates=parse_dates,
        )

    return dataset


def unpickle_data(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_ready_data(which: DatasetType) -> pd.DataFrame:
    """
    Get data after data preparation.

    Parameters:
        which: DatasetType
            Which dataset to get.

    Returns:
        pd.DataFrame:
            DataFrame of the dataset.
    """
    _check_dataset_type(which)

    dataset_dir = get_dataset_directory(which)
    return unpickle_data(f"{dataset_dir}/{DATA_PREPROCESSED_FILENAME}")


if __name__ == "__main__":
    expert_df = get_ready_data(which="2019-2021")
    # expert_df = get_original_dataset(which="2019-2021")
    print(expert_df.shape)
