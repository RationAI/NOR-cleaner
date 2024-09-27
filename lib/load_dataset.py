"""
File for parsing datasets into DataFrames.
"""

import logging
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd

import data_preparation
from data_preparation.fold_unfold_merged_data import unfold_merged_data

DATA_DIR = "data"

logger = logging.getLogger(__name__)

"""
DatasetType
    Type of dataset. It can be one of the following:
        - "2019-2021"      -- Dataset of records from 2019 to 2021.
        - "2022"           -- Dataset records from 2019 to 2022.
        - "verify_dataset" -- Dataset of new records to test the model.
"""
DatasetType = Literal["2019-2021", "2022", "verify_dataset"]


def parse_dtypes(dtypes_csv: Path) -> tuple[dict[str, str], list[str]]:
    """
    Parse the dtypes file into a dictionary.

    Parameters:
        dtypes_csv: Path
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


# Exception for missing dtypes file
class DTypesNotPresentError(Exception):
    pass


def get_original_dataset(path: Path) -> pd.DataFrame:
    """
    Parse the original dataset into a DataFrame.
    Requires to have the dtypes data_dtypes.csv in the same directory.

    Parameters:
        path: Path
            Path to the original dataset.

    Returns:
        pd.DataFrame:
            DataFrame of the dataset.
    """
    if path.suffix != ".csv":
        raise ValueError(f"The file must be a CSV file. Got: {path.suffix}")

    # Check if the dtypes file is present
    dtypes_path = path.parent / "data_dtypes.csv"
    if not dtypes_path.exists():
        raise DTypesNotPresentError(
            f"The dtypes file is missing. Expected at: {dtypes_path}"
        )

    dtype, parse_dates = parse_dtypes(dtypes_path)

    dataset = pd.read_csv(
        path,
        dtype=dtype,
        parse_dates=parse_dates,
    )

    return dataset


def unpickle_data(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_merged_data_from_csv(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load the merged data from CSV files.

    Parameters:
       path: Path
              Path to the merged data CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
            Tuple of X_merged, y_merged, report_ids and patient_ids.
    """
    merged_data = pd.read_csv(path)

    return unfold_merged_data(merged_data)


def load_merged_data(
    which: DatasetType,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the merged data.

    Parameters:
        which: DatasetType
            Which dataset to get.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
            Tuple of X_merged, y_merged, report_ids and patient_ids.
    """
    data_dir = Path(DATA_DIR) / which / "merged_data"

    def join(data_dir, filename):
        return data_dir + "/" + filename

    x_merged_file = join(data_dir, "X_merged.pkl")
    preds_file = join(data_dir, "y_merged.pkl")
    report_ids_file = join(data_dir, "record_ids.pkl")
    patient_ids_file = join(data_dir, "patient_ids.pkl")

    X_merged = unpickle_data(x_merged_file)
    y_merged = unpickle_data(preds_file)
    report_ids = unpickle_data(report_ids_file)
    patient_ids = unpickle_data(patient_ids_file)

    return X_merged, y_merged, report_ids, patient_ids


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
        # Attempt to load the original dataset with the dtypes
        try:
            data = get_original_dataset(path)
            logger.info("The dataset was loaded with the dtypes.")
        except DTypesNotPresentError:
            logger.warning(
                "The dtypes file is missing. The data types will be inferred."
            )
            # Otherwise, load the dataset without the dtypes
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

    logger.info(f"Loaded the raw dataset with shape: {data.shape}")

    return data


if __name__ == "__main__":
    # expert_df = get_ready_data(which="2019-2021")
    # expert_df = get_original_dataset(which="2019-2021")
    # print(expert_df.shape)

    merged_data = load_merged_data(which="2019-2021")
    print(merged_data[0].shape)
