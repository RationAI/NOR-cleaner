"""
File for data preparation
"""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd

from data_preparation.algo_filtering import algorithmic_filtering_icd_10
from data_preparation.columns_check import check_columns
from data_preparation.drop_columns import drop_columns
from data_preparation.feature_transformation import (
    cols_to_int,
    count_records_per_patient,
    count_unknown_values,
    fill_nan_to_zero,
    transform_all_tnm,
    transform_date,
    transform_dg_to_number,
    transform_distant_metastasis,
    transform_extend_of_disease,
    transform_lateralita_kod,
    transform_medical_institute_code,
    transform_morfologie_klasifikace_kod,
    transform_pn_examination_cols,
    transform_sentinel_lymph_node,
    transform_stadium,
    transform_stanoveni_to_num,
    transform_topografie_kod,
)
from data_preparation.translate_english import df_english_translation
from lib.column_names import TARGET_COLUMN
from lib.dataset_names import DATA_DIR, DATA_PREPROCESSED_FILENAME, DatasetType
from lib.load_dataset import get_original_dataset
from scripts.constants import PREPARED_DATA_PATH

logger = logging.getLogger(__name__)


def all_transformations(df: pd.DataFrame) -> pd.DataFrame:
    PIPELINE: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
        drop_columns,
        fill_nan_to_zero,
        cols_to_int,
        transform_pn_examination_cols,
        transform_sentinel_lymph_node,
        transform_stadium,
        transform_extend_of_disease,
        transform_distant_metastasis,
        partial(transform_dg_to_number, drop=True),
        transform_topografie_kod,
        transform_lateralita_kod,
        transform_morfologie_klasifikace_kod,
        transform_medical_institute_code,
        transform_all_tnm,
        transform_stanoveni_to_num,
        count_unknown_values,
        partial(transform_date, drop=True),
        count_records_per_patient,
        # df_english_translation,
    ]

    df = df.copy()
    for f in PIPELINE:
        df = f(df)

    return df


def preprocess_data(
    data: pd.DataFrame, save_path: Path | None = None
) -> pd.DataFrame:
    """
    Prepare the data before merging the records.
    This consists of:
    - Algorithmic filtering of ICD-10 codes
    - Dropping unnecessary columns
    - Applying all transformations
    - Saving the prepared data

    Parameters:
        data: pd.DataFrame
            The DataFrame to prepare.
        save_path: Path | None
            The path to save the prepared data.
            If None, the data will not be saved.

    Returns:
        pd.DataFrame:
            The prepared DataFrame.
    """

    # Copy the data to avoid modifying the original
    data = data.copy()

    logger.info(f"Data loaded, shape: {data.shape}")

    # Check that all necessary columns are present
    check_columns(data)
    logger.info("All necessary columns are present.")

    logger.info(
        "Algorithmic filtering of ICD-10 codes. It may take a while..."
    )
    data, dropped_record_ids = algorithmic_filtering_icd_10(data)
    logger.info("Algorithmic filtering done.")
    logger.info(
        f"Number of algorithmically filtered records: {len(dropped_record_ids)}"
    )

    # Apply all transformations
    logger.info("Applying all transformations...")
    data = all_transformations(data)
    logger.info("All transformations applied.")

    # Move vyporadani_final to the end
    cols = list(data.columns)
    cols.remove(TARGET_COLUMN)
    cols.append(TARGET_COLUMN)
    data = data[cols]

    # Check there are no null values
    if data.isnull().sum().sum() != 0:
        raise ValueError("There are still null values")

    # Check that there are no object columns
    if not data.select_dtypes(include="object").columns.empty:
        raise TypeError(
            f"Object columns: {data.select_dtypes(include='object').columns}"
        )

    # Save the data
    if save_path is None:
        logger.info("No save path provided. The data will not be saved.")
    else:
        data.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")

    return data
