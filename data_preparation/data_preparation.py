"""
File for data preparation
"""

import logging
import pickle
from functools import partial
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


def _save_data(data: pd.DataFrame, dataset_type: DatasetType) -> None:
    """
    Save the data to a pickle file
    """
    data_preprocessed = (
        f"{DATA_DIR}/{dataset_type}/{DATA_PREPROCESSED_FILENAME}"
    )

    with open(data_preprocessed, "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Data saved to {data_preprocessed}")


def prepare_data(dataset_type: DatasetType) -> pd.DataFrame:
    """
    Prepare the data for the model
    """
    data = get_original_dataset(which=dataset_type)
    logger.info(f"Data of type `{dataset_type}` loaded, shape: {data.shape}")

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
    _save_data(data, dataset_type)

    return data
