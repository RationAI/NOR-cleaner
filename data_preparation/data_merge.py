import argparse
import datetime
import logging
import pickle
from functools import partial
from pathlib import Path

import pandas as pd

import lib.utils
from data_preparation.column_names import (
    ALGO_FILTERED_COLUMN,
    NOVELTY_RANK,
    RECORD_COUNT_NAME,
)
from data_preparation.fold_unfold_merged_data import (
    fold_merged_data,
    unfold_merged_data,
)
from data_preparation.merged_transformation import (
    add_cols_equal,
    any_c76_c80_check,
    difference_in_dates,
    equality_of_dg_codes,
    feature_difference,
    init_fillna_dict,
    records_equal,
)
from lib.column_names import (
    CREATED_WITH_BATCH,
    PATIENT_ID_NAME,
    RECORD_ID_NAME,
    SENTINEL_LYMPH_NODE,
    TARGET_COLUMN,
)
from lib.merge_records import drop_multi_cols, merge_groups_each_row
from scripts.constants import MERGED_DATA_PATH, PREPARED_DATA_PATH, TAKE_RANGE

logger = logging.getLogger(__name__)

# Number of records that will be merged
N_MERGED = TAKE_RANGE[1]

MULTI_COLS_TO_DROP = [PATIENT_ID_NAME, RECORD_COUNT_NAME]

DONT_TAKE_COLS_EQUAL_COUNT = [
    RECORD_ID_NAME,
    PATIENT_ID_NAME,
    RECORD_COUNT_NAME,
    NOVELTY_RANK,
    "ICD_Equal_With",
    "Any_ICD_C76_C80",
]

DONT_TAKE_COLS_FEATURE_DIFF = [
    RECORD_ID_NAME,
    PATIENT_ID_NAME,
    RECORD_COUNT_NAME,
    NOVELTY_RANK,
    "ICD_Equal_With",
    "Any_ICD_C76_C80",
    "ICDRangeC76-C80",
    CREATED_WITH_BATCH,
    SENTINEL_LYMPH_NODE,
]


def all_merged_transformations(df: pd.DataFrame) -> pd.DataFrame:
    COLS_EQUAL_COUNT = [
        col for col, _ in df.columns if col not in DONT_TAKE_COLS_EQUAL_COUNT
    ]

    COLS_FEATURE_DIFF = [
        col for col, _ in df.columns if col not in DONT_TAKE_COLS_FEATURE_DIFF
    ]

    PIPELINE = [
        partial(any_c76_c80_check, n=N_MERGED),
        # partial(equality_of_dg_codes, n=N_MERGED),
        # partial(records_equal, n=N_MERGED),
        partial(add_cols_equal, cols=COLS_EQUAL_COUNT, n=N_MERGED),
        partial(feature_difference, cols=COLS_FEATURE_DIFF, n=N_MERGED),
        # partial(
        #     difference_in_dates,
        #     n=N_MERGED,
        #     date_cols=["DateOfEstablishingDg"],
        #     drop=True,
        # ),
    ]

    for func in PIPELINE:
        df = func(df)

    return df


# Save the data
X_MERGED_FILENAME = "X_merged"
PREDS_FILENAME = "y_merged"
REPORT_IDS_FILENAME = "record_ids"
PATIENT_IDS_FILENAME = "patient_ids"


def prepare_merged_data(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Prepare the data for the model

    Parameters:
        data: pd.DataFrame | None
            The DataFrame to prepare.
            If None, the data will be loaded from
            the path `SAVE_PREPARED_DATA / PREPARED_DATA_FILENAME`.
    """
    if data is None:
        data = pd.read_csv(PREPARED_DATA_PATH)
    else:
        # Copy the data to avoid modifying the original
        data = data.copy()

    # Check if there are any missing values
    if data.isna().sum().sum() != 0:
        raise ValueError(
            f"There are missing values in the data: {data.isna().sum().sum()} values."
        )

    # Drop records which were algorithmically filtered
    if ALGO_FILTERED_COLUMN in data.columns:
        data = data[data[ALGO_FILTERED_COLUMN] == 0]
        data.drop(columns=ALGO_FILTERED_COLUMN, inplace=True)

    # Filter range of number of records per patient
    len_bef = len(data)
    data = data[data[RECORD_COUNT_NAME].between(TAKE_RANGE[0], TAKE_RANGE[1])]
    len_aft = len(data)

    logger.info(
        f"{len_bef - len_aft} rows removed after taking in range {TAKE_RANGE}"
    )

    FILLNA_DICT = init_fillna_dict(data)
    X_merged = merge_groups_each_row(
        df=data,
        group_col=PATIENT_ID_NAME,
        n=N_MERGED,
        drop_padded_by=RECORD_COUNT_NAME,
        null_value=FILLNA_DICT[RECORD_COUNT_NAME],
        fillna=FILLNA_DICT,
    ).sort_index()

    y_merged = X_merged[TARGET_COLUMN, 0].astype(int)
    y_merged.name = (TARGET_COLUMN, 0)

    record_ids = X_merged[RECORD_ID_NAME].copy()
    record_ids.columns = pd.MultiIndex.from_tuples(
        [(RECORD_ID_NAME, i) for i in range(N_MERGED)]
    )

    patient_ids = X_merged[PATIENT_ID_NAME, 0].copy()
    patient_ids.name = (PATIENT_ID_NAME, 0)

    X_merged = drop_multi_cols(X_merged, MULTI_COLS_TO_DROP, N_MERGED)
    # Drop column to predict, record_id and patient_id
    X_merged.drop(
        columns=[TARGET_COLUMN, RECORD_ID_NAME, PATIENT_ID_NAME],
        inplace=True,
    )

    # Transform float columns to int
    float_cols = X_merged.select_dtypes(include=float).columns
    X_merged[float_cols] = X_merged[float_cols].astype(int)

    logger.info(
        f"Shape after dropping columns: {X_merged.shape}, {y_merged.shape}"
    )

    X_merged = all_merged_transformations(X_merged)

    # Check if there are any missing values
    non_null_num = X_merged.isnull().sum().sum()
    if non_null_num != 0:
        raise ValueError(f"Number of null values: {non_null_num}")

    # Concat to one dataframe and save
    out = fold_merged_data(X_merged, y_merged, record_ids, patient_ids)
    out.to_csv(MERGED_DATA_PATH, index=False)

    logger.info(f"Saved data to {MERGED_DATA_PATH}")

    return out


if __name__ == "__main__":
    merged_data = prepare_merged_data()

    X_merged, y_merged, record_ids, patient_ids = unfold_merged_data(
        merged_data
    )
