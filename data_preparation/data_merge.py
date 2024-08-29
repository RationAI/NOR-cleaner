import argparse
import datetime
import pickle
from functools import partial
from pathlib import Path

import pandas as pd

import lib.utils
from data_preparation.column_names import (
    ALGO_FILTERED_COLUMN,
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
    PATIENT_ID_NAME_ENG,
    RECORD_ID_NAME_ENG,
    TARGET_COLUMN_ENG,
)
from lib.dataset_names import DATASET_LIST, DatasetType, get_dataset_directory
from lib.load_dataset import get_ready_data
from lib.merge_records import drop_multi_cols, merge_groups_each_row

# Take patients with RecordCount in the range
TAKE_RANGE = (2, 3)
# Number of records that will be merged
N_MERGED = TAKE_RANGE[1]

MERGED_DIR = "merged_data"


MULTI_COLS_TO_DROP = [
    "PatientId",
    "RecordCount",
]

DONT_TAKE_COLS_EQUAL_COUNT = [
    "RecordId",
    "PatientId",
    "RecordCount",
    "NoveltyRank",
    "ICD_Equal_With",
    "Any_ICD_C76_C80",
]

DONT_TAKE_COLS_FEATURE_DIFF = [
    "RecordId",
    "PatientId",
    "RecordCount",
    "NoveltyRank",
    "ICD_Equal_With",
    "Any_ICD_C76_C80",
    "ICDRangeC76-C80",
    "CreatedWithBatch",
    "SentinelLymphNode",
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


# TODO: replace print with logging
def prepare_merged_data(dataset_type: DatasetType) -> None:
    """
    Prepare the data for the model
    """
    getter = partial(get_ready_data, which=dataset_type)
    X, y = lib.utils.get_X_y(getter=getter)

    # Drop records which were algorithmically filtered
    if ALGO_FILTERED_COLUMN in X.columns:
        X = X[X[ALGO_FILTERED_COLUMN] == 0]
        X = X.drop(columns=ALGO_FILTERED_COLUMN)
        y = y.loc[X.index]

    if not X.index.equals(y.index):
        raise ValueError("Indices are not the same for X and y.")

    # Filter range of number of records per patient
    X_reduced = pd.concat([X, y], axis=1)
    X_reduced = X_reduced[
        (X_reduced["RecordCount"] >= TAKE_RANGE[0])
        & (X_reduced["RecordCount"] <= TAKE_RANGE[1])
    ]

    y_reduced = X_reduced[TARGET_COLUMN_ENG]
    X_reduced.drop(columns=TARGET_COLUMN_ENG, inplace=True)

    print(
        len(X) - len(X_reduced), "rows removed after filtering by RecordCount."
    )

    # Check if there are any missing values
    if X.isna().sum().sum() != 0:
        raise ValueError(
            f"There are missing values in the data: {X.isna().sum().sum()} values."
        )

    FILLNA_DICT = init_fillna_dict(X_reduced)
    X_merged = merge_groups_each_row(
        df=pd.concat([X_reduced, y_reduced], axis=1),
        group_col=PATIENT_ID_NAME_ENG,
        n=N_MERGED,
        drop_padded_by=RECORD_COUNT_NAME,
        null_value=FILLNA_DICT[RECORD_COUNT_NAME],
        fillna=FILLNA_DICT,
    ).sort_index()

    y_merged = X_merged[TARGET_COLUMN_ENG, 0].astype(int)
    y_merged.name = (TARGET_COLUMN_ENG, 0)

    record_ids = X_merged[RECORD_ID_NAME_ENG].copy()
    record_ids.columns = pd.MultiIndex.from_tuples(
        [(RECORD_ID_NAME_ENG, i) for i in range(N_MERGED)]
    )

    patient_ids = X_merged[PATIENT_ID_NAME_ENG, 0].copy()
    patient_ids.name = (PATIENT_ID_NAME_ENG, 0)

    X_merged = drop_multi_cols(X_merged, MULTI_COLS_TO_DROP, N_MERGED)
    # Drop column to predict
    X_merged.drop(
        columns=[TARGET_COLUMN_ENG, RECORD_ID_NAME_ENG, PATIENT_ID_NAME_ENG],
        inplace=True,
    )

    # Transform float columns to int
    float_cols = X_merged.select_dtypes(include=float).columns
    X_merged[float_cols] = X_merged[float_cols].astype(int)

    print("Shape after dropping columns:", X_merged.shape, y_merged.shape)

    X_merged = all_merged_transformations(X_merged)

    # Check if there are any missing values
    non_null_num = X_merged.isnull().sum().sum()
    if non_null_num != 0:
        raise ValueError(f"Number of null values: {non_null_num}")

    # Save the data
    TO_SAVE = f"{get_dataset_directory(dataset_type)}/{MERGED_DIR}"
    Path(TO_SAVE).mkdir(parents=True, exist_ok=True)

    # for df, filename in [
    #     (X_merged, X_MERGED_FILENAME),
    #     (y_merged, PREDS_FILENAME),
    #     (record_ids, REPORT_IDS_FILENAME),
    #     (patient_ids, PATIENT_IDS_FILENAME),
    # ]:
    #     dump_df(df, f"{TO_SAVE}/{filename}.pkl")

    # Concat to one dataframe and save
    df = fold_merged_data(X_merged, y_merged, record_ids, patient_ids)

    df.to_csv(f"{TO_SAVE}/merged_data.csv", index=False)

    print_str = f"Saved data to {TO_SAVE}"
    print_str += f" at {datetime.datetime.now():%H:%M:%S, %d.%m.%Y}"
    print(print_str)


def dump_df(df: pd.DataFrame, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_type",
        type=str,
        help=f"The type of dataset to prepare. One of: {DATASET_LIST}",
    )

    args = parser.parse_args()
    prepare_merged_data(args.dataset_type)

    MERGED_FILE_PATH = f"{get_dataset_directory(args.dataset_type)}/{MERGED_DIR}/merged_data.csv"
    merged_data = pd.read_csv(MERGED_FILE_PATH)

    for col in merged_data.columns:
        print(col)

    X_merged, y_merged, record_ids, patient_ids = unfold_merged_data(
        merged_data
    )

    print(record_ids.head())
