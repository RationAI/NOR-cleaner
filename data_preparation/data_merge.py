import datetime
import pickle
from functools import partial
from pathlib import Path

import pandas as pd

import lib.utils
from data_preparation.column_names import ALGO_FILTERED_COLUMN
from data_preparation.merged_transformation import (
    add_cols_equal,
    any_c76_c80_check,
    equality_of_dg_codes,
    feature_difference,
    init_fillna_dict,
    records_equal,
)
from lib.column_names import PREDICTED_COLUMN_ENG
from lib.dataset_names import get_dataset_directory
from lib.parse import get_ready_data
from lib.merge_records import drop_multi_cols, merge_groups_each_row

ID_COLS: list[str] = lib.utils.TO_DROP_IDS.copy()

# DATASET_TYPE = "2022"
DATASET_TYPE = "2019-2021"
# # Test data
# DATASET_TYPE = "verify_dataset"

# Take patients with RecordCount in the range
TAKE_RANGE = (2, 3)
# Number of records that will be merged
N_MERGED = TAKE_RANGE[1]

MERGED_DIR = "merged_data"
TO_SAVE = f"{get_dataset_directory(DATASET_TYPE)}/{MERGED_DIR}"

getter = partial(get_ready_data, which=DATASET_TYPE)
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

y_reduced = X_reduced[PREDICTED_COLUMN_ENG]
X_reduced.drop(columns=PREDICTED_COLUMN_ENG, inplace=True)

print(len(X) - len(X_reduced), "rows removed after filtering by RecordCount.")

# Check if there are any missing values
if X.isna().sum().sum() != 0:
    raise ValueError(
        f"There are missing values in the data: {X.isna().sum().sum()} values."
    )

# Check that there is no 0 in RecordId, since we pad with 0
if 0 in X["RecordId"].values:
    raise ValueError("There is a 0 in RecordId, which should not be the case.")

FILLNA_DICT = init_fillna_dict(X_reduced)

X_merged = merge_groups_each_row(
    df=pd.concat([X_reduced, y_reduced], axis=1),
    group_col="PatientId",
    n=N_MERGED,
    drop_padded_by="RecordCount",
    null_value=FILLNA_DICT["RecordCount"],
    fillna=FILLNA_DICT,
).sort_index()


print("Shape after permuting:", X_merged.shape)


def drop_columns_padded_rows(
    merged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop columns that are the same for each row in the merged data.
    """
    merged_df = merged_df.copy()

    multi_cols = [
        "PatientId",
        "RecordCount",
    ]

    merged_df = drop_multi_cols(merged_df, multi_cols, N_MERGED)

    y_merged = merged_df[PREDICTED_COLUMN_ENG][0].astype(int)
    report_ids = merged_df["RecordId"]
    patient_ids = merged_df["PatientId"][0].copy()

    # Drop column to predict
    merged_df.drop(columns=[PREDICTED_COLUMN_ENG], inplace=True)

    # Transform float columns to int
    float_cols = merged_df.select_dtypes(include=float).columns
    merged_df[float_cols] = merged_df[float_cols].astype(int)

    return merged_df, y_merged, report_ids, patient_ids


X_merged, y_merged, report_ids, patient_ids = drop_columns_padded_rows(
    X_merged
)

print("Shape after dropping columns:", X_merged.shape, y_merged.shape)


DONT_TAKE_COLS_EQUAL_COUNT = [
    "RecordId",
    "PatientId",
    "RecordCount",
    "NoveltyRank",
    "ICD_Equal_With",
    "Any_ICD_C76_C80",
]

COLS_EQUAL_COUNT = [
    col for col, _ in X_merged.columns if col not in DONT_TAKE_COLS_EQUAL_COUNT
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

COLS_FEATURE_DIFF = X_merged.columns.drop(
    [col for col in DONT_TAKE_COLS_FEATURE_DIFF if col in X_merged.columns]
)


def all_merged_transformations(df: pd.DataFrame) -> pd.DataFrame:
    PIPELINE = [
        partial(any_c76_c80_check, n=N_MERGED),
        # partial(equality_of_dg_codes, n=N_MERGED),
        # partial(records_equal, n=N_MERGED),
        partial(add_cols_equal, cols=COLS_EQUAL_COUNT, n=N_MERGED),
        partial(feature_difference, cols=COLS_FEATURE_DIFF, n=N_MERGED),
    ]

    for func in PIPELINE:
        df = func(df)

    return df


X_merged = all_merged_transformations(X_merged)

# Check if there are any missing values
non_null_num = X_merged.isnull().sum().sum()
if non_null_num != 0:
    raise ValueError(f"Number of null values: {non_null_num}")


# Save the data
X_MERGED_FILENAME = "X_merged.pkl"
PREDS_FILENAME = "y_merged.pkl"
REPORT_IDS_FILENAME = "report_ids.pkl"
PATIENT_IDS_FILENAME = "patient_ids.pkl"

Path(TO_SAVE).mkdir(parents=True, exist_ok=True)


def dump_df(df: pd.DataFrame, filename: str) -> None:
    with open(TO_SAVE + f"/{filename}", "wb") as f:
        pickle.dump(df, f)


for df, filename in [
    (X_merged, X_MERGED_FILENAME),
    (y_merged, PREDS_FILENAME),
    (report_ids, REPORT_IDS_FILENAME),
    (patient_ids, PATIENT_IDS_FILENAME),
]:
    dump_df(df, filename)

print_str = f"Saved data to {TO_SAVE}"
print_str += f" at {datetime.datetime.now():%H:%M:%S, %d.%m.%Y}"
print(print_str)
