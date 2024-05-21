#!/usr/bin/env python
# coding: utf-8

"""
File for data preparation
"""


import datetime
import pickle
from functools import partial

import pandas as pd

from data_preparation.drop_columns import drop_columns
from data_preparation.feature_transformation import (
    cols_to_int,
    count_records_per_patient,
    count_unknown_values,
    dg_code_divide_into_cols,
    transform_dg_to_number,
    transform_distant_metastasis,
    transform_extend_of_disease,
    fill_nan_to_zero,
    transform_sentinel_lymph_node,
    transform_all_tnm,
    transform_date,
    transform_lateralita_kod,
    transform_medical_institute_code,
    transform_morfologie_klasifikace_kod,
    transform_pn_examination_cols,
    transform_stadium,
    transform_stanoveni_to_categories,
    transform_stanoveni_to_num,
    transform_topografie_kod,
)
from lib.dataset_names import DATA_PREPROCESSED_FILENAME, DATA_DIR
from data_preparation.translate_english import df_english_translation
from lib.algo_filtering import algorithmic_filtering_icd_10
from lib.column_names import PREDICTED_COLUMN_ENG
from lib.parse import get_original_dataset


DATASET_TYPE = "2019-2021"
# DATASET = "2022"
# DATASET = "verify_dataset"


if DATASET_TYPE != "verify_dataset":
    data = get_original_dataset(which=DATASET_TYPE)
else:
    # Unpickle completely new data
    data = pd.read_pickle("data/verify_dataset/data_new_records.pkl")

# Set the flag to use the algorithmic data

print("Data by expert shape:", data.shape)


# Drop Rows
# Range C81-C96, D45-D47
def drop_icd_in_range(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    df = df.copy()

    icd_three = df["DgKod"].str[:3]
    df = df[~icd_three.between(a, b)]

    return df


do_drop_icd_ranges = False

if do_drop_icd_ranges:
    # Drop ICD codes that are not in the range of interest
    data = drop_icd_in_range(data, "C81", "C96")
    data = drop_icd_in_range(data, "D45", "D47")

    print("Data by expert shape after dropping ICD codes:", data.shape)


data, dropped_record_ids = algorithmic_filtering_icd_10(data)
print("Number of algorithmically filtered records:", len(dropped_record_ids))

# Division into Columns

do_dg_code_divide_into_cols = False
# do_dg_code_divide_into_cols = True

if do_dg_code_divide_into_cols:
    data = dg_code_divide_into_cols(data)

# StanoveniDgKod to Categories

do_transform_stanoveni_to_categories = False
# do_transform_stanoveni_to_categories = True

if do_transform_stanoveni_to_categories:
    data = transform_stanoveni_to_categories(data)


def all_transformations(df: pd.DataFrame) -> pd.DataFrame:
    PIPELINE = [
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
        partial(transform_date, drop=True),
        count_unknown_values,
        count_records_per_patient,
        df_english_translation,
    ]

    df = df.copy()
    for f in PIPELINE:
        df = f(df)

    return df


data = all_transformations(data)


# Data are ready

# Move vyporadani_final to the end
cols = list(data.columns)
cols.remove(PREDICTED_COLUMN_ENG)

if "by_expert" in cols:
    cols.remove("by_expert")
    cols.append("by_expert")

cols.append(PREDICTED_COLUMN_ENG)
data = data[cols]

# Check there are no null values
if data.isnull().sum().sum() != 0:
    raise ValueError("There are still null values")

print("Shape after all transformations:", data.shape)
print(data.shape)

# Check that there are no object columns
if not data.select_dtypes(include="object").columns.empty:
    raise ValueError(
        f"Object columns: {data.select_dtypes(include='object').columns}"
    )


# Save the data
data_preprocessed = f"{DATA_DIR}/{DATASET_TYPE}/{DATA_PREPROCESSED_FILENAME}"

with open(data_preprocessed, "wb") as f:
    pickle.dump(data, f)
    print_str = f"Saved data to {data_preprocessed}"
    print_str += f" at {datetime.datetime.now():%H:%M:%S, %d.%m.%Y}"
    print(print_str)
