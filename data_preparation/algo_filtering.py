"""
Algorithmical filtering of the records.
Records of one patient are grouped together and the records where
the ICD-10 code matches the given code are filtered out based on a scoring
function.
"""

import pandas as pd
from tqdm import tqdm

import data_preparation.feature_transformation as data_prep
from data_preparation.column_names import MEDICAL_INSTITUTE_TYPE
from lib.column_names import (
    ICD_CODE_NAME,
    PATIENT_ID_NAME,
    RECORD_ID_NAME,
    TARGET_COLUMN,
    TYPE_OF_CARE,
)

# Path to the file with ICD-10 ranges
ICD_RANGES_FILE = "data_preparation/algo_icd_ranges.txt"


def algorithmic_filtering_icd_10(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Filter the records of a patient where the ICD-10 code is duplicated.
    The ICDs are filtered based on the given ranges from ICD_RANGES_FILE.

    Parameters:
        df: pd.DataFrame
            The DataFrame with the records.

    Returns:
        pd.DataFrame
            The filtered DataFrame.
    """
    df = df.copy()
    to_drop_after = [
        "icd_three",
        "scores",
    ]

    df["icd_three"] = df[ICD_CODE_NAME].str[:3]
    df["scores"] = _give_scores(df)

    assert df["scores"].isna().sum() == 0

    icd_range_3, icd_range_4 = process_icd_ranges_file(ICD_RANGES_FILE)

    # First filter the records with ICD-10 codes with 3 characters
    to_drop_records, _ = _drop_records_with_lower_score(
        df, icd_range_3, icd_range_4
    )

    # Tag the records which were filtered
    df["AlgoFiltered"] = 0
    df.loc[df[RECORD_ID_NAME].isin(to_drop_records), "AlgoFiltered"] = 1

    # Update the final predictions
    df = _update_prediction(df)

    df.drop(
        columns=to_drop_after,
        inplace=True,
    )
    return df, to_drop_records


def _drop_records_with_lower_score(
    df: pd.DataFrame,
    icd_range_3: set[str],
    icd_range_4: set[str],
) -> tuple[list[int], list[int]]:
    """
    Drop the records with the lower score based on the ICD-10 code ranges.

    Parameters:
        df: pd.DataFrame
            The DataFrame with the records.

        icd_range_3: set[str]
            The set of ICD-10 codes with 3 characters.

        icd_range_4: set[str]
            The set of ICD-10 codes with 4 characters.

    Returns:
        tuple[list[int], list[int]]
            The list of record IDs to drop and the list of record IDs left out.
    """
    to_drop: list[int] = []

    record_ids_left_out: list[int] = []

    for _, group in tqdm(
        df.groupby(PATIENT_ID_NAME), desc="Filtering records"
    ):
        # Get unique ICD-10 codes
        codes_three = set(group["icd_three"].unique())
        codes_four = set(group[ICD_CODE_NAME].unique())

        # For each code, get the best score and drop the records with lower score
        for code in codes_three:
            if code in icd_range_3:
                _filter_by_score(
                    group, record_ids_left_out, to_drop, "icd_three", code
                )

        for code in codes_four:
            if code in icd_range_4:
                _filter_by_score(
                    group, record_ids_left_out, to_drop, ICD_CODE_NAME, code
                )

    return to_drop, record_ids_left_out


def _filter_by_score(
    group: pd.DataFrame,
    ids_left_out: list[int],
    to_drop: list[int],
    column: str,
    code: str,
) -> None:

    icd_group = group[group[column] == code]

    best_score = icd_group["scores"].max()
    best_score_id = icd_group[icd_group["scores"] == best_score][
        RECORD_ID_NAME
    ].values[0]

    to_add_to_drop = icd_group[
        (icd_group[RECORD_ID_NAME] != best_score_id)
        & (icd_group["scores"] <= best_score)
    ][RECORD_ID_NAME].tolist()

    to_drop.extend(to_add_to_drop)

    # Set as accepted afterwards
    if len(to_add_to_drop) > 0:
        ids_left_out.append(best_score_id)


def _give_scores(df: pd.DataFrame) -> pd.Series:
    """
    Give scores to the records based on the scoring table.
    This is the scoring:
    + 10**5 when morphological code is present
    + 10**4 when all of T, N, M codes are present
    + 10**3 * tnm_count, where tnm_count is the number of T, N, M codes present
    + 10**2 when grading code is present
    + 10**1 * MedicalInstituteType
    + 10**0 * TypeOfCare

    Parameters:
        df: pd.DataFrame
            The original DataFrame with the records.

    Returns:
        pd.Series
            The Series with the scores.
    """
    df_for_score = _df_transform_for_scoring(df)

    df_for_score[TYPE_OF_CARE] = (
        df_for_score[TYPE_OF_CARE].fillna(0).astype(float).astype("int64")
    )

    predicted_column_score = (
        df_for_score[TARGET_COLUMN]
        if TARGET_COLUMN in df_for_score.columns
        else 0
    )

    # Give scores based on the scoring table
    score = (
        df_for_score["KnownHistology"] * 10**6
        + df_for_score["TNM_Stadium_Range_Known"] * 10**5
        + df_for_score["TNM_Count"] * 10**4
        + df_for_score["Grading_Known"] * 10**3
        + df_for_score[MEDICAL_INSTITUTE_TYPE] * 10**2
        + df_for_score[TYPE_OF_CARE] * 10**1
        + predicted_column_score * 10**0
    )

    return score


def _df_transform_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the DataFrame for the scoring function.
    """
    pipeline = [
        data_prep.transform_histology_known,
        data_prep.tnm_stadium_range_known,
        data_prep.transform_tnm_count,
        data_prep.transform_grading_known,
        data_prep.transform_medical_institute_code,
    ]

    for func in pipeline:
        df = func(df)

    return df


def process_icd_ranges_file(filename: str) -> tuple[set[str], set[str]]:
    """
    Process the ICD ranges file and return a set of ICD codes.

    Parameters:
        filename: str
            The path to the file with ICD ranges.

    Returns:
        set[str]
            The set of ICD codes.
    """

    icd_range_3 = set()
    icd_range_4 = set()

    with open(filename, "r") as f:
        for line in f:
            icd_code = line.strip()
            if len(icd_code) == 3:
                icd_range_3.add(icd_code)
            else:
                if len(icd_code) != 4:
                    raise ValueError(
                        "ICD code has to be 3 or 4 characters long. Found:",
                        icd_code,
                    )
                icd_range_4.add(icd_code)

    return icd_range_3, icd_range_4


def _update_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the final predictions based on the algorithmical filtering.
    AlgoFiltering column must be present in the DataFrame.
    If AlgoFiltering == 1 and PREDICTED_COLUMN == 1, then set
    the PREDICTED_COLUMN for the record which was not filtered to 1
    to preserve the original prediction of the record.

    Parameters:
        df: pd.DataFrame
            The DataFrame with the records.

    Returns:
        pd.DataFrame
            The updated DataFrame.
    """

    assert "icd_three" in df.columns

    # Make a copy of the DataFrame
    df = df.copy()

    patient_icd_filtered = df[df["AlgoFiltered"] == 1][
        [PATIENT_ID_NAME, ICD_CODE_NAME]
    ]
    seen = set()

    for _, row in patient_icd_filtered.iterrows():
        patient_id = row[PATIENT_ID_NAME]
        icd_code = row[ICD_CODE_NAME]
        if (patient_id, icd_code) in seen:
            continue

        seen.add((patient_id, icd_code))

        # Get the record with the same patient ID and ICD code
        # where AlgoFiltered == 0
        record = df[
            (df[PATIENT_ID_NAME] == patient_id)
            & (df[ICD_CODE_NAME] == icd_code)
            & (df["AlgoFiltered"] == 0)
        ]

        # If not found, check for the first three ICD-10 symbols
        if record.shape[0] == 0:
            record = df[
                (df[PATIENT_ID_NAME] == patient_id)
                & (df["icd_three"] == icd_code[:3])
                & (df["AlgoFiltered"] == 0)
            ]

            if record.shape[0] == 0:
                raise ValueError(
                    "No record found for the patient with the same ICD code and AlgoFiltered == 0."
                )

        if record.shape[0] != 1:
            raise ValueError(
                f"The dataset has duplicated ICD-10 after algorithmic filtering: {patient_id}, {icd_code}"
            )

        # Set the PREDICTED_COLUMN to 1
        df.loc[record.index, TARGET_COLUMN] = 1

    return df
