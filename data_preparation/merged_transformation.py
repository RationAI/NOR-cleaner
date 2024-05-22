from typing import Any
import pandas as pd


def init_fillna_dict(df: pd.DataFrame) -> dict[str, Any]:

    get_min = lambda col: df[col].min()
    FILLNA_DICT = {
        "RecordId": 0,
        "PatientId": 0,
        "CodeEstablishingDg": -1,
        "SentinelLymphNode": get_min("SentinelLymphNode"),
        "TypeOfCare": -1,
        "CreatedWithBatch": -1,
        "ClinicalStadium": get_min("ClinicalStadium"),
        "ExtendOfDisease": -1,
        "DistantMetastasis": -1,
        "ICDRangeC76-C80": -1,
        "ICD": -1,
        "ICDLoc": get_min("ICDLoc"),
        "Topography": -1,
        "Laterality": -1,
        "MedicalInstituteType": -1,
        "T": get_min("T"),
        "N": get_min("N"),
        "M": get_min("M"),
        "NoveltyRank": -1,
        "YearDg": -1,
        "FinalDecision": -1,
        "PNExamination": -1,
        "PNExaminationPos": -1,
        "MorphHistology": get_min("MorphHistology"),
        "MorphBehavior": get_min("MorphBehavior"),
        "MorphGrading": get_min("MorphGrading"),
        "RecordCount": -1,
        "UnknownCount": -1,
    }

    # Check that all columns from the DataFrame are in the FILLNA_DICT
    for col in df.columns:
        if col not in FILLNA_DICT:
            raise ValueError(f"Column {col} not in FILLNA_DICT")

    return FILLNA_DICT


def any_c76_c80_check(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    If True, a column is created which checks
    if any of the C76-C80 is present in the row.

    Parameters:
        df: pd.DataFrame
            DataFrame with the data.

        n: int
            Number of columns in a row.
    """
    df = df.copy()

    def _check(x):
        for i in range(n):
            if x[i] == 1:
                return 1
        return 0

    df["Any_ICD_C76_C80", 0] = df["ICDRangeC76-C80"].apply(_check, axis=1)

    return df


def equality_of_dg_codes(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    If True, columns "ICD_Equal_With_i" are created which check
    if the first ICD is equal to i-th ICD.

    Parameters:
        df: pd.DataFrame
            DataFrame with the data.

        n: int
            Number of columns in a row.
    """
    df = df.copy()

    def check_equality_dg_codes(x, i):
        return int(x[0] == x[i])

    for i in range(1, n):
        df[f"ICD_Equal_With", i] = df["ICD"].apply(
            lambda x: check_equality_dg_codes(x, i), axis=1
        )

    return df


def records_equal(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.copy()

    def check_equality_reports(cols_to_take, x, i):
        for col in cols_to_take:
            if x.loc[col, 0] != x.loc[col, i]:
                return 0

        return 1

    cols_to_take = []
    ignore_cols = [
        "NoveltyRank",
        "RecordId",
        "PatientId",
        "RecordCount",
    ]
    for col, i in df.columns:
        if i == 0 and col not in ignore_cols:
            cols_to_take.append(col)

    for i in range(1, n):
        df[f"Report_Equal_With", i] = df.apply(
            lambda x: check_equality_reports(cols_to_take, x, i), axis=1
        )

    return df


def add_cols_equal(df: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    """
    If True, columns "Row_Similarity_i" are created which check
    how are each records similar to the first record. It is calculated
    as the number of equal values divided by the length of `cols`.

    Parameters:
        df: pd.DataFrame
            DataFrame with the data.

        cols: list[str]
            List of columns to check.

        n: int
            Number of columns in a row.
    """
    df = df.copy()

    def row_similarity(x, i):
        vals_0 = [x[col, 0] for col in cols]
        vals_i = [x[col, i] for col in cols]
        vals_equal = [val_0 == val_i for val_0, val_i in zip(vals_0, vals_i)]

        # Sum up the equal values
        equal = pd.Series(0, index=df.index)
        for equal_vals in vals_equal:
            equal += equal_vals.astype(int)

        return equal / len(cols)

    for i in range(1, n):
        df[f"Row_Similarity", i] = row_similarity(df, i)

    return df


def feature_difference(df: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    """
    Subtracts the i-th column from the first column.
    Add columns `colname_Diff_i` for each column `colname_i` in `cols`.
    It is calculated as `colname_0 - colname_i`.
    
    Parameters:
        df: pd.DataFrame
            DataFrame with the data.

        cols: list[str]
            List of columns to check.

        n: int
            Number of columns in a row.
    """
    df = df.copy()

    for colname, idx in cols:
        if idx != 0:
            continue

        for i in range(1, n):
            if (colname, i) not in df.columns:
                break

            new_column = f"{colname}_Diff"
            df[new_column, i] = (
                df[colname][0]
                - df[colname][i]
            )

    return df
