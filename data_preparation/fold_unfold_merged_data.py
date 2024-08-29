import pandas as pd

from lib.column_names import (
    PATIENT_ID_NAME_ENG,
    RECORD_ID_NAME_ENG,
    TARGET_COLUMN_ENG,
)


def fold_merged_data(
    X_merged: pd.DataFrame,
    y_merged: pd.Series,
    record_ids: pd.DataFrame,
    patient_ids: pd.Series,
) -> pd.DataFrame:
    df = pd.concat([X_merged, y_merged, record_ids, patient_ids], axis=1)
    # Rename columns to NAME_IDX
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]

    return df


def unfold_merged_data(
    merged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    # Transform columns to multiindex
    merged_df.columns = pd.MultiIndex.from_tuples(
        [tuple(col.rsplit("_", 1)) for col in merged_df.columns]
    )

    y_merged = merged_df[TARGET_COLUMN_ENG]
    record_ids = merged_df[RECORD_ID_NAME_ENG]
    patient_ids = merged_df[PATIENT_ID_NAME_ENG]

    X_merged = merged_df.drop(
        columns=[TARGET_COLUMN_ENG, RECORD_ID_NAME_ENG, PATIENT_ID_NAME_ENG],
        level=0,
    )

    return X_merged, y_merged, record_ids, patient_ids
