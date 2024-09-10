import pandas as pd

from lib.column_names import PATIENT_ID_NAME, RECORD_ID_NAME, TARGET_COLUMN


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

    # Set the right index to int
    merged_df.columns = pd.MultiIndex.from_tuples(
        [(name, int(idx)) for name, idx in merged_df.columns]
    )

    y_merged = merged_df[TARGET_COLUMN]
    record_ids = merged_df[RECORD_ID_NAME]

    patient_ids = merged_df[PATIENT_ID_NAME]
    patient_ids = pd.Series(patient_ids.iloc[:, 0], name=PATIENT_ID_NAME)

    X_merged = merged_df.drop(
        columns=[TARGET_COLUMN, RECORD_ID_NAME, PATIENT_ID_NAME],
        level=0,
    )

    return X_merged, y_merged, record_ids, patient_ids
