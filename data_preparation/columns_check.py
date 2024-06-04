"""
Check whether all columns necessary for the data preparation are present in the dataset.
"""

import pandas as pd
from lib.column_names import *


def check_columns(df: pd.DataFrame) -> None:
    """
    Check if all necessary columns are present in the DataFrame.
    Otherwise, raise a ValueError.

    Parameters:
        df: pd.DataFrame
            The DataFrame to check.
    """
    COLUMNS = [
        PATIENT_ID_NAME,
        RECORD_ID_NAME,
        ICD_CODE_NAME,
        DATE_ESTABLISHING_DG,
        CODE_ESTABLISHING_DG,
        PN_EXAMINATION,
        PN_EXAMINATION_POS,
        SENTINEL_LYMPH_NODE,
        MORPHOLOGY_CODE,
        MORPHOLOGY_GRADING,
        TYPE_OF_CARE,
        YEAR_ESTABLISHING_DG,
        CREATED_WITH_BATCH,
        CLINICAL_STADIUM,
        EXTEND_OF_DISEASE,
        DISTANT_METASTASIS,
        TOPOGRAPHY_CODE,
        LATERALITY_CODE,
        MEDICAL_INSTITUTE_CODE,
        T,
        N,
        M,
        PT,
        PN,
        PM,
        PREDICTED_COLUMN,
    ]

    missing_columns = [col for col in COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. Add them to the dataset."
        )
