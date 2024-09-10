"""
Translate Columns to English
"""

import pandas as pd

from data_preparation.column_names import *
from lib.column_names import *

# Translation dictionary for the columns from Czech to English
COLS_TO_ENG_DICT = {
    PATIENT_ID_NAME: PATIENT_ID_NAME_ENG,
    RECORD_ID_NAME: RECORD_ID_NAME_ENG,
    CODE_ESTABLISHING_DG: CODE_ESTABLISHING_DG_ENG,
    PN_EXAMINATION: PN_EXAMINATION_ENG,
    PN_EXAMINATION_POS: PN_EXAMINATION_POS_ENG,
    SENTINEL_LYMPH_NODE: SENTINEL_LYMPH_NODE_ENG,
    # "MorphologyHistologyCode": "MorphHistology",
    # "MorphologyBehaviorCode": "MorphBehavior",
    # "MorfologieGradingKod": "MorphGrading",
    TYPE_OF_CARE: TYPE_OF_CARE_ENG,
    YEAR_ESTABLISHING_DG: YEAR_ESTABLISHING_DG_ENG,
    CREATED_WITH_BATCH: CREATED_WITH_BATCH_ENG,
    CLINICAL_STADIUM: CLINICAL_STADIUM_ENG,
    EXTEND_OF_DISEASE: EXTEND_OF_DISEASE_ENG,
    DISTANT_METASTASIS: DISTANT_METASTASIS_ENG,
    # "DgKod_C76_C80": "ICDRangeC76-C80",
    # "DgKod_Transformed": ICD_CODE_NAME_ENG,
    # "DgKod_Specific_Encoded": "ICDLoc",
    TOPOGRAPHY_CODE: TOPOGRAPHY_CODE_ENG,
    LATERALITY_CODE: LATERALITY_CODE_ENG,
    MEDICAL_INSTITUTE_TYPE: MEDICAL_INSTITUTE_TYPE_ENG,
    # "T": "T",
    # "N": "N",
    # "M": "M",
    NOVELTY_RANK: NOVELTY_RANK,
    # "DatumStanoveniDg": "DateOfEstablishingDg",
    ALGO_FILTERED_COLUMN: ALGO_FILTERED_COLUMN,
    RECORD_COUNT_NAME: RECORD_COUNT_NAME,
    UNKNOWN_COUNT: UNKNOWN_COUNT,
    TARGET_COLUMN: TARGET_COLUMN_ENG,
}


def df_english_translation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate the columns of the DataFrame to English.

    Parameters:
        df: pd.DataFrame
            The DataFrame to translate.

    Returns:
        pd.DataFrame
            The DataFrame with the columns translated to English.
    """

    df = df.copy()

    # If a column is not in the dict, raise an error
    for col in df.columns:
        if col not in COLS_TO_ENG_DICT:
            raise ValueError(f"Column {col} not in `COLS_TO_ENG_DICT`")

    df.columns = [COLS_TO_ENG_DICT[col] for col in df.columns]

    return df
