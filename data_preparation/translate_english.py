"""
Translate Columns to English
"""

import pandas as pd


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

    translation_dict = {
        "HlaseniIdDtb": "RecordId",
        "PacientId": "PatientId",
        "StanoveniDgKod": "CodeEstablishingDg",
        "PNVysetreni": "PNExamination",
        "PNVysetreniPozitivnich": "PNExaminationPos",
        "SentinelovaMizniUzlinaKod": "SentinelLymphNode",
        "MorphologyHistologyCode": "MorphHistology",
        "MorphologyBehaviorCode": "MorphBehavior",
        "MorfologieGradingKod": "MorphGrading",
        "TypPece": "TypeOfCare",
        "RokStanoveniDg": "YearDg",
        "ZalozenoDavkou": "CreatedWithBatch",
        "vyporadani_final": "FinalDecision",
        "Stadium_Transformed": "ClinicalStadium",
        "OnemocneniKod_Transformed": "ExtendOfDisease",
        "VzdalenaMetastaze_Transformed": "DistantMetastasis",
        "DgKod_C76_C80": "ICDRangeC76-C80",
        "DgKod_Encoded": "ICD",
        "DgKod_Specific_Encoded": "ICDLoc",
        "TopografieKod_Encoded": "Topography",
        "LateralitaKod_Transformed": "Laterality",
        "DiagnostikujiciZZTyp": "MedicalInstituteType",
        "T_Encoded": "T",
        "N_Encoded": "N",
        "M_Encoded": "M",
        "PT_Encoded": "PT",
        "PN_Encoded": "PN",
        "PM_Encoded": "PM",
        "DatumStanoveniDg_Encoded": "NoveltyRank",
        "RecordCount": "RecordCount",
        "AlgoFiltered": "AlgoFiltered",
        "UnknownCount": "UnknownCount",
    }

    df.columns = [translation_dict[col] for col in df.columns]

    return df
