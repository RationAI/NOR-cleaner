import numpy as np
import pandas as pd

from data_preparation.column_names import *

TNM_LIST = [T, N, M, PT, PN, PM]


def transform_replace_tnm_with_ptnm(
    df: pd.DataFrame, name: str
) -> pd.DataFrame:
    """
    If PT, PN, PM columns are present, replace T, N, M columns with PT, PN, PM.
    Drop PT, PN, PM columns.

    Parameters:
        df: pd.DataFrame
            The DataFrame with the records.
        name: str
            The name of the column to transform. Either T, N or M.
    """

    if name not in [T, N, M]:
        raise ValueError(f"{name} is not {T}, {N} or {M}")

    if name not in df.columns:
        raise KeyError(f"{name} not in columns")

    df = df.copy()

    # Replace when PT, PN, PM are present
    df[name] = df[name].where(df[f"P{name}"] < 0, df[f"P{name}"])

    df.drop([f"P{name}"], axis=1, inplace=True)

    return df


def transform_tnm(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Transform the T, N, M, PT, PN, PM columns in the DataFrame.
    The transformation is as follows:
    - Replace is.* and [abcm] with 0
    - Replace [0-4].* with [0-4] + 1
    - Replace NaN with -2
    - Replace [xX] with -1

    Parameters:
        df: pd.DataFrame
            The DataFrame with the records.
        name: str
            The name of the column to transform.
    """
    if name not in df.columns:
        raise KeyError(f"{name} not in columns")

    df = df.copy()

    df[name] = df[name].fillna("-2").astype(str)

    # [0-4] -> [0-4] + 1
    df[name] = df[name].str.replace(
        r"^([0-4]).*$", lambda x: str(int(x.group()[0]) + 1), regex=True
    )

    # [abcm] -> 0
    df[name] = df[name].replace(r"^[abcm].*$", "0", regex=True)

    # is.* -> 0
    df[name] = df[name].replace(r"^is.*$", "0", regex=True)

    # [xX] -> -1
    df[name] = df[name].replace(r"^[xX]$", "-1", regex=True)

    assert (
        df[name].isnull().sum() == 0
    ), f"There are still NaN values: {df[name].isnull().sum()}"

    # Set as int
    df[name] = df[name].astype("int64")

    return df


def transform_all_tnm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name in TNM_LIST:
        df = transform_tnm(df, name)

    for name in [T, N, M]:
        df = transform_replace_tnm_with_ptnm(df, name)

    return df


def transform_tnm_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = transform_all_tnm(df)

    df["TNM_Count"] = (
        (df[T] >= 0).astype("int64")
        + (df[N] >= 0).astype("int64")
        + (df[M] >= 0).astype("int64")
    )

    return df


def tnm_stadium_range_known(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = transform_extend_of_disease(df)
    known_disease_extend = df[EXTEND_OF_DISEASE] >= 0

    df = transform_stadium(df)
    known_stadium = df["Stadium"] >= 0

    df["TNM_Count"] = transform_tnm_count(df)["TNM_Count"]
    tnm_known = df["TNM_Count"] == 3

    df["TNM_Stadium_Range_Known"] = (
        known_disease_extend | known_stadium | tnm_known
    ).astype("int64")

    return df


def transform_stadium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the Stadium column.
    Only the first digit is taken into account.
    """
    df = df.copy()
    df[CLINICAL_STADIUM] = df[CLINICAL_STADIUM].str[0].astype("int64")

    unknown_val = 9
    df[CLINICAL_STADIUM] = (
        df[CLINICAL_STADIUM].fillna(unknown_val).astype("int64")
    )

    # 7 means stadium is not given
    # 9 means stadium is not known
    # 6 means metastasis for unknown primary location
    # 6 is set as lowest since most of the records with this value are declined
    df.replace({CLINICAL_STADIUM: {7: -1, 9: -2, 6: -3}}, inplace=True)

    return df


def transform_histology_known(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    UNKNOWN_VAL = 33333
    df[MORPHOLOGY_CODE] = df[MORPHOLOGY_CODE].fillna(UNKNOWN_VAL)
    df[MORPHOLOGY_CODE] = df[MORPHOLOGY_CODE].astype("int64")

    df["KnownHistology"] = (df[MORPHOLOGY_CODE] != UNKNOWN_VAL).astype("int64")

    return df


def transform_grading_known(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    UNKNOWN_VAL = 9

    df[MORPHOLOGY_GRADING] = df[MORPHOLOGY_GRADING].fillna(UNKNOWN_VAL)
    df[MORPHOLOGY_GRADING] = (
        df[MORPHOLOGY_GRADING].astype(float).astype("int64")
    )

    df["Grading_Known"] = (df[MORPHOLOGY_GRADING] != UNKNOWN_VAL).astype(
        "int64"
    )

    return df


def transform_medical_institute_code(data):
    """
    Transform DiagnostikujiciZdravotnickeZarizeniKod to number
    """
    if MEDICAL_INSTITUTE_CODE not in data.columns:
        raise KeyError(f"{MEDICAL_INSTITUTE_CODE} not in columns")

    data[MEDICAL_INSTITUTE_CODE] = data[MEDICAL_INSTITUTE_CODE].astype(
        "string"
    )

    # Other value
    data[MEDICAL_INSTITUTE_TYPE] = 0

    # NOR
    data.loc[
        data[MEDICAL_INSTITUTE_CODE]
        .str[:11]
        .isin(
            [
                "48363057001",
                "27872963001",
                "25903659002",
                "27797660003",
            ]
        ),
        MEDICAL_INSTITUTE_TYPE,
    ] = 2

    to_find = [
        "00064165",
        "00064190",
        "00064203",
        "00064211",
        "00090638",
        "00092584",
        "00098892",
        "00179906",
        "00209805",
        "00226912",
        "00386634",
        "00390780",
        "00511951",
        "00534188",
        "00635162",
        "00669806",
        "00829838",
        "00839205",
        "00842001",
        "00843989",
        "01619748",
        "02239701",
        "03542742",
        "03998878",
        "04315065",
        "04562429",
        "05229723",
        "06199518",
        "08176302",
        "12863297",
        "18235956",
        "24747246",
        "25202171",
        "25202189",
        "25488627",
        "25886207",
        "26000202",
        "26000237",
        "26001551",
        "26068877",
        "26095149",
        "26095157",
        "26095165",
        "26095181",
        "26095190",
        "26095203",
        "26330334",
        "26354250",
        "26360527",
        "26360870",
        "26360900",
        "26361078",
        "26361086",
        "26365804",
        "26376245",
        "26376709",
        "27085031",
        "27253236",
        "27256391",
        "27256456",
        "27256537",
        "27283933",
        "27520536",
        "27661989",
        "27918335",
        "27958639",
        "27968472",
        "27994767",
        "27998878",
        "27998991",
        "29122562",
        "29124263",
        "29158834",
        "29159288",
        "29161487",
        "29299667",
        "45331430",
        "46458085",
        "46885251",
        "47454504",
        "47682795",
        "47697695",
        "47701846",
        "47714913",
        "47813750",
        "62653792",
        "66364183",
        "69969892",
        "69972036",
        "69977020",
        "72047771",
    ]

    data.loc[
        data[MEDICAL_INSTITUTE_CODE].str[:8].isin(to_find),
        MEDICAL_INSTITUTE_TYPE,
    ] = 2

    # ----------------------------

    # KOC -> Complex Oncology Center
    to_find_koc = [
        "00023736",
        "00023884",
        "00064165",
        "00064173",
        "00064190",
        "00064203",
        "00064211",
        "00064238",
        "00068659",
        "00072711",
        "00089915",
        "00090638",
        "00098892",
        "00159816",
        "00159832",
        "00160008",
        "00179906",
        "00190489",
        "00209805",
        "00669806",
        "00673544",
        "00829951",
        "00843989",
        "00844781",
        "25488627",
        "25886207",
        "26068877",
        "26476444",
        "27283933",
        "27520536",
        "27661989",
        "61383082",
        "65269705",
    ]

    data.loc[
        data[MEDICAL_INSTITUTE_CODE].str[:8].isin(to_find_koc),
        MEDICAL_INSTITUTE_TYPE,
    ] = 1

    # Drop column
    data.drop([MEDICAL_INSTITUTE_CODE], axis=1, inplace=True)

    return data


def transform_extend_of_disease(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the `EXTEND_OF_DISEASE` column.
    """
    data = data.copy()
    UNKNOWN_VAL = 9

    data[EXTEND_OF_DISEASE] = (
        data[EXTEND_OF_DISEASE]
        .fillna(UNKNOWN_VAL)
        .astype(float)
        .astype(int)
        .replace(
            {
                # 1, 2 stays the same
                9: -1,
            }
        )
    )

    return data


def transform_distant_metastasis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform column `DISTANT_METASTASIS` to number.
    """
    data = data.copy()
    data[DISTANT_METASTASIS] = (
        data[DISTANT_METASTASIS].fillna(-1).astype("int64")
    )

    return data


def dg_code_divide_into_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide DgKod into three columns:
    - DgKodLetter (0 for C, 1 for D)
    - DgKodNumber (first 2 numbers)
    - DgKodSpecific (last number)

    DgKod is not dropped.
    """

    df = df.copy()

    if "DgKodLetter" in df.columns:
        return

    df["DgKodLetter"] = (
        df["DgKod"]
        .apply(
            # Get first letter
            lambda x: x[0]
        )
        .apply(
            # Encode C as 0 and D as 1
            lambda x: 0 if x == "C" else 1
        )
    )

    df["DgKodNumber"] = (
        df["DgKod"]
        .apply(
            # Get first two numbers
            lambda x: x[1:3]
        )
        .astype("int64")
    )

    df[ICD_SPECIFIC] = (
        df["DgKod"]
        .apply(
            # Get last number
            # If length is not 4, then return -1
            lambda x: x[3] if len(x) == 4 else -1
        )
        .astype("int64")
    )

    # If there is 9 in DgKodSpecific, then it is unspecified
    df[ICD_SPECIFIC] = df[ICD_SPECIFIC].replace({9: -2})

    # Add unspecified localization column
    df[ICD_CODE_RANGE_C76_C80] = (
        df["DgKod"]
        .apply(lambda x: 1 if "C76" <= x < "C81" else 0)
        .astype("int64")
    )

    return df


def transform_dg_to_number(
    df: pd.DataFrame, drop: bool = False
) -> pd.DataFrame:
    """
    Encode DgKod to number:

    First integer is letter, second two are number, last is specific
    If C, then 0, else 1

    Drops DgKod column if `drop` is True.
    """

    df = df.copy()

    TRANSF_NAME = ICD_CODE_NAME + "_Transformed"

    if ICD_CODE_NAME not in df.columns:
        raise KeyError(f"{ICD_CODE_NAME} not in columns")

    df = dg_code_divide_into_cols(df)

    # Combine into one number
    df[TRANSF_NAME] = df["DgKodLetter"] * 1000 + df["DgKodNumber"] * 10

    # Drop columns
    df.drop(
        # Do not drop `ICD_SPECIFIC` column
        ["DgKodLetter", "DgKodNumber"],
        axis=1,
        inplace=True,
    )

    if drop:
        df.drop([ICD_CODE_NAME], axis=1, inplace=True)

    # Rename back to ICD_CODE_NAME
    df.rename(columns={TRANSF_NAME: ICD_CODE_NAME}, inplace=True)

    return df


def transform_topografie_kod(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform topografie kod to a number
    """
    if TOPOGRAPHY_CODE not in data.columns:
        raise KeyError(f"{TOPOGRAPHY_CODE} not in columns")

    # Remove first letter and convert to number
    # All topographies start with "C"
    data[TOPOGRAPHY_CODE] = data[TOPOGRAPHY_CODE].str[1:].astype("int64")

    return data


def transform_lateralita_kod(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform lateralita kod to a number
    """
    if LATERALITY_CODE not in data.columns:
        raise KeyError(f"{LATERALITY_CODE} not in columns")

    # Transform to number
    UNKNOWN_VAL = 9
    data[LATERALITY_CODE] = (
        data[LATERALITY_CODE].fillna(UNKNOWN_VAL).astype("int64")
    )

    # Set 9 to -1
    data.loc[data[LATERALITY_CODE] == UNKNOWN_VAL, LATERALITY_CODE] = -1

    # Set 4 to 0 -- means no laterality for the given tumor
    data.loc[data[LATERALITY_CODE] == 4, LATERALITY_CODE] = 0

    return data


def transform_morfologie_klasifikace_kod(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform MorfologieKlasifikaceKod to two numbers - histology and behavior
    """

    data = data.copy()

    data[MORPHOLOGY_CODE] = data[MORPHOLOGY_CODE].astype(float).astype("int64")

    UNKNOWN_MORPHOLOGY = 33333

    data[MORPHOLOGY_CODE] = data[MORPHOLOGY_CODE].fillna(UNKNOWN_MORPHOLOGY)

    data[MORPHOLOGY_HISTOLOGY] = data[MORPHOLOGY_CODE] // 10
    data[MORPHOLOGY_BEHAVIOR] = data[MORPHOLOGY_CODE] % 10

    data[MORPHOLOGY_HISTOLOGY] = data[MORPHOLOGY_HISTOLOGY].replace(
        {UNKNOWN_MORPHOLOGY // 10: -1}
    )

    # Replace behavior to -2 where morphology is unknown
    data.loc[data[MORPHOLOGY_HISTOLOGY] == -1, MORPHOLOGY_BEHAVIOR] = -2
    # Code 9 means Malignant, uncertain whether primary or metastatic site
    data[MORPHOLOGY_BEHAVIOR] = data[MORPHOLOGY_BEHAVIOR].replace({9: -1})

    # Transform grading
    # Code 9 -- Grade or differentiation not determined,
    # not stated or not applicable
    UNKNOWN_GRADING = 9
    data[MORPHOLOGY_GRADING] = (
        data[MORPHOLOGY_GRADING]
        .fillna(UNKNOWN_GRADING)
        .replace({UNKNOWN_GRADING: -1})
        .astype(float)
        .astype("int64")
    )

    data.drop(columns=MORPHOLOGY_CODE, inplace=True)

    return data


class MorfologieKlasifikaceKodTransformer:
    def __init__(self, data):
        self.data = data.copy()

        self.col_name = "MorfologieKlasifikaceKod"
        assert self.col_name in self.data.columns

        # Get first three digits
        self.transf_name = "MorfologieKlasifikaceKodThreeDigits"
        self.data[self.transf_name] = self.data[self.col_name] // 100

        # Other value
        self.data["hist_typ"] = 0

        # Null value
        self.data.loc[
            (self.data[self.col_name] == 0)
            | (self.data[self.col_name] == 33333),
            "hist_typ",
        ] = -1

    def get_data(self, drop_morfologie_klasifikace_kod: bool = True):
        to_drop = [self.transf_name]
        if drop_morfologie_klasifikace_kod:
            to_drop.append(self.col_name)

        # Drop columns
        self.data = self.data.drop(to_drop, axis=1)

        return self.data

    def clear(self):
        self.data = pd.DataFrame(None)

    def set_hist_typ_ranges(self, ranges, value):
        for r in ranges:
            self.data.loc[
                (self.data[self.transf_name] >= r[0])
                & (self.data[self.transf_name] <= r[1]),
                "hist_typ",
            ] = value

    def set_hist_typ_equals(self, values, value):
        for v in values:
            self.data.loc[self.data[self.transf_name] == v, "hist_typ"] = value


# Morfologie to HistTyp
def transform_morfologie_klasifikace_kod_to_hist_typ(
    data, drop_col: bool = True
):
    """
    Transform MorfologieKlasifikaceKod to number
    """
    transformer = MorfologieKlasifikaceKodTransformer(data)

    # Set to 1
    transformer.set_hist_typ_ranges(
        [(805, 808), (812, 813)],
        1,
    )

    # Set to 2
    transformer.set_hist_typ_ranges(
        [(809, 811)],
        2,
    )

    # Set to 3
    transformer.set_hist_typ_equals(
        [814, 816, 857, 894],
        3,
    )

    transformer.set_hist_typ_ranges(
        [(819, 822), (826, 833), (835, 855)],
        3,
    )

    # Set to 4
    transformer.set_hist_typ_equals([803, 804, 815, 817, 818, 834, 856], 4)

    transformer.set_hist_typ_ranges(
        [(823, 825), (858, 867)],
        4,
    )

    # Set to 5
    transformer.set_hist_typ_equals(
        [801, 802],
        5,
    )

    # Set to 6
    transformer.set_hist_typ_ranges(
        [(868, 871), (880, 892), (915, 925), (954, 958)],
        6,
    )

    transformer.set_hist_typ_equals(
        [899, 904, 912, 913, 937],
        6,
    )

    # Set to 7
    transformer.set_hist_typ_ranges(
        [(959, 972)],
        7,
    )

    # Set to 8
    transformer.set_hist_typ_ranges(
        [(980, 994)],
        8,
    )

    transformer.set_hist_typ_equals(
        [995, 996, 998],
        8,
    )

    # Set to 9
    transformer.set_hist_typ_equals(
        [914],
        9,
    )

    # Set to 10
    transformer.set_hist_typ_equals(
        [905],
        10,
    )

    # Set to 11
    transformer.set_hist_typ_ranges(
        [
            (872, 879),
            (895, 898),
            (900, 903),
            (906, 911),
            (926, 936),
            (938, 953),
            (973, 975),
        ],
        11,
    )

    transformer.set_hist_typ_equals(
        [893, 976],
        11,
    )

    # Set to 12
    transformer.set_hist_typ_equals(
        [800, 997],
        12,
    )

    # Save data
    data = transformer.get_data(drop_morfologie_klasifikace_kod=drop_col)
    transformer.clear()

    return data


def transform_stanoveni_to_num(df: pd.DataFrame) -> pd.DataFrame:
    if CODE_ESTABLISHING_DG not in df.columns:
        raise KeyError(f"{CODE_ESTABLISHING_DG} not in columns")

    df = df.copy()

    df[CODE_ESTABLISHING_DG] = (
        df[CODE_ESTABLISHING_DG]
        .fillna("-1")
        .astype(float)
        .astype(int)
        # 99 means unknown
        .replace({99: -1})
    )

    return df


def transform_stanoveni_to_categories(df: pd.DataFrame) -> pd.DataFrame:
    assert (
        CODE_ESTABLISHING_DG in df.columns
    ), f"{CODE_ESTABLISHING_DG} not in columns"

    suffix = "_Transformed"

    stanoveni_dict: dict[int, str] = {
        0: "KlinickyJasne",
        1: "KlinickeVysetreni",
        2: "LaboratorniVysetreni",
        4: "Cytologie",
        8: "HistologieMetastazy",
        16: "HistologiePrimarniNador",
        32: "Pitva",
        99: "Neznamo",
    }

    # Add suffix
    stanoveni_dict = {k: v + suffix for k, v in stanoveni_dict.items()}

    stanoveni_list = list(stanoveni_dict.items())
    stanoveni_list.sort(key=lambda x: x[0], reverse=True)

    out = df.copy()
    # Add columns
    out[list(stanoveni_dict.values())] = 0

    # For each row, determine the categories
    for i, row in out.iterrows():
        if row[CODE_ESTABLISHING_DG] == "0":
            out.loc[i, "Neznamo"] = 1
            continue

        if row[CODE_ESTABLISHING_DG] == "00":
            out.loc[i, "KlinickyJasne"] = 1
            continue

        stanoveni_int = int(row[CODE_ESTABLISHING_DG])
        for code, col in stanoveni_list:
            if stanoveni_int == 0:
                break

            if stanoveni_int >= code:
                out.loc[i, col] = 1
                stanoveni_int -= code

    # Drop column
    out.drop([CODE_ESTABLISHING_DG], axis=1, inplace=True)

    return out


def transform_date(data: pd.DataFrame, drop: bool = False) -> pd.DataFrame:
    """
    Transform date to a number. 1 is the oldest date, 2 is the second oldest, etc.
    """
    if PATIENT_ID_NAME not in data.columns:
        raise KeyError(f"{PATIENT_ID_NAME} not in columns")

    if DATE_ESTABLISHING_DG not in data.columns:
        raise KeyError(f"{DATE_ESTABLISHING_DG} not in columns")

    data = data.copy()

    # Convert the date column to datetime if it's not already
    data[DATE_ESTABLISHING_DG] = pd.to_datetime(data[DATE_ESTABLISHING_DG])

    data[NOVELTY_RANK] = -1

    # Group by patient and rank entries within each group
    # data.loc[data["AlgoFiltered"] == 0, NOVELTY_RANK] = (
    #     data.loc[data["AlgoFiltered"] == 0, :]
    data[NOVELTY_RANK] = (
        data.groupby(PATIENT_ID_NAME)[DATE_ESTABLISHING_DG]
        .rank(method="average", ascending=True)
        .astype("int64")
    )

    # Drop column
    if drop:
        data.drop([DATE_ESTABLISHING_DG], axis=1, inplace=True)

    return data


def count_unknown_values(
    df: pd.DataFrame, ignore_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Count the number of unknown values in each row.
    The unknown values are values < 0.
    If there are non int values, these are ignored.

    Parameters:
        df: pd.DataFrame
            DataFrame with the data.
        ignore_cols: list[str] | None
            List of columns to ignore. Default is None.
    """
    if ignore_cols is None:
        ignore_cols = []

    cols_to_take = df.select_dtypes(include=[np.number]).columns.drop(
        ignore_cols
    )

    df = df.copy()
    df[UNKNOWN_COUNT] = df[cols_to_take].apply(
        lambda row: sum(row < 0), axis=1
    )

    return df


def count_records_per_patient(data: pd.DataFrame) -> pd.DataFrame:
    if ALGO_FILTERED_COLUMN not in data.columns:
        raise KeyError(f"{ALGO_FILTERED_COLUMN} not in columns")

    data = data.copy()

    rec_count = (
        (data[data[ALGO_FILTERED_COLUMN] == 0].groupby(PATIENT_ID_NAME).size())
        .reset_index()
        .rename({0: RECORD_COUNT_NAME}, axis=1)
    )

    data = data.merge(
        rec_count,
        on=PATIENT_ID_NAME,
        how="left",
    )

    return data


def transform_pn_examination_cols(df: pd.DataFrame) -> pd.DataFrame:
    COL_NAMES = [PN_EXAMINATION, PN_EXAMINATION_POS]

    df = df.copy()
    df[COL_NAMES] = df[COL_NAMES].fillna(-1).astype(float).astype("int64")

    return df


def empty_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace empty strings with NaN values.

    Parameters:
        df: pd.DataFrame
            DataFrame with the data.

    Returns:
        pd.DataFrame
            DataFrame with empty strings replaced with NaN values.
    """
    df = df.copy()

    df.replace("", np.nan, inplace=True)

    return df


def fill_nan_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    to_zero = [
        CREATED_WITH_BATCH,
        TYPE_OF_CARE,
    ]

    df[to_zero] = df[to_zero].fillna(0).astype(float).astype("int64")

    return df


def cols_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the columns to int, if NaN then error
    """
    df = df.copy()

    TO_INT = [
        YEAR_ESTABLISHING_DG,
        TARGET_COLUMN,
        RECORD_ID_NAME,
        PATIENT_ID_NAME,
    ]
    df[TO_INT] = df[TO_INT].astype(float).astype("int64")

    return df


def transform_sentinel_lymph_node(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[SENTINEL_LYMPH_NODE] = (
        df[SENTINEL_LYMPH_NODE].replace("X", -1).fillna(-2).astype("int64")
    )
    return df
