""" 
Drop Columns
"""

import pandas as pd

# fmt: off
NULL_COLUMNS = [
    # Column name, # Percentage of null values
    "RizikovaKategorieKod",                         # 100.00
    "HistologieKlasifikaceKod",                     # 100.00
    "ZjednoduseneTnm",                              # 100.00
    "DuvodZmeny",                                   # 100.00
    "DiagnostickaSkupinaKod",                       # 100.00
    "DgRevizeMkn",                                  # 100.00
    "CytologieKlasifikaceKod",                      # 100.00
    "DiagnostickoSpecifStadium",                    #  99.89
    "SerovyNadorovyMarkerKod",                      #  99.74
    "OdpovedNaLecbu",                               #  99.65
    "LecbaNovotvaru",                               #  99.16
    "DiagnostickaSkupina",                          #  99.03
    "Py",                                           #  97.58
    "P16PozitivniKod",                              #  96.71
    "SpecifickaDg",                                 #  96.20
    "IdentifikaceZaznamuZeZdravotnickehoZarizeni",  #  96.15
]

# These columns are redundant or not important
REDUNDANT_COLUMNS = [
    "DgNazev",  # DgKod already there
    "NadorZrusen",
    "StatKod",  # Not sure if needed

    # "DgKod",
    # "TopografieKod", # This corresponds to the ICD-O-3 code (MKN-O-3 in Czech)

    # "HlasiciLekarDiagnostickeCasti", # Not sure about this one
    "Poznamka",  # Same as with NadorId
    "MorfologieNazev",  # There is morfology classif. code
    "Cizinec",  # Not sure how important
    "Bezdomovec",  # Not sure how important
    "vyporadani_kat",  # In our case it is always "vypořádáno expertně"
    "StavVyporadano", # filled after the decision of the report

    # Columns which are not important for acceptance
    "StavUplne",
    "ICCCKodDiagnozy",
    "RokNar",
    "PohlaviKod",
    "VekPriDg",
    "BydlisteObecKod",
    "BydlisteOrp",
    "BydlisteOkres",
    "BydlisteKraj",
    "IncidenceObec",
    "IncidenceOrp",
    "IncidenceOkres",
    "IncidenceKraj",
    "VekPriDg",
    "EvidencniCislo",
    "EvidencniCisloTisk",
    "TnmSkupina",
    "ObecIncidenceKod",
    # "StanoveniDgKod",
    "HlasiciLekarDiagnostickeCasti",
    # "OnemocneniKod",
    # "PNVysetreni",
    # "PNVysetreniPozitivnich",

    # Column which could be important
    "MorfologieTyp",                    # Mostly category "H"
    # "ZalozenoDavkou",

    "DiagnostikujiciOddeleniKod",
    # "DiagnostikujiciZdravotnickeZarizeniKod", # Do not drop, transformed to a number

    # "Stadium",
]

# Time columns which are not important for accePTmodmodmodmodmodance
TIME_COLUMNS = [
    "DatumHlaseniDiagnostickeCasti",
    # "DatumStanoveniDg",

    "DatumVyporadani",  # Same as NadorId -> filled afterwards
    "VytvorilKdo",
    "VytvorilKdy",
    "ZmenilKdo",
    "ZmenilKdy",

    # Maybe important
    # "RokStanoveniDg",
    "MesicStanoveniDg",
]

ID_COLUMNS = [
    # "HlaseniIdDtb",
    # "PacientId",
    "NadorId",  # Strong indicator of correctness
    # ^^ Cannot be used as it is filled after the decision of the report
]

# Columns that were added in the year 2022
YEAR_2022_COLS = [
    "ZmenilKdyVerze",
]
# fmt: on


def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = (
        NULL_COLUMNS
        + REDUNDANT_COLUMNS
        + list(set(ID_COLUMNS) - set(["PacientId"]))
        + TIME_COLUMNS
        + [col for col in YEAR_2022_COLS if col in data.columns]
    )

    # Do not drop PacientId since it is needed for grouping
    data_columns_dropped = data.drop(
        cols_to_drop,
        axis=1,
    )

    print("Shape after dropping columns:", data_columns_dropped.shape)
    return data_columns_dropped
