"""
Check whether all columns necessary for the data preparation are present in the dataset.
"""

import pandas as pd
from lib.column_names import REQUIRED_COLUMNS


def check_columns(df: pd.DataFrame) -> None:
    """
    Check if all necessary columns are present in the DataFrame.
    Otherwise, raise a ValueError.

    Parameters:
        df: pd.DataFrame
            The DataFrame to check.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS.values() if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. Add them to the dataset."
        )
