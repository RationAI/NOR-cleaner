from typing import Literal


DATA_DIR = "data"
ORIGINAL_DATASET_FILENAME = "data_by_expert.csv"
DATA_PREPROCESSED_FILENAME = "data_preprocessed.pkl"

"""
DatasetType
    Type of dataset. It can be one of the following:
        - "2019-2021"      -- Dataset of records from 2019 to 2021.
        - "2022"           -- Dataset records from 2019 to 2022.
        - "verify_dataset" -- Dataset of new records to test the model.
"""
DatasetType = Literal["2019-2021", "2022", "verify_dataset"]
DATASET_LIST = ["2019-2021", "2022", "verify_dataset"]


def get_dataset_directory(which: DatasetType) -> str:
    """
    Get the directory of the dataset.

    Parameters:
        which: DatasetType
            Which dataset to get.

    Returns:
        str:
            Directory of the dataset.
    """
    return f"{DATA_DIR}/{which}"