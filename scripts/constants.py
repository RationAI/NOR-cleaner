"""
This file contains the constants used in the scripts.
"""

from pathlib import Path

# Set path to the raw data
# By default, the data is stored in the data/ directory
DATASET_PATH: Path = Path("data/tmp_test/data_by_expert.csv")
# The path to the file with prepared data
SAVE_PREPARED_DATA: Path = Path("data", "tmp_test", "prepared_data.csv")
# The path to the file with merged data -> the final dataset
SAVE_MERGED_DATA: Path = Path("data", "tmp_test", "merged_data.csv")

# Take patients with RecordCount in the range [a, b] (inclusive)
# In this case, we take patients with 2 or 3 records
TAKE_RANGE: tuple[int, int] = (2, 3)
