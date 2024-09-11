"""
This file contains the constants used in the scripts.
"""

from pathlib import Path

# Set path to the raw data
# By default, the data is stored in the data/ directory
DATASET_PATH: Path = Path("data")

SAVE_PREPARED_DATA: Path = Path("data", "prepared_data.csv")
SAVE_MERGED_DATA: Path = Path("data", "merged_data.csv")

# Take patients with RecordCount in the range [a, b] (inclusive)
# In this case, we take patients with 2 or 3 records
TAKE_RANGE: tuple[int, int] = (2, 3)
