"""
This file contains the constants used in the scripts.
Mainly, the paths to the data files.
"""

from pathlib import Path

# The data are stored in the data/ directory

# Set path to the raw data
DATASET_PATH: Path = Path(
    # INSERT THE PATH TO THE RAW DATA
    "data/PATH/TO/DATA.csv"
)
# The path to the file where the prepared data will be saved
PREPARED_DATA_PATH: Path = Path(
    # INSERT THE PATH TO THE PREPARED DATA
    "data/PATH/TO/PREPARED_DATA.csv"
)
# The path to to save the merged data -> the final dataset for training
MERGED_DATA_PATH: Path = Path(
    # INSERT THE PATH TO THE MERGED DATA
    "data/PATH/TO/MERGED_DATA.csv"
)
# The path to the file where the model will be saved or loaded from
MODEL_PATH: Path = Path(
    # INSERT THE PATH TO THE MODEL
    "data/PATH/TO/MODEL.json"
)
# The path to the file with new raw data to predict on
PREDICT_DATA_PATH: Path = Path(
    # INSERT THE PATH TO THE RAW DATA TO PREDICT
    "data/PATH/TO/PREDICT_DATA.csv"
)
# The path to the file where the predictions will be saved
PREDICTIONS_PATH: Path = Path(
    # INSERT THE PATH TO THE PREDICTIONS
    "data/PATH/TO/PREDICTIONS.csv"
)

# Take patients with RecordCount in the range [a, b] (inclusive)
# In this case, we take patients with 2 or 3 records
TAKE_RANGE: tuple[int, int] = (2, 3)
