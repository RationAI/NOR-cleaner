"""
File for preparing data for the model
"""

# Parse --which argument from the command line
import argparse
import logging

from data_preparation.data_merge import prepare_merged_data
from data_preparation.preprocess_data import preprocess_data
from lib.dataset_names import DATASET_LIST, DatasetType

# Set up the logger
# Format: LEVEL:__name__:TIME:MESSAGE
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(asctime)s:%(message)s"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--which",
    type=str,
    default="2019-2021",
    help=f"Dataset type to prepare, options: {DATASET_LIST}",
)
# Optional arguments
parser.add_argument(
    "--merge",
    action=argparse.BooleanOptionalAction,
    help="Whether to merge the records after preparation",
    default=True,
)

parser.add_argument(
    "--skip-prepare",
    action=argparse.BooleanOptionalAction,
    help="Skip data preparation before merging",
    default=False,
)


args = parser.parse_args()
DATASET_TYPE: DatasetType = args.which
MERGE = args.merge
SKIP_PREPARE = args.skip_prepare

if DATASET_TYPE not in DATASET_LIST:
    raise ValueError(f"Invalid dataset type. Choose from: {DATASET_LIST}")


# Prepare the data
if not SKIP_PREPARE:
    preprocess_data(DATASET_TYPE)
else:
    print("Skipping data preparation.")


# Merge the records
if MERGE:
    prepare_merged_data(DATASET_TYPE)
else:
    print("Skipping merging the records.")
