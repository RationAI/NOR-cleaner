"""
File for preparing data for the model
"""

# Parse --which argument from the command line
import argparse
from lib.dataset_names import DATASET_LIST
from data_preparation.data_preparation import prepare_data


parser = argparse.ArgumentParser()
parser.add_argument("--which", type=str, default="2019-2021")
parser.add_argument("--merge", type=bool, default=False)

args = parser.parse_args()
DATASET_TYPE = args.which
MERGE_RECORDS = args.merge

if DATASET_TYPE not in DATASET_LIST:
    raise ValueError(f"Invalid dataset type. Choose from: {DATASET_LIST}")


# Prepare the data
prepare_data(DATASET_TYPE)


# # Merge the records
# if MERGE_RECORDS:
#     import data_preparation.data_merge