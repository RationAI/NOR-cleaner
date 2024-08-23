# NOR-cleaner
AI model for resolving duplicities in Czech National Oncology Register.
Bachelor's thesis about this project is available [here](https://is.muni.cz/th/lkudy/development_of_data_deduplication_model_nor.pdf).

## Requirements
- Python 3.11 or newer
- libraries listed in `requirements.txt`
  - `pip install -r requirements.txt`

## Usage
### Data preparation
1. Import data to directory `data/`:

    a) Directory `data/2019-2021/` contains data from years 2019-2021

    b) Directory `data/2022/` contains data from the year 2019-2022

    c) Directory `data/verify_dataset` contains test data -- records of unseen patients
2. Run `prepare_data.py` to prepare data for training and testing:
    - `python -m prepare_data --which [2019-2021|2022|verify_dataset] [flags]`
    - call `python -m prepare_data --help` for information about flags
    - This script will preprocess data for the models, both for models with one record and models with concatenated records
3. Use Jupyter Notebooks `models_*.ipynb` to train models and evaluate their performance
    - `models_concatenated_data.ipynb` is used for models with concatenated records
    - `models_one_entry.ipynb` is used for models with one record
