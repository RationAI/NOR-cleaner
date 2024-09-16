# NOR-cleaner
AI model for resolving duplicities in Czech National Oncology Register.
Bachelor's thesis about this project is available [here](https://is.muni.cz/th/lkudy/development_of_data_deduplication_model_nor.pdf).

## Requirements
- Python $\ge 3.10$

## Installation
Clone the repository.

### Windows
Run the file `install.bat`.

### Linux
Create a virtual environment and activate it (optional but recommended):
```bash
python -m venv venv
```

Activate the virtual environment:
```bash
source venv/bin/activate
```

Install the requirements:
```bash
pip install .
```

## Usage

In terminal, run the command:
```bash
nor-cleaner [-h] {prepare,train,predict,evaluate} ...
```

E.g.:
```bash
nor-cleaner prepare
```

**NOTE:** use the constants file in `scripts/constants.py` to set paths and other parameters.

### Arguments
- `prepare` - Prepare the data for training.
- `train` - Train the model.
- `predict` - Predict the duplicities.
- `evaluate` - Evaluate the model.
