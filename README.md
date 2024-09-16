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

### Windows
Run the file `run.bat`.

### Linux
Run the command:
```bash
nor-cleaner run
```
For more information, run:
```bash
nor-cleaner -h
```
