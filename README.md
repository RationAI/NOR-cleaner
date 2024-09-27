# NOR-cleaner

<a href="https://github.com/RationAI/NOR-cleaner/archive/refs/heads/main.zip">
    <img src="https://img.shields.io/badge/Code-Download%20ZIP-green" alt="Download ZIP" style="display: inline-block; margin: 0; padding: 0;"/>
</a>

AI model for resolving duplicities in Czech National Oncology Register.
Bachelor's thesis about this project is available [here](https://is.muni.cz/th/lkudy/development_of_data_deduplication_model_nor.pdf).

## Table of contents
- [Requirements](#requirements)
- [Installation](#installation)
  - [Windows](#windows)
  - [Linux](#linux)
- [How to use](#how-to-use)
    - [Edit paths](#edit-paths)
    - [Run the scripts](#run-the-scripts)
        - [Order of execution](#order-of-execution)

## Requirements
- Python $\ge 3.10$

## Installation
Install the package either by:

- Downloading the repository as a ZIP file by clicking the on the badge at the beginning of this `README` file.
- Cloning the repository

### Windows
Run the file `install.ps1` either by right-clicking and selecting `Run with PowerShell` or by running the command in PowerShell:
```powershell
.\install.ps1
```

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

## How to use

### Edit paths
Edit the paths in the `scripts/constants.py` file, e.g. paths to the data or model.

### Run the scripts
In terminal, run the command:
```bash
nor-cleaner [-h] {prepare,train,predict,evaluate} ...
```

#### Order of execution
1. Prepare the data for training.
```bash
nor-cleaner prepare
```
2. Train the model.
```bash
nor-cleaner train
```
3. Predict whether to preserve or drop a record.
```bash
nor-cleaner predict
```
4. (Optional) Evaluate the model using cross-validation on the training data.
```bash
nor-cleaner evaluate
```

**NOTE:** use the constants file in `scripts/constants.py` to set paths.