@REM Script for installing the application on Windows

@REM Create virtual environment
python -m venv venv

@REM Activate virtual environment
.\venv\Scripts\Activate.ps1

@REM Install dependencies
pip install .
