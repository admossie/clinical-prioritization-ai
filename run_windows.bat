@echo off
set PYTHON_EXE=python
if exist .venv\Scripts\python.exe set PYTHON_EXE=.venv\Scripts\python.exe

%PYTHON_EXE% -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)"
if errorlevel 1 (
	echo Python 3.10+ is required by this project.
	echo Current interpreter: %PYTHON_EXE%
	echo Install Python 3.10 or 3.11, then recreate .venv.
	echo Recommended bootstrap command:
	echo   setup_windows_env.bat
	exit /b 1
)

if not exist models\best_model.joblib (
	echo Missing models\best_model.joblib. Train the pipeline first with:
	echo   %PYTHON_EXE% -m src.train --data-path data\raw\diabetic_data.csv
	exit /b 1
)

echo Starting AI Care Prioritization Engine...
%PYTHON_EXE% -m streamlit run app.py
