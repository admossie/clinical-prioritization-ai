@echo off
set PYTHON_EXE=python
if exist .venv\Scripts\python.exe set PYTHON_EXE=.venv\Scripts\python.exe

if not exist models\best_model.joblib (
	echo Missing models\best_model.joblib. Train the pipeline first with:
	echo   %PYTHON_EXE% -m src.train --data-path data\raw\diabetic_data.csv
	exit /b 1
)

%PYTHON_EXE% -m streamlit run app\streamlit_app.py
