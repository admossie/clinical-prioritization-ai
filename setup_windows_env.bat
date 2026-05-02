@echo off
setlocal

set PY_LAUNCHER=py
where %PY_LAUNCHER% >nul 2>nul
if errorlevel 1 (
    echo Python launcher 'py' was not found on PATH.
    echo Install Python 3.11 or 3.10 from python.org and select "Add python.exe to PATH".
    exit /b 1
)

set TARGET_PY=
%PY_LAUNCHER% -3.11 --version >nul 2>nul
if not errorlevel 1 set TARGET_PY=3.11
if "%TARGET_PY%"=="" (
    %PY_LAUNCHER% -3.10 --version >nul 2>nul
    if not errorlevel 1 set TARGET_PY=3.10
)

if "%TARGET_PY%"=="" (
    echo Python 3.11 or 3.10 is required but not installed.
    echo Install one of them, then rerun this script.
    exit /b 1
)

echo Using Python %TARGET_PY% to create virtual environment...
if exist .venv (
    echo Removing existing .venv...
    rmdir /s /q .venv
)

%PY_LAUNCHER% -%TARGET_PY% -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

set VENV_PY=.venv\Scripts\python.exe

%VENV_PY% -m pip install --upgrade pip
if errorlevel 1 exit /b 1

%VENV_PY% -m pip install -r requirements.txt -r requirements-dev.txt
if errorlevel 1 exit /b 1

echo Running validation checks...
%VENV_PY% --version
if errorlevel 1 exit /b 1

%VENV_PY% -m black --check app api src tests outputs
if errorlevel 1 exit /b 1

%VENV_PY% -m flake8 app api src tests outputs
if errorlevel 1 exit /b 1

%VENV_PY% -m pytest -q
if errorlevel 1 exit /b 1

%VENV_PY% -m bandit -q -r src api app
if errorlevel 1 exit /b 1

%VENV_PY% -m pip_audit -r requirements.txt
if errorlevel 1 exit /b 1

echo Environment setup and validation completed successfully.
exit /b 0
