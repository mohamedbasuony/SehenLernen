@echo off
REM Sehen Lernen - Automated Setup for Windows
REM This script sets up the application for first-time use

setlocal enabledelayedexpansion

cls
echo.
echo ================================================
echo   SEHEN LERNEN - SETUP WIZARD
echo ================================================
echo.

REM Check if Python is installed
echo Checking for Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.9 or higher is required but not found!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo.
    echo IMPORTANT: When installing, check the box "Add Python to PATH"
    echo.
    echo After installing Python, run this script again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found: %PYTHON_VERSION%
echo.

REM Create virtual environment
echo Creating isolated Python environment...
if exist venv (
    echo [OK] Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python packages (this may take 5-10 minutes)...
echo Please be patient and do not close this window...
echo.

pip install --upgrade pip >nul 2>&1

if exist Backend\requirements.txt (
    echo Installing backend dependencies...
    pip install -q -r Backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install backend dependencies
        pause
        exit /b 1
    )
    echo [OK] Backend dependencies installed
)

if exist Fronted\requirements.txt (
    echo Installing frontend dependencies...
    pip install -q -r Fronted\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies
        pause
        exit /b 1
    )
    echo [OK] Frontend dependencies installed
)

echo.
echo ================================================
echo   SETUP COMPLETE!
echo ================================================
echo.
echo To start the application:
echo   1. Run: START_APP.bat
echo      OR
echo   2. Run this command:
echo      venv\Scripts\activate.bat
echo      Then follow the prompts
echo.
echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:8501
echo.
pause
