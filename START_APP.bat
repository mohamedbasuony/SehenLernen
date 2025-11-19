@echo off
REM Sehen Lernen - Application Launcher
REM This script starts both the backend and frontend services

setlocal enabledelayedexpansion

cls
echo.
echo ================================================
echo   SEHEN LERNEN - APPLICATION LAUNCHER
echo ================================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run SETUP_WINDOWS.bat first to set up the application.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if backend requirements are installed
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Dependencies not found!
    echo.
    echo Please run SETUP_WINDOWS.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

echo [OK] Environment ready
echo.

REM Start Backend
echo Starting Backend Server...
start "Sehen Lernen Backend" cmd /k ^
    "call venv\Scripts\activate.bat && cd Backend && ^
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to start
timeout /t 3 /nobreak

REM Start Frontend
echo Starting Frontend Application...
start "Sehen Lernen Frontend" cmd /k ^
    "call venv\Scripts\activate.bat && cd Fronted && ^
    streamlit run app.py --server.port 8501"

REM Wait for frontend to start
timeout /t 4 /nobreak

echo.
echo ================================================
echo   SERVICES STARTED SUCCESSFULLY!
echo ================================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:8501
echo.
echo Opening application in browser...
timeout /t 1 /nobreak

REM Open browser
start http://localhost:8501

echo.
echo Keep these Command Prompt windows open while using the application.
echo To stop: Close both Command Prompt windows.
echo.
pause
