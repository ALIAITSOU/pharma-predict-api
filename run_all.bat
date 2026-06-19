@echo off
REM =================================================================
REM  PharmaPredict API - Full pipeline execution (Windows)
REM =================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===============================================================
echo   PHARMAPREDICT API - FULL PIPELINE
echo ===============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [1/6] Installing dependencies...
echo ---------------------------------------------------------------
pip install -r requirements.txt
if errorlevel 1 goto :error
echo.

echo [2/6] Loading and cleaning historical data...
echo ---------------------------------------------------------------
python -m src.data.load_data
if errorlevel 1 goto :error
echo.

echo [3/6] Building the ML training dataset...
echo ---------------------------------------------------------------
python -m src.features.build_features
if errorlevel 1 goto :error
echo.

echo [4/6] Training and evaluating models...
echo ---------------------------------------------------------------
python -m src.models.train
if errorlevel 1 goto :error
echo.

echo [5/6] Running the test suite...
echo ---------------------------------------------------------------
python -m tests.test_pipeline
if errorlevel 1 goto :error
echo.

echo [6/6] Generating the evaluation dashboard...
echo ---------------------------------------------------------------
python notebooks\evaluation.py
if errorlevel 1 goto :error
echo.

echo ===============================================================
echo   PIPELINE COMPLETE
echo ===============================================================
echo.
echo   Results available at:
echo     data\processed\training_dataset.csv
echo     models\*.pkl + metadata.json
echo     docs\evaluation_dashboard.png
echo.
echo   To start the API server, run:
echo     uvicorn src.api.main:app --reload --port 8000
echo.
echo   Then open http://localhost:8000/docs
echo.
pause
goto :eof

:error
echo.
echo ===============================================================
echo   [ERROR] A pipeline step failed. See the message above.
echo ===============================================================
pause
exit /b 1
