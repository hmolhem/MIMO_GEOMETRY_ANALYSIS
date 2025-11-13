@echo off
REM Run Benchmarks with Virtual Environment (Batch Version)
REM This script ensures the virtual environment is activated before running benchmarks
REM Usage: run_benchmarks_with_venv.bat [arrays] [N] [trials] [coupling]

echo ========================================
echo MIMO Array Benchmark Runner
echo ========================================
echo.

REM Parse arguments (with defaults)
set ARRAYS=%1
if "%ARRAYS%"=="" set ARRAYS=Z5
set NUM_SENSORS=%2
if "%NUM_SENSORS%"=="" set NUM_SENSORS=7
set TRIALS=%3
if "%TRIALS%"=="" set TRIALS=100
set COUPLING=%4

REM Activate virtual environment
set VENV_PATH=envs\mimo-geom-dev\Scripts\activate.bat
if exist "%VENV_PATH%" (
    echo Activating virtual environment...
    call %VENV_PATH%
    echo [32m^√ Virtual environment activated[0m
) else (
    echo [31mERROR: Virtual environment not found![0m
    echo Expected location: %VENV_PATH%
    exit /b 1
)

echo.
echo Python Environment Info:
python --version
echo Location: 
python -c "import sys; print(sys.executable)"
echo.

REM Build benchmark command
set BENCHMARK_SCRIPT=core\analysis_scripts\run_benchmarks.py
set COMMAND=python %BENCHMARK_SCRIPT% --arrays %ARRAYS% --N %NUM_SENSORS% --trials %TRIALS%

if "%COUPLING%"=="coupling" (
    set COMMAND=%COMMAND% --coupling exponential --coupling-strength 0.3
    echo [33mNote: Running WITH mutual coupling enabled[0m
)

echo Running benchmark:
echo   %COMMAND%
echo.
echo ========================================
echo.

REM Run the benchmark
%COMMAND%

echo.
echo ========================================
echo [32mBenchmark Complete![0m
echo ========================================
