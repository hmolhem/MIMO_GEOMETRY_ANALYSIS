@echo off
REM Activate Virtual Environment - mimo-geom-dev
REM Usage: activate_venv.bat

echo ========================================
echo Activating mimo-geom-dev Environment
echo ========================================
echo.

set VENV_PATH=envs\mimo-geom-dev\Scripts\activate.bat

if exist "%VENV_PATH%" (
    echo Found virtual environment at: %VENV_PATH%
    echo Activating...
    echo.
    call %VENV_PATH%
    echo.
    echo Virtual environment activated successfully!
    echo.
    echo Python version:
    python --version
    echo.
    echo Python location:
    where python
    echo.
    echo Ready to run benchmarks and tests!
    echo.
) else (
    echo ERROR: Virtual environment not found!
    echo Expected location: %VENV_PATH%
    echo.
    echo Please check that the virtual environment exists.
    echo.
)
