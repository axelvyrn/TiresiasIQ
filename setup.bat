@echo off
setlocal

REM ─── Step 1: Check for Python 3.11 ─────────────────────────────────────────
echo Checking for Python 3.11...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python not found in PATH.
    echo Please install Python 3.11 from https://www.python.org/downloads/release/python-3110/
    pause
    exit /b
)

for /f "tokens=2 delims=[]" %%V in ('python -V 2^>^&1') do set version=%%V
echo Found Python %version%

echo %version% | findstr /r "^3\.11" >nul
if %errorlevel% neq 0 (
    echo ❌ Python 3.11 is not the default version.
    echo Please install Python 3.11 and make sure it's added to PATH.
    pause
    exit /b
)

REM ─── Step 2: Create Virtual Environment ────────────────────────────────────
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM ─── Step 3: Activate Virtual Environment ──────────────────────────────────
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM ─── Step 4: Install Dependencies ──────────────────────────────────────────
echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

REM ─── Step 5: Run the App ───────────────────────────────────────────────────
echo Starting TiresiasIQ Dashboard...
python dashboard_gui.py

endlocal
pause
