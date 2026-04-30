@echo off
echo ======================================================
echo  NeuroAds - Brain Response Predictor Setup
echo  Python 3.12 compatible
echo ======================================================
echo.

REM Step 1: Create virtual environment using Python 3.12
echo [1/5] Creating Python 3.12 virtual environment...
py -3.12 -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.12 not found.
    echo Download it from https://python.org/downloads/
    pause & exit /b 1
)

REM Step 2: Activate venv
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 3: Upgrade pip
echo [3/5] Upgrading pip and installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Step 4: Install TRIBE v2 from GitHub
echo [4/5] Installing TRIBE v2 from GitHub...
pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"

REM Step 5: Create required runtime folders
echo [5/5] Creating runtime folders...
if not exist "uploads" mkdir uploads
if not exist "cache"   mkdir cache
if not exist "results" mkdir results
if not exist "model"   mkdir model

echo.
echo ======================================================
echo  Setup complete!
echo.
echo  To run the app:
echo     venv\Scripts\activate
echo     python app.py
echo.
echo  Then open: http://localhost:5000
echo.
echo  NOTE: The first prediction will download ~2 GB of model
echo  weights into the ./model/ folder automatically.
echo ======================================================
pause
