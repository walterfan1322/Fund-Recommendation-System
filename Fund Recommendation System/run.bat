@echo off
echo Starting Fund Recommendation System...
cd /d "%~dp0"

REM Check if venv exists
if not exist .venv (
    echo Virtual environment not found. Installing dependencies...
    python -m venv .venv
    ".venv\Scripts\pip" install -r requirements.txt
)

REM Check if streamlit is installed
if not exist ".venv\Scripts\streamlit.exe" (
    echo Streamlit not found. Installing dependencies...
    ".venv\Scripts\pip" install -r requirements.txt
)

echo Launching App...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
