@echo off
echo [ClipTrace] Setting up backend...
cd /d "%~dp0backend"

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

echo [ClipTrace] Starting backend on http://localhost:8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 120
