@echo off
echo [ClipTrace] Launching full stack...
start "ClipTrace Backend" cmd /k "call start_backend.bat"
timeout /t 4 /nobreak >nul
start "ClipTrace Frontend" cmd /k "call start_frontend.bat"
echo.
echo [ClipTrace] Backend  → http://localhost:8000
echo [ClipTrace] Frontend → http://localhost:3000
echo [ClipTrace] API Docs → http://localhost:8000/docs
