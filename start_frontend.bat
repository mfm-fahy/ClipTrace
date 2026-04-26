@echo off
echo [ClipTrace] Setting up frontend...
cd /d "%~dp0frontend"

where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js / npm not found. Install from https://nodejs.org
    pause
    exit /b 1
)

if not exist node_modules (
    echo [ClipTrace] Installing dependencies...
    npm install
)

echo [ClipTrace] Starting frontend on http://localhost:3000
npm run dev
