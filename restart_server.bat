@echo off
REM Windows batch script to restart FastAPI server and clear cache

echo.
echo ========================================
echo Stopping any running FastAPI servers...
echo ========================================
echo.

REM Kill all Python processes
taskkill /IM python.exe /F /T 2>nul
timeout /t 2 /nobreak

echo.
echo ========================================
echo Clearing Python cache files...
echo ========================================
echo.

REM Clear __pycache__ directories
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

REM Delete .pyc files
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f" 2>nul

REM Delete .pyo files
for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f" 2>nul

echo Cache cleared successfully!
echo.
echo ========================================
echo Starting fresh FastAPI server...
echo ========================================
echo.

REM Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
