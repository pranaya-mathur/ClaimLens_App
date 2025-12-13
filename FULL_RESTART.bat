@echo off
REM Full system restart script for Windows - NUCLEAR OPTION
REM This completely clears Python cache and restarts fresh

echo.
echo ======================================================================
echo                    FULL SYSTEM RESTART FOR CLAIMLENS
echo ======================================================================
echo.
echo This will:
echo 1. Kill ALL Python processes
echo 2. Clear ALL __pycache__ directories
echo 3. Clear ALL .pyc files
echo 4. Clear pip cache
echo 5. Delete Python cache in temp folders
echo 6. Restart server fresh
echo.
pause

echo.
echo [STEP 1/5] Killing ALL Python processes...
echo.
taskkill /IM python.exe /F /T 2>nul
taskkill /IM pythonw.exe /F /T 2>nul
timeout /t 3 /nobreak

echo.
echo [STEP 2/5] Clearing __pycache__ directories...
echo.
for /d /r . %%d in (__pycache__) do @if exist "%%d" (
    echo Deleting: %%d
    rd /s /q "%%d" 2>nul
)

echo.
echo [STEP 3/5] Clearing .pyc and .pyo files...
echo.
for /r . %%f in (*.pyc) do @if exist "%%f" (
    echo Deleting: %%f
    del /q "%%f" 2>nul
)
for /r . %%f in (*.pyo) do @if exist "%%f" (
    echo Deleting: %%f
    del /q "%%f" 2>nul
)

echo.
echo [STEP 4/5] Clearing pip and Python cache...
echo.
pip cache purge 2>nul
if exist %TEMP%\pip rd /s /q %TEMP%\pip 2>nul
if exist %APPDATA%\Python rd /s /q %APPDATA%\Python 2>nul

echo.
echo [STEP 5/5] Starting fresh FastAPI server...
echo.
echo ======================================================================
echo All caches cleared! Server starting fresh...
echo ======================================================================
echo.

uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
