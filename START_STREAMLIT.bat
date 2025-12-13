@echo off
REM Streamlit Frontend Startup Script for ClaimLens AI

echo.
echo ======================================================================
echo                      CLAIMLENS STREAMLIT FRONTEND
echo ======================================================================
echo.

REM Check if backend API is running
echo [STEP 1/2] Checking if backend API is running...
curl -s http://localhost:8000/health/liveness >nul 2>&1

if %errorlevel% neq 0 (
    echo.
    echo *** WARNING: Backend API is NOT running! ***
    echo.
    echo Please start the FastAPI backend first:
    echo   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    echo.
    echo Press any key to continue anyway (Streamlit will start but API calls will fail)...
    pause
)

echo Backend API: RUNNING ✓
echo.

echo [STEP 2/2] Starting Streamlit frontend...
echo.
echo ======================================================================
echo  Streamlit will open in your browser at: http://localhost:8501
echo  Press CTRL+C to stop the server
echo ======================================================================
echo.

REM Start Streamlit
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address localhost

pause
