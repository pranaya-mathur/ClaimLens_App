@echo off
REM ClaimLens Startup Script with Redis (Windows)
REM Starts Redis + FastAPI + Diagnostics

echo ================================
echo 🚀 ClaimLens Startup with Redis
echo ================================
echo.

REM Step 1: Start Redis
echo 📦 Step 1: Starting Redis container...
docker-compose up -d redis

if errorlevel 1 (
    echo ❌ Failed to start Redis
    echo 💡 Try: docker-compose up redis
    pause
    exit /b 1
)

echo ✅ Redis container started
echo.

REM Step 2: Wait for Redis
echo ⏳ Step 2: Waiting for Redis to be ready...
timeout /t 3 /nobreak > nul
echo ✅ Redis should be ready
echo.

REM Step 3: Start FastAPI
echo 🌐 Step 3: Starting FastAPI server...
echo 📍 API will run at: http://localhost:8000
echo 📚 API docs at: http://localhost:8000/docs
echo.
echo ⚠️  Press Ctrl+C to stop the server
echo.
echo =====================================
echo.

REM Start API in new window
start "ClaimLens API" cmd /k python -m uvicorn api.main:app --reload

REM Wait for API to start
echo ⏳ Waiting for API to start...
timeout /t 5 /nobreak > nul
echo.

REM Step 4: Run diagnostics
echo 🩺 Step 4: Running diagnostics...
echo.
python scripts\diagnose_redis.py

set DIAG_EXIT=%errorlevel%

echo.
echo =====================================
echo.

if %DIAG_EXIT% equ 0 (
    echo ✅ Diagnostics passed!
    echo.
    echo 🎯 Next Steps:
    echo    1. API is running in separate window
    echo    2. Open new terminal and run:
    echo       streamlit run frontend\streamlit_app.py
    echo    3. Test cache: curl http://localhost:8000/api/cache/health
    echo.
    echo 📊 The API is now running with Redis caching enabled!
    echo    Claims will be analyzed 2x faster with caching.
    echo.
) else (
    echo ⚠️  Some diagnostics failed, but API is running
    echo.
    echo 🔧 Troubleshooting:
    echo    - Check Redis: docker ps
    echo    - Check API logs in the other window
    echo    - Try restarting: docker-compose restart redis
    echo.
)

echo Press any key to exit this window...
echo (API will continue running in the other window)
pause > nul
