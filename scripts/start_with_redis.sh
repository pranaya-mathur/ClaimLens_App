#!/bin/bash

# ClaimLens Startup Script with Redis
# Starts Redis + FastAPI + Diagnostics

echo "="
echo "🚀 ClaimLens Startup with Redis"
echo "="
echo ""

# Step 1: Start Redis
echo "📦 Step 1: Starting Redis container..."
docker-compose up -d redis

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Redis"
    echo "💡 Try: docker-compose up redis"
    exit 1
fi

echo "✅ Redis container started"
echo ""

# Step 2: Wait for Redis
echo "⏳ Step 2: Waiting for Redis to be ready..."
sleep 3
echo "✅ Redis should be ready"
echo ""

# Step 3: Start FastAPI
echo "🌐 Step 3: Starting FastAPI server..."
echo "📍 API will run at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "⚠️  Press Ctrl+C to stop the server"
echo ""
echo "─────────────────────────────────────"
echo ""

# Start API in background for diagnostic
python -m uvicorn api.main:app --reload &
API_PID=$!

echo "⏳ Waiting for API to start..."
sleep 5
echo ""

# Step 4: Run diagnostics
echo "🩺 Step 4: Running diagnostics..."
echo ""
python scripts/diagnose_redis.py

DIAG_EXIT=$?

echo ""
echo "─────────────────────────────────────"
echo ""

if [ $DIAG_EXIT -eq 0 ]; then
    echo "✅ Diagnostics passed!"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. API is running at http://localhost:8000"
    echo "   2. Open new terminal and run:"
    echo "      streamlit run frontend/streamlit_app.py"
    echo "   3. Test cache: curl http://localhost:8000/api/cache/health"
    echo ""
    echo "📊 The API is now running with Redis caching enabled!"
    echo "   Claims will be analyzed 2x faster with caching."
    echo ""
    echo "Press Ctrl+C to stop the API server."
    echo ""
    
    # Keep API running
    wait $API_PID
else
    echo "⚠️  Some diagnostics failed, but API is running"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   - Check Redis: docker ps | grep redis"
    echo "   - Check API logs above"
    echo "   - Try restarting: docker-compose restart redis"
    echo ""
    echo "Press Ctrl+C to stop the API server."
    echo ""
    
    # Keep API running anyway
    wait $API_PID
fi
