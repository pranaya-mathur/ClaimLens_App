#!/bin/bash

# Kill any existing uvicorn/FastAPI processes
echo "🛑 Stopping any running servers..."
pkill -f "uvicorn" || true
pkill -f "fastapi" || true
sleep 2

# Clear Python cache
echo "🧹 Clearing Python cache files..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Clear pip cache (optional)
echo "🗑️ Clearing pip cache..."
pip cache purge 2>/dev/null || true

echo "✅ Cache cleared!"
echo "🚀 Starting fresh server..."

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
