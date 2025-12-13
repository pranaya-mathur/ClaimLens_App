#!/usr/bin/env python3
"""
Redis Cache Diagnostic Script

Tests:
1. Redis connection
2. Cache read/write operations
3. API cache endpoints
4. Performance benchmarks
"""

import sys
import time
import requests
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


API_BASE_URL = "http://localhost:8000"


def test_redis_connection():
    """Test Redis connection via API"""
    logger.info("🔍 Testing Redis connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/cache/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            health = data.get("health", {})
            
            if health.get("connected"):
                logger.success(f"✅ Redis connected: {health.get('version', 'unknown')}")
                logger.info(f"   - Memory used: {health.get('used_memory_human', 'N/A')}")
                logger.info(f"   - Clients: {health.get('connected_clients', 0)}")
                logger.info(f"   - Uptime: {health.get('uptime_in_seconds', 0)}s")
                return True
            else:
                logger.warning("⚠️ Redis not connected")
                logger.info("   - App will work without cache (slower)")
                return False
        else:
            logger.error(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to API. Is it running?")
        logger.info("👉 Start API: python -m uvicorn api.main:app --reload")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_cache_operations():
    """Test cache read/write via API"""
    logger.info("🔍 Testing cache operations...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/cache/test", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "success":
                logger.success("✅ Cache read/write working")
                return True
            elif data.get("status") == "disabled":
                logger.warning("⚠️ Cache disabled (Redis not available)")
                return False
            else:
                logger.error(f"❌ Cache test failed: {data.get('message')}")
                return False
        else:
            logger.error(f"❌ Cache test endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_cache_stats():
    """Get cache statistics"""
    logger.info("🔍 Fetching cache stats...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/cache/stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("enabled"):
                logger.success("✅ Cache stats retrieved")
                logger.info(f"   - Status: {data.get('status')}")
                logger.info(f"   - Total commands: {data.get('total_commands', 0):,}")
                logger.info(f"   - Memory: {data.get('memory_used', 'N/A')}")
                return True
            else:
                logger.warning("⚠️ Cache not enabled")
                return False
        else:
            logger.error(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def benchmark_cache_performance():
    """Benchmark cache performance"""
    logger.info("🔍 Benchmarking cache performance...")
    
    # Check if Redis is available first
    health_response = requests.get(f"{API_BASE_URL}/api/cache/health", timeout=5)
    if not health_response.json().get("health", {}).get("connected"):
        logger.warning("⚠️ Skipping benchmark (Redis not connected)")
        return False
    
    try:
        # Measure cache test response time
        start = time.time()
        response = requests.get(f"{API_BASE_URL}/api/cache/test", timeout=5)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        if response.status_code == 200:
            logger.success(f"✅ Cache operation: {elapsed:.2f}ms")
            
            if elapsed < 50:
                logger.success("   🚀 Excellent performance!")
            elif elapsed < 100:
                logger.info("   ✅ Good performance")
            else:
                logger.warning("   ⚠️ Slower than expected")
            
            return True
        else:
            logger.error("❌ Benchmark failed")
            return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_api_root():
    """Test API root endpoint"""
    logger.info("🔍 Testing API root endpoint...")
    try:
        response = requests.get(API_BASE_URL, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if cache endpoint is listed
            if "cache" in data.get("endpoints", {}):
                logger.success("✅ Cache endpoints registered in API")
                return True
            else:
                logger.warning("⚠️ Cache endpoints not found in root")
                return False
        else:
            logger.error(f"❌ API root failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def main():
    """Run all diagnostic tests"""
    logger.info("="*60)
    logger.info("🩺 CLAIMLENS REDIS DIAGNOSTIC")
    logger.info("="*60)
    logger.info("")
    
    results = {
        "API Root": test_api_root(),
        "Redis Connection": test_redis_connection(),
        "Cache Operations": test_cache_operations(),
        "Cache Stats": test_cache_stats(),
        "Performance Benchmark": benchmark_cache_performance()
    }
    
    logger.info("")
    logger.info("="*60)
    logger.info("📊 DIAGNOSTIC SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status} - {test_name}")
    
    logger.info("")
    logger.info(f"🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("✨ ALL TESTS PASSED! Redis fully integrated! ✨")
        logger.info("")
        logger.info("🚀 Next steps:")
        logger.info("   1. Run Streamlit: streamlit run frontend/streamlit_app.py")
        logger.info("   2. Test claim analysis (should be 2x faster with cache)")
        logger.info("   3. Check cache stats: curl http://localhost:8000/api/cache/stats")
        return 0
    elif passed >= 3:
        logger.warning("⚠️ Some tests failed, but core functionality works")
        return 0
    else:
        logger.error("❌ Multiple failures detected. Check API and Redis.")
        logger.info("")
        logger.info("🔧 Troubleshooting:")
        logger.info("   1. Start Redis: docker-compose up redis -d")
        logger.info("   2. Start API: python -m uvicorn api.main:app --reload")
        logger.info("   3. Check logs for errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
