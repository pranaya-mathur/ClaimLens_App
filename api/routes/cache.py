"""
Cache API Routes - Redis Health & Management
"""
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.cache import get_redis_cache


router = APIRouter()


@router.get("/health")
async def cache_health():
    """
    Check Redis cache health.
    
    Returns:
        Redis status and connection info
    """
    try:
        cache = get_redis_cache()
        health = cache.health_check()
        
        return {
            "cache_type": "redis",
            "health": health,
            "enabled": cache.is_available()
        }
    except Exception as e:
        logger.error(f"Cache health check error: {e}")
        return {
            "cache_type": "redis",
            "health": {
                "status": "error",
                "error": str(e)
            },
            "enabled": False
        }


@router.get("/stats")
async def cache_stats():
    """
    Get cache usage statistics.
    
    Returns:
        Cache performance stats
    """
    try:
        cache = get_redis_cache()
        
        if not cache.is_available():
            return {
                "enabled": False,
                "message": "Redis cache not available"
            }
        
        health = cache.health_check()
        
        return {
            "enabled": True,
            "status": health.get("status"),
            "connected_clients": health.get("connected_clients", 0),
            "total_commands": health.get("total_commands_processed", 0),
            "uptime_seconds": health.get("uptime_in_seconds", 0),
            "memory_used": health.get("used_memory_human", "unknown"),
            "version": health.get("version", "unknown")
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flush")
async def flush_cache():
    """
    Flush entire cache (use with caution!).
    
    **Admin only** - Clears all cached data.
    
    Returns:
        Success status
    """
    try:
        cache = get_redis_cache()
        
        if not cache.is_available():
            raise HTTPException(
                status_code=503,
                detail="Redis cache not available"
            )
        
        success = cache.flush_all()
        
        if success:
            return {
                "status": "success",
                "message": "Cache flushed successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to flush cache"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache flush error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test")
async def test_cache():
    """
    Test cache read/write operations.
    
    Returns:
        Test results
    """
    try:
        cache = get_redis_cache()
        
        if not cache.is_available():
            return {
                "status": "disabled",
                "message": "Redis cache not available"
            }
        
        # Test write
        test_key = "test:cache:ping"
        test_value = {"message": "Hello from ClaimLens", "timestamp": "2025-12-13"}
        
        write_success = cache.set(test_key, test_value, ttl=60)
        
        if not write_success:
            return {
                "status": "error",
                "message": "Failed to write to cache"
            }
        
        # Test read
        read_value = cache.get(test_key)
        
        if read_value == test_value:
            # Cleanup
            cache.delete(test_key)
            
            return {
                "status": "success",
                "message": "Cache read/write working correctly",
                "test_passed": True
            }
        else:
            return {
                "status": "error",
                "message": "Cache read/write mismatch",
                "test_passed": False
            }
    except Exception as e:
        logger.error(f"Cache test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
