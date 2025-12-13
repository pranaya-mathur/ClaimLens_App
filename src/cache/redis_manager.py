"""
Redis Cache Manager

Provides caching layer for:
- ML fraud predictions
- Document verification results
- Graph analysis results
- API responses

Features:
- Connection pooling
- Automatic retry
- Graceful degradation if Redis unavailable
- TTL (Time To Live) support
- Health monitoring
"""

import json
import redis
from redis.connection import ConnectionPool
from typing import Any, Optional
from loguru import logger
import os


class RedisManager:
    """
    Singleton Redis manager with connection pooling.
    
    Usage:
        cache = RedisManager()
        cache.set("key", {"data": "value"}, ttl=300)
        result = cache.get("key")
    """
    
    _instance = None
    _client: Optional[redis.Redis] = None
    _pool: Optional[ConnectionPool] = None
    _available: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._client:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client with connection pool"""
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_db = int(os.getenv("REDIS_DB", "0"))
            redis_password = os.getenv("REDIS_PASSWORD", None)
            
            # Create connection pool
            self._pool = ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                max_connections=20,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            self._client.ping()
            self._available = True
            
            logger.success(f"✅ Redis connected: {redis_host}:{redis_port}")
            
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.info("👉 Caching disabled. App will work without Redis.")
            self._available = False
            self._client = None
        except Exception as e:
            logger.error(f"❌ Redis initialization error: {e}")
            self._available = False
            self._client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self._available and self._client is not None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value (parsed from JSON) or None
        """
        if not self.is_available():
            return None
        
        try:
            value = self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"⚠️ Redis GET error for key '{key}': {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (default: 300 = 5 minutes)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            serialized = json.dumps(value)
            self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis SET error for key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis DELETE error for key '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.warning(f"⚠️ Redis EXISTS error for key '{key}': {e}")
            return False
    
    def flush_all(self) -> bool:
        """
        Clear entire cache (use with caution!).
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self._client.flushdb()
            logger.warning("🗑️ Redis cache flushed")
            return True
        except Exception as e:
            logger.error(f"❌ Redis FLUSH error: {e}")
            return False
    
    def health_check(self) -> dict:
        """
        Check Redis health and stats.
        
        Returns:
            Health status dict
        """
        if not self.is_available():
            return {
                "status": "unavailable",
                "connected": False,
                "message": "Redis not connected"
            }
        
        try:
            # Test ping
            self._client.ping()
            
            # Get info
            info = self._client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "version": info.get("redis_version", "unknown"),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"❌ Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    # Domain-specific cache methods
    
    def cache_ml_prediction(self, claim_id: str, result: dict, ttl: int = 3600) -> bool:
        """
        Cache ML fraud prediction.
        
        Args:
            claim_id: Claim identifier
            result: ML prediction result
            ttl: Cache for 1 hour (default)
            
        Returns:
            True if cached successfully
        """
        key = f"ml:prediction:{claim_id}"
        return self.set(key, result, ttl)
    
    def get_ml_prediction(self, claim_id: str) -> Optional[dict]:
        """
        Get cached ML prediction.
        
        Args:
            claim_id: Claim identifier
            
        Returns:
            Cached prediction or None
        """
        key = f"ml:prediction:{claim_id}"
        return self.get(key)
    
    def cache_document_verification(self, doc_hash: str, result: dict, ttl: int = 7200) -> bool:
        """
        Cache document verification result.
        
        Args:
            doc_hash: Document hash/identifier
            result: Verification result
            ttl: Cache for 2 hours (default)
            
        Returns:
            True if cached successfully
        """
        key = f"doc:verification:{doc_hash}"
        return self.set(key, result, ttl)
    
    def get_document_verification(self, doc_hash: str) -> Optional[dict]:
        """
        Get cached document verification.
        
        Args:
            doc_hash: Document hash/identifier
            
        Returns:
            Cached verification or None
        """
        key = f"doc:verification:{doc_hash}"
        return self.get(key)
    
    def cache_graph_analysis(self, claim_id: str, result: dict, ttl: int = 1800) -> bool:
        """
        Cache graph analysis result.
        
        Args:
            claim_id: Claim identifier
            result: Graph analysis result
            ttl: Cache for 30 minutes (default)
            
        Returns:
            True if cached successfully
        """
        key = f"graph:analysis:{claim_id}"
        return self.set(key, result, ttl)
    
    def get_graph_analysis(self, claim_id: str) -> Optional[dict]:
        """
        Get cached graph analysis.
        
        Args:
            claim_id: Claim identifier
            
        Returns:
            Cached analysis or None
        """
        key = f"graph:analysis:{claim_id}"
        return self.get(key)
    
    def close(self):
        """Close Redis connection"""
        if self._client:
            try:
                self._client.close()
                logger.info("🔌 Redis connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Redis close error: {e}")


# Global instance
_redis_cache = None

def get_redis_cache() -> RedisManager:
    """
    Get global Redis cache instance (singleton).
    
    Returns:
        RedisManager instance
    """
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisManager()
    return _redis_cache
