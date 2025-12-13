"""
Cache Package

Provides Redis-based caching for ClaimLens AI.
"""

from .redis_manager import RedisManager, get_redis_cache

__all__ = ["RedisManager", "get_redis_cache"]
